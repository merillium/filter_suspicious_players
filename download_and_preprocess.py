import argparse
import os
import subprocess
import time
import duckdb
import requests

from enums import Folders

HF_DATASET = "Lichess/standard-chess-games"
HF_TREE_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/main/data"

# only these columns are fetched from the parquet files -- movetext is never downloaded
HEADER_COLUMNS = [
    "Event",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "TimeControl",
    "UTCDate",
    "UTCTime",
]
MAX_RETRIES = 8

GAMES_QUERY = """
WITH games AS (
    SELECT
        CASE
            WHEN Event ILIKE '%bullet%' THEN 'bullet'
            WHEN Event ILIKE '%blitz%' THEN 'blitz'
            WHEN Event ILIKE '%rapid%' THEN 'rapid'
            -- older games label the classical rating pool as "Standard"
            WHEN Event ILIKE '%classical%' OR Event ILIKE '%standard%' THEN 'classical'
            ELSE 'other'
        END AS time_control,
        White, Black, WhiteElo, BlackElo, WhiteRatingDiff, BlackRatingDiff,
        CASE Result WHEN '1-0' THEN 1.0 WHEN '1/2-1/2' THEN 0.5 ELSE 0.0 END AS white_score,
        -- TimeControl is "base+increment", e.g. "180+2"
        CASE WHEN split_part(TimeControl, '+', 2) IN ('', '0') THEN 0 ELSE 1 END AS increments,
        UTCDate + CAST(UTCTime AS TIME) AS utc_datetime
    FROM read_parquet('{parquet_path}')
    -- skip games with unknown players, ratings, rating difference, or result
    -- if either opponent has not played rated games, their rating is 1500
    -- but a rating difference is not calculated because this rating is misleading
    WHERE White IS NOT NULL AND Black IS NOT NULL
        AND White NOT LIKE '%?%' AND Black NOT LIKE '%?%'
        AND WhiteElo IS NOT NULL AND BlackElo IS NOT NULL
        AND WhiteRatingDiff IS NOT NULL AND BlackRatingDiff IS NOT NULL
        AND Result IN ('1-0', '0-1', '1/2-1/2')
),
player_games AS (
    SELECT White AS player, time_control, WhiteElo AS ratings, BlackElo AS opponent_ratings,
        white_score AS actual_scores, WhiteRatingDiff AS rating_gains, increments, utc_datetime
    FROM games
    UNION ALL
    SELECT Black AS player, time_control, BlackElo AS ratings, WhiteElo AS opponent_ratings,
        1.0 - white_score AS actual_scores, BlackRatingDiff AS rating_gains, increments, utc_datetime
    FROM games
)
SELECT player, time_control, ratings, opponent_ratings, actual_scores, rating_gains, increments, utc_datetime
FROM player_games
-- exclude a player's leading games at a rating of exactly 1500 as these could be first games
QUALIFY SUM(CASE WHEN ratings != 1500 THEN 1 ELSE 0 END) OVER (
    PARTITION BY player, time_control ORDER BY utc_datetime
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) > 0
ORDER BY player, time_control, utc_datetime
"""


def list_parquet_files(year, month):
    """Returns the paths of the parquet files for one month of the Hugging Face dataset."""
    response = requests.get(f"{HF_TREE_API_URL}/year={year}/month={month}")
    if response.status_code == 404:
        raise ValueError(f"No data for {year}-{month} in the {HF_DATASET} dataset.")
    response.raise_for_status()
    return sorted(f["path"] for f in response.json() if f["path"].endswith(".parquet"))


def download_header_columns(con, remote_path, local_path):
    """Copies only HEADER_COLUMNS from one remote parquet file to a local parquet file,
    retrying with exponential backoff if Hugging Face rate limits the request."""
    columns = ", ".join(HEADER_COLUMNS)
    tmp_path = f"{local_path}.tmp"
    for attempt in range(MAX_RETRIES):
        try:
            con.execute(
                f"COPY (SELECT {columns} FROM read_parquet('hf://datasets/{HF_DATASET}/{remote_path}')) "
                f"TO '{tmp_path}' (FORMAT parquet)"
            )
            os.replace(tmp_path, local_path)
            return
        except duckdb.HTTPException as e:
            if "429" not in str(e) or attempt == MAX_RETRIES - 1:
                raise
            wait_seconds = 2 ** (attempt + 2)
            print(f"Rate limited by Hugging Face, retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def download_and_parse_games(year, month, source):
    """Downloads only the header columns needed for player features from the Lichess open database
    parquet files on Hugging Face, and writes one row per (player, game) to a csv file."""
    if source != "lichess-open-database":
        raise ValueError(
            "Source must be lichess-open-database. Support for additional sources will be added in the future."
        )

    year = str(year)
    month = str(month).zfill(2)
    BASE_FILE_NAME = f"lichess_db_standard_rated_{year}-{month}"
    CSV_RAW_FEATURES_FILE_PATH = (
        f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}.csv"
    )
    # header-only parquet files are cached here, so an interrupted download can be resumed
    PARQUET_FOLDER = f"{Folders.LICHESS_DOWNLOADED_GAMES.value}/{BASE_FILE_NAME}"
    os.makedirs(Folders.LICHESS_PLAYER_DATA.value, exist_ok=True)
    os.makedirs(PARQUET_FOLDER, exist_ok=True)

    con = duckdb.connect()
    # an access token raises Hugging Face rate limits: https://huggingface.co/settings/tokens
    # use the HF_TOKEN environment variable if set, otherwise the token saved by `huggingface-cli login`
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        con.execute(f"CREATE SECRET hf_token (TYPE huggingface, TOKEN '{hf_token}')")
    else:
        con.execute("CREATE SECRET hf_token (TYPE huggingface, PROVIDER credential_chain)")

    remote_paths = list_parquet_files(year, month)
    for i, remote_path in enumerate(remote_paths, start=1):
        local_path = f"{PARQUET_FOLDER}/{os.path.basename(remote_path)}"
        if os.path.exists(local_path):
            continue
        print(f"Downloading header columns from file {i} of {len(remote_paths)}...")
        download_header_columns(con, remote_path, local_path)

    print("Parsing games...")
    query = GAMES_QUERY.format(parquet_path=f"{PARQUET_FOLDER}/*.parquet")
    con.execute(f"COPY ({query}) TO '{CSV_RAW_FEATURES_FILE_PATH}' (HEADER)")
    print(f"Saved player games to {CSV_RAW_FEATURES_FILE_PATH}")

    return CSV_RAW_FEATURES_FILE_PATH


def preprocess_data(CSV_RAW_FEATURES_FILE_PATH):
    """This function calls make_player_features.py and make_exploratory_plots.py with csv filepaths"""

    subprocess.run(["python3", "make_player_features.py", CSV_RAW_FEATURES_FILE_PATH])

    ## make exploratory plots
    CSV_PLAYER_FEATURE_FILE_PATH = CSV_RAW_FEATURES_FILE_PATH.replace(
        ".csv", "_player_features.csv"
    )
    subprocess.run(
        ["python3", "make_exploratory_plots.py", CSV_PLAYER_FEATURE_FILE_PATH]
    )


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Lichess data")
    parser.add_argument("--year", type=int, help="Year of the data", required=True)
    parser.add_argument("--month", type=int, help="Month of the data", required=True)
    parser.add_argument(
        "--source",
        type=str,
        help="Source of the data to download",
        choices=["lichess-open-database"],
        default="lichess-open-database",
    )
    args = parser.parse_args()

    CSV_RAW_FEATURES_FILE_PATH = download_and_parse_games(
        args.year, args.month, args.source
    )
    preprocess_data(CSV_RAW_FEATURES_FILE_PATH)


if __name__ == "__main__":
    main()
