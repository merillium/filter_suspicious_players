import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

from enums import Folders

## columns saved for each game (with player and time_control) to the player games parquet file
PLAYER_GAMES_COLUMNS = [
    "utc_datetime",
    "ratings",
    "opponent_ratings",
    "actual_scores",
    "rating_gains",
    "estimated_rd",
    "performance_difference",
]


def make_player_features(CSV_RAW_FEATURES_FILE_PATH):
    """Creates features at the player + time control level from the CSV file containing raw features."""

    all_player_games_df = pd.read_csv(
        CSV_RAW_FEATURES_FILE_PATH,
        index_col=[0, 1],
    )
    all_player_games_df.index = all_player_games_df.index.set_names(
        ["player", "time_control"]
    )

    ## filter out users who have not played enough games
    MIN_GAMES = 30
    all_player_games_filtered_df = all_player_games_df.loc[
        all_player_games_df.groupby(level=["player", "time_control"]).size()
        >= MIN_GAMES
    ].copy()

    ## calculate how much someone exceeds expectations: (actual win rate - expected win rate)
    ## someone who has a high win rate could just play mostly lower rated opposition

    ## this is more involved and requires figuring out expected scores for each game
    ## e.g. if player 1 is 1500 and player 2 is also 1500, player 1 should have an expected score of 0.5
    ## but the exact nature of the curve depends on the glicko-2 rating system

    ## we use will the following paper: http://www.glicko.net/glicko/glicko2.pdf
    ## and this comment left by @chess_in_sgv:

    # Let P2 = Expected outcome for player 2. Then:
    # P2 = 1 / (1 + e^-a)
    # with a = g(sqrt(r12+r22)) * (s2-s1))
    # and g(x) = 1/sqrt(1+3x2/pi2)

    ##  source: https://www.reddit.com/r/chess/comments/i0pnv1/comment/fzrhhwi

    ## ratings and RDs must first be converted to the glicko-2 scale by dividing by 400 / ln(10),
    ## otherwise the expected score is far too extreme (e.g. 0.83 instead of ~0.63 for a 100 point favorite)
    GLICKO2_SCALE = 400 / np.log(10)

    def g(x):
        return 1 / np.sqrt(1 + 3 * x**2 / np.pi**2)

    def get_player_expected_score(
        player_rating, opponent_rating, player_rd=80.0, opponent_rd=80.0
    ):
        """Returns expected score of player based on player rating, opponent rating, and RDs (if known)."""
        combined_rd = np.sqrt(player_rd**2 + opponent_rd**2) / GLICKO2_SCALE
        A = g(combined_rd) * (player_rating - opponent_rating) / GLICKO2_SCALE
        return 1 / (1 + np.exp(-A))

    all_player_games_filtered_df["expected_scores"] = get_player_expected_score(
        player_rating=all_player_games_filtered_df["ratings"].to_numpy(),
        opponent_rating=all_player_games_filtered_df["opponent_ratings"].to_numpy(),
    )

    ## how much better did a player perform than expected?
    all_player_games_filtered_df["performance_difference"] = (
        all_player_games_filtered_df["actual_scores"]
        - all_player_games_filtered_df["expected_scores"]
    )

    ## RD is not included in the lichess game data, so we estimate each player's RD before each game
    ## from the size of their rating change, by inverting the glicko-1 update (with q = 1 / GLICKO2_SCALE):
    ##     rating_gain = q / (1/RD^2 + 1/d^2) * g(RD_opp) * (score - E),  where 1/d^2 = q^2 * g(RD_opp)^2 * E * (1 - E)
    ## e.g. +10 for beating an equal opponent gives RD ~60 (established), +150 gives RD ~290 (new account)
    ## the estimate assumes the opponent's RD is 80, and is left empty when the result was close to expected,
    ## because the rating change is then too small and too heavily rounded to be informative
    def estimate_player_rd(
        player_rating, opponent_rating, score, rating_gain, opponent_rd=80.0
    ):
        """Returns the estimated RD of a player before a game based on their rating change, or nan."""
        q = 1 / GLICKO2_SCALE
        g_opponent = g(opponent_rd * q)
        expected_score = 1 / (
            1 + np.exp(-g_opponent * (player_rating - opponent_rating) * q)
        )
        score_surprise = score - expected_score
        with np.errstate(divide="ignore", invalid="ignore"):
            K = rating_gain / (q * g_opponent * score_surprise)
            inverse_rd_squared = 1 / K - q**2 * g_opponent**2 * expected_score * (
                1 - expected_score
            )
            estimated_rd = 1 / np.sqrt(inverse_rd_squared)
        is_valid = (np.abs(score_surprise) > 0.1) & (K > 0) & (inverse_rd_squared > 0)
        return np.where(is_valid, estimated_rd, np.nan)

    all_player_games_filtered_df["estimated_rd"] = estimate_player_rd(
        player_rating=all_player_games_filtered_df["ratings"].to_numpy(),
        opponent_rating=all_player_games_filtered_df["opponent_ratings"].to_numpy(),
        score=all_player_games_filtered_df["actual_scores"].to_numpy(),
        rating_gain=all_player_games_filtered_df["rating_gains"].to_numpy(),
    )

    ## AGGREGATE GAME RESULTS FEATURES by player + time control
    all_player_features = all_player_games_filtered_df.groupby(
        level=["player", "time_control"]
    ).agg(
        number_of_games=("ratings", "count"),
        mean_perf_diff=(
            "performance_difference",
            "mean",
        ),  # this takes into account opponent strength
        std_perf_diff=(
            "performance_difference",
            "std",
        ),  # this takes into account opponent strength
        mean_rating=("ratings", "mean"),
        median_rating=("ratings", "median"),
        std_rating=("ratings", "std"),
        mean_opponent_rating=("opponent_ratings", "mean"),
        # median_opponent_rating=('opponent_ratings', 'median'),
        # probably not needed, it's unlikely that opponent ratings will be skewed or provisional
        std_opponent_rating=("opponent_ratings", "std"),
        mean_rating_gain=("rating_gains", "mean"),
        std_rating_gain=("rating_gains", "std"),
        proportion_increment_games=("increments", "mean"),
        median_estimated_rd=("estimated_rd", "median"),
    )

    ## players new to the rating pool (high RD) start at 1500 and move quickly towards their true rating,
    ## e.g. a master who starts at 1500 overperforms in every game on the way up, and a beginner underperforms
    ## on the way down, so we also aggregate settled (RD < 80) and new (RD >= 80) games separately
    ## games without an RD estimate are excluded from both groups
    MAX_SETTLED_RD = 80.0
    for rd_group, rd_mask in {
        "settled": all_player_games_filtered_df["estimated_rd"] < MAX_SETTLED_RD,
        "new": all_player_games_filtered_df["estimated_rd"] >= MAX_SETTLED_RD,
    }.items():
        rd_group_features = (
            all_player_games_filtered_df.loc[
                rd_mask, ["ratings", "performance_difference", "rating_gains"]
            ]
            .groupby(level=["player", "time_control"])
            .agg(
                **{
                    f"number_of_{rd_group}_games": ("ratings", "count"),
                    f"mean_rating_{rd_group}": ("ratings", "mean"),
                    f"mean_perf_diff_{rd_group}": ("performance_difference", "mean"),
                    f"mean_rating_gain_{rd_group}": ("rating_gains", "mean"),
                }
            )
        )
        all_player_features = all_player_features.join(rd_group_features)
        all_player_features[f"number_of_{rd_group}_games"] = (
            all_player_features[f"number_of_{rd_group}_games"].fillna(0).astype(int)
        )

    ## some useful red flags for suspicious behavior:
    # (1) consistently performing above expectation
    # (i.e. mean performance difference far from 0.00 with low standard deviation performance difference)
    # we may refine this to drop the low standard deviation performance difference condition
    # (2) high correlation between increment and expectation -- not yet implemented
    # players who perform much better when playing increment are potentially suspicious
    # but there are players who are not that fast with a mouse
    # (3) high proportion of losses on time -- not yet implemented
    # not conclusive by itself, but certainly supporting evidence
    # most players don't want to lose!
    # (4) analysis of move times -- not yet implemented (unknown if such data is available)

    min_rating, max_rating = (
        all_player_features["mean_rating"].min(),
        all_player_features["mean_rating"].max(),
    )
    min_bin_rating = np.floor(all_player_features["mean_rating"].min() / 100.0) * 100
    max_bin_rating = (
        100 + np.ceil(all_player_features["mean_rating"].max() / 100.0) * 100
    )
    rating_bins = np.arange(min_bin_rating, max_bin_rating, 100)

    ## assign rating bin string to each mean_rating
    rating_bin_labels = [f"{str(int(x))} - {str(int(x)+100)}" for x in rating_bins[:-1]]
    all_player_features["rating_bin"] = pd.cut(
        all_player_features["mean_rating"],
        rating_bins,
        right=True,
        labels=rating_bins[:-1],
    ).astype(int)

    ## save to csv
    BASE_FILE_NAME = Path(CSV_RAW_FEATURES_FILE_PATH).stem.split(".")[0]
    all_player_features.to_csv(
        f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}_player_features.csv"
    )

    ## save the individual games with their timestamps and estimated RDs, used for player rating graphs
    ## as parquet, because the csv of all games is too slow to reload (older csv files have no timestamps)
    if "utc_datetime" in all_player_games_filtered_df.columns:
        all_player_games_filtered_df["utc_datetime"] = pd.to_datetime(
            all_player_games_filtered_df["utc_datetime"], format="%Y-%m-%d %H:%M:%S"
        )
        all_player_games_filtered_df[PLAYER_GAMES_COLUMNS].reset_index().to_parquet(
            f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}_player_games.parquet",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create player features from CSV file")
    parser.add_argument(
        "CSV_RAW_FEATURES_FILE_PATH", type=str, help="Path to the CSV file"
    )
    args = parser.parse_args()

    ## create features from the CSV file
    make_player_features(args.CSV_RAW_FEATURES_FILE_PATH)
