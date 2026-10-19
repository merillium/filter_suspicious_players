import argparse
import os
import pandas as pd
import chess.pgn
import zstandard as zstd
from enums import TimeControl, Folders
from pathlib import Path


all_player_info = {}
# dictionary storing player info in the following format:
# {
#     ('player1', 'bullet'): {
#         'ratings': [rating1, rating2, ...],
#         'opponent_ratings': [opp_rating1, opp_rating2, ...],
#         'actual_scores': [score1, score2, ...],
#         'rating_gains': [rating_gain1, rating_gain2, ...]
#         'increments': [increment1, increment2, ...]
#     },
#     ('player1', 'blitz'): {
#         ...
#     },
#     ...
#     ('playerN', 'bullet'): {
#         ...
#     }
# }


def update_all_player_info(
    player: str,
    time_control: str,
    current_rating: float,
    opponent_rating: float,
    score: int,
    rating_gain: float,
    is_increment: int,
    all_player_info: dict = all_player_info,
) -> None:
    """Updates all_player_info dictionary with the information from a single game."""

    # this particular (player, time control) has not been added to all_player_info
    if (all_player_info.get((player, time_control)) is None) & (
        current_rating != 1500.0
    ):
        all_player_info[(player, time_control)] = {
            "ratings": [current_rating],
            "opponent_ratings": [opponent_rating],
            "actual_scores": [score],
            "rating_gains": [rating_gain],
            "increments": [is_increment],
        }

    # exclude a rating of 1500.0 exactly as this could be a first game
    # refine analysis by excluding the first N_0 = 10 games if the first rating is 1500.0
    elif (all_player_info.get((player, time_control)) is None) & (
        current_rating == 1500.0
    ):
        pass

    # this particular (player, time control) is already in all_player_info
    # so update or append each field as needed
    else:
        all_player_info[(player, time_control)]["ratings"].append(current_rating)
        all_player_info[(player, time_control)]["opponent_ratings"].append(
            opponent_rating
        )
        all_player_info[(player, time_control)]["actual_scores"].append(score)
        all_player_info[(player, time_control)]["rating_gains"].append(rating_gain)
        all_player_info[(player, time_control)]["increments"].append(is_increment)


def parse_pgn(PGN_FILE_PATH):
    """Parses the pgn file and extracts information from each game, calls update_all_player_info after each game,
    and creates a DataFrameom from all_player_info which is then written to a csv file.
    """

    print(f"Parsing {PGN_FILE_PATH}...")

    if not os.path.exists(Folders.LICHESS_PLAYER_DATA.value):
        os.mkdir(Folders.LICHESS_PLAYER_DATA.value)

    pgn = open(PGN_FILE_PATH)

    # parse the pgn file, and extract information from each game
    number_of_games_parsed = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            print(f"{number_of_games_parsed} [valid] games parsed.")
            break

        headers = game.headers

        # get time control
        event = headers["Event"]
        if TimeControl.BULLET.value in event.lower():
            time_control = TimeControl.BULLET.value
        elif TimeControl.BLITZ.value in event.lower():
            time_control = TimeControl.BLITZ.value
        elif TimeControl.RAPID.value in event.lower():
            time_control = TimeControl.RAPID.value
        # older games label the classical rating pool as "Standard"
        elif (TimeControl.CLASSICAL.value in event.lower()) | ("standard" in event.lower()):
            time_control = TimeControl.CLASSICAL.value
        else:
            time_control = TimeControl.OTHER.value

        # get info for both players
        white_player, black_player = headers.get("White"), headers.get("Black")
        white_rating, black_rating = headers.get("WhiteElo"), headers.get("BlackElo")
        white_gain, black_gain = headers.get("WhiteRatingDiff"), headers.get(
            "BlackRatingDiff"
        )
        # TimeControl is "base+increment", e.g. "180+2"
        increment = headers["TimeControl"].partition("+")[2]
        result = headers["Result"]

        # skip games with unknown players, ratings, rating difference, or result
        # if either opponent has not played rated games, their rating is 1500
        # but a rating difference is not calculated because this rating is misleading
        # therefore, we will exclude such games
        skip_game_condition = (
            ("?" in white_player)
            | ("?" in black_player)
            | (white_player is None)
            | (black_player is None)
            | ("?" in str(white_rating))
            | ("?" in str(black_rating))
            | (white_gain is None)
            | (black_gain is None)
            | (result not in ["1-0", "0-1", "1/2-1/2"])
        )
        if skip_game_condition:
            continue
        else:
            white_score = 1 if result == "1-0" else 0.5 if result == "1/2-1/2" else 0
            black_score = 0 if result == "1-0" else 0.5 if result == "1/2-1/2" else 1

            ## only convert rating and rating gain to a number once we know it's not None
            white_rating = float(white_rating)
            black_rating = float(black_rating)
            white_gain = float(white_gain)
            black_gain = float(black_gain)

            is_increment = 0 if increment in ("", "0") else 1

            # update white player info
            update_all_player_info(
                player=white_player,
                time_control=time_control,
                current_rating=white_rating,
                opponent_rating=black_rating,
                score=white_score,
                rating_gain=white_gain,
                is_increment=is_increment,
            )

            # update black player info
            update_all_player_info(
                player=black_player,
                time_control=time_control,
                current_rating=black_rating,
                opponent_rating=white_rating,
                score=black_score,
                rating_gain=black_gain,
                is_increment=is_increment,
            )

            number_of_games_parsed += 1
            if number_of_games_parsed % 10000 == 0:
                print(f"{number_of_games_parsed} games parsed...")

    # convert to pandas DataFrame
    all_player_df = pd.DataFrame.from_dict(
        all_player_info,
        orient="index",
        columns=[
            "ratings",
            "opponent_ratings",
            "actual_scores",
            "rating_gains",
            "increments",
        ],
    )

    # explode all_player_df to each row corresponds to one game
    all_player_games_exploded = all_player_df.explode(
        column=[
            "ratings",
            "opponent_ratings",
            "actual_scores",
            "rating_gains",
            "increments",
        ]
    )

    # save to csv
    BASE_FILE_NAME = Path(PGN_FILE_PATH).stem.split(".")[0]
    all_player_games_exploded.to_csv(
        f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PGN file")
    parser.add_argument("PGN_FILE_PATH", type=str, help="Path to the PGN file")
    args = parser.parse_args()

    ## parse PGN file
    parse_pgn(args.PGN_FILE_PATH)
