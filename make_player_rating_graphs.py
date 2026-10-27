import argparse
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from dropdown_page import write_dropdown_page
from enums import Folders, TimeControl
from make_player_features import make_player_features
from model import PlayerAnomalyDetectionModel, RD_GROUP_MIN_GAMES, get_rd_group_data
from player_account_handler import PlayerAccountHandler

## each player's rating graph is colored by the status of their account
ACCOUNT_STATUS_COLORS = {
    "open": "#2a78d6",
    "closed": "#eb6834",
    "tosViolation": "#d62728",
}
UNKNOWN_STATUS_COLOR = "#999999"
PLAYERS_PER_PAGE = 10
RESULTS = {1.0: "win", 0.5: "draw", 0.0: "loss"}


def make_player_rating_graph(
    flagged_players: pd.DataFrame,
    player_games: pd.DataFrame,
    title: str,
    first_rank=1,
):
    """Returns a figure with the rating graph of each flagged player, in the order of flagged_players,
    numbered from first_rank.

    flagged_players has columns player, time_control, mean_perf_diff, and account_status,
    and player_games is indexed by (player, time_control) with one row per game.
    """
    ## players play at different rates, so ratings are plotted against the number of games played,
    ## up to the largest number of games played by any of the players
    fig = go.Figure()
    numbers_of_games = []
    for rank, player in enumerate(flagged_players.itertuples(), start=first_rank):
        games = player_games.loc[(player.player, player.time_control)].sort_values(
            "utc_datetime"
        )
        numbers_of_games.append(len(games))
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(games) + 1)),
                y=games["ratings"] + games["rating_gains"],
                mode="lines+markers",
                line={
                    "color": ACCOUNT_STATUS_COLORS.get(
                        player.account_status, UNKNOWN_STATUS_COLOR
                    ),
                    "width": 2,
                },
                marker={"size": 3},
                name=f"{rank}. {player.player} ({player.account_status}, {player.mean_perf_diff:+.3f})",
                customdata=list(
                    zip(
                        games["rating_gains"],
                        games["opponent_ratings"],
                        games["actual_scores"].map(RESULTS),
                        games["estimated_rd"].round(0),
                        games["utc_datetime"].dt.strftime("%Y-%m-%d %H:%M"),
                    )
                ),
                hovertemplate=(
                    f"<b>{player.player}</b><br>"
                    "Game %{x}, %{customdata[4]} UTC<br>"
                    "Rating: %{y} (%{customdata[0]:+d})<br>"
                    "%{customdata[2]} vs %{customdata[1]}<br>"
                    "Estimated RD: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    status_key = " ".join(
        f"<span style='color:{color}'>■ {status}</span>"
        for status, color in ACCOUNT_STATUS_COLORS.items()
    )
    fig.update_layout(
        title=f"{title}<br><sup>{status_key}</sup>",
        xaxis_title="Number of games",
        xaxis_range=[1, max(numbers_of_games)],
        yaxis_title="Rating",
        hovermode="closest",
        plot_bgcolor="white",
        height=650,
        legend_title="Player (account status, mean performance difference)",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig


def get_player_pages(flagged_players: pd.DataFrame, players_per_page=PLAYERS_PER_PAGE):
    """Returns [(label, first rank, players on that page)] for pages of players_per_page players."""
    number_of_players = len(flagged_players)
    return [
        (
            f"Players {start + 1}-{min(start + players_per_page, number_of_players)}",
            start + 1,
            flagged_players.iloc[start : start + players_per_page],
        )
        for start in range(0, number_of_players, players_per_page)
    ]


def make_player_rating_graphs(
    model: PlayerAnomalyDetectionModel,
    player_features: pd.DataFrame,
    player_games: pd.DataFrame,
    base_file_name: str,
    rating_graphs_folder=Folders.PLAYER_RATING_GRAPHS.value,
    players_per_page=PLAYERS_PER_PAGE,
):
    """Saves one html file of rating graphs for the players above the threshold of a fitted model,
    with dropdowns for the time control, settled or new games, rating bin, and page of players_per_page
    players, with the players in order of their mean performance difference (within settled or new games)."""
    predictions = model.predict(player_features)

    ## the account status of flagged players is usually cached during fitting, but not for new data
    player_account_handler = model._player_account_handler
    player_account_handler.update_player_account_statuses(
        predictions.loc[predictions["is_anomaly"], "player"].unique()
    )
    predictions["account_status"] = (
        predictions["player"]
        .map(player_account_handler._account_statuses)
        .fillna("unknown")
    )

    player_games = player_games.set_index(["player", "time_control"]).sort_index()
    rating_bin_figures = []
    for rd_group in RD_GROUP_MIN_GAMES:
        rd_group_data = get_rd_group_data(predictions, rd_group)
        flagged = rd_group_data[
            predictions.loc[rd_group_data.index, f"is_anomaly_{rd_group}"]
        ].copy()
        flagged["account_status"] = predictions.loc[flagged.index, "account_status"]

        for (time_control, rating_bin), flagged_players in flagged.groupby(
            ["time_control", "rating_bin"]
        ):
            rating_bin_key = f"{rating_bin}-{rating_bin+100}"
            threshold = model._thresholds[(time_control, rd_group)][rating_bin_key]
            flagged_players = flagged_players.sort_values(
                "mean_perf_diff", ascending=False
            )
            title = (
                f"{time_control.capitalize()} {rating_bin_key}, {rd_group} games: "
                f"{len(flagged_players)} players above a mean performance difference of {threshold:.3f}"
            )
            ## each page of players is a separate figure, so the number of pages can differ between rating bins
            rating_bin_figures.append(
                (
                    (
                        TimeControl.ALL.value.index(time_control),
                        list(RD_GROUP_MIN_GAMES).index(rd_group),
                        rating_bin,
                    ),
                    [
                        (
                            (
                                time_control.capitalize(),
                                rd_group.capitalize(),
                                rating_bin_key,
                                page_label,
                            ),
                            make_player_rating_graph(
                                page_players, player_games, title, first_rank
                            ),
                        )
                        for page_label, first_rank, page_players in get_player_pages(
                            flagged_players, players_per_page
                        )
                    ],
                )
            )

    rating_bin_figures.sort(key=lambda sort_key_and_pages: sort_key_and_pages[0])
    write_dropdown_page(
        f"{rating_graphs_folder}/{base_file_name}_rating_graphs.html",
        "Player Rating Graphs",
        {
            key: fig
            for _, pages in rating_bin_figures
            for key, fig in pages
        },
        dropdown_labels=["Time control", "Games", "Rating bin", "Players"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make rating graphs of the players flagged by the model"
    )
    parser.add_argument(
        "CSV_RAW_FEATURES_FILE_PATH", type=str, help="Path to the CSV file of games"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Regenerate player features and fit a new model, instead of loading the saved features and model",
    )
    args = parser.parse_args()

    BASE_FILE_NAME = Path(args.CSV_RAW_FEATURES_FILE_PATH).stem.split(".")[0]
    PLAYER_FEATURES_FILE_PATH = (
        f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}_player_features.csv"
    )
    PLAYER_GAMES_FILE_PATH = (
        f"{Folders.LICHESS_PLAYER_DATA.value}/{BASE_FILE_NAME}_player_games.parquet"
    )

    ## player features are preloaded unless training, or they have not been made yet
    if args.train or not (
        os.path.exists(PLAYER_FEATURES_FILE_PATH)
        and os.path.exists(PLAYER_GAMES_FILE_PATH)
    ):
        print("Making player features...")
        make_player_features(args.CSV_RAW_FEATURES_FILE_PATH)
    player_features = pd.read_csv(PLAYER_FEATURES_FILE_PATH)
    player_games = pd.read_parquet(PLAYER_GAMES_FILE_PATH)

    player_account_handler = PlayerAccountHandler(
        f"{Folders.LICHESS_PLAYER_DATA.value}/account_statuses.csv"
    )
    model = PlayerAnomalyDetectionModel(player_account_handler)
    if args.train:
        model.fit(player_features)
        model.save_model(f"{BASE_FILE_NAME}_model")
    else:
        model.load_model(f"{BASE_FILE_NAME}_model")

    print("Making player rating graphs...")
    make_player_rating_graphs(model, player_features, player_games, BASE_FILE_NAME)
