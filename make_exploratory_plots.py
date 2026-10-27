import argparse
import os
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from dropdown_page import write_dropdown_page
from enums import Folders, TimeControl
from model_plots import DASHED_LINE_COLOR, RD_GROUP_COLORS

RD_GROUP_FILL_COLORS = {
    "settled": "rgba(42, 120, 214, 0.3)",
    "new": "rgba(235, 104, 52, 0.3)",
}


def make_exploratory_plots(CSV_PLAYER_FEATURE_FILE_PATH):
    if not os.path.exists(Folders.EXPLORATORY_PLOTS.value):
        os.mkdir(Folders.EXPLORATORY_PLOTS.value)

    BASE_FILE_NAME = Path(CSV_PLAYER_FEATURE_FILE_PATH).stem.split(".")[0]

    ## load the player features dataframe
    all_player_features = pd.read_csv(CSV_PLAYER_FEATURE_FILE_PATH)
    all_player_features = all_player_features[
        all_player_features["time_control"].isin(TimeControl.ALL.value)
    ]

    ## plot distribution of mean_perf_diff for settled (RD < 80) and new (RD >= 80) games, overlaid on the
    ## same axes, with a dropdown to show either or both. players are binned by their mean rating within each group
    ## all time controls are saved to one html file, with a dropdown to choose the time control
    rd_groups = {
        "settled": ("RD < 80", 30, "Settled"),
        "new": ("RD >= 80", 10, "New"),
    }
    ## the month of the training data, from a base file name like lichess_db_standard_rated_2018-01
    month_match = re.search(r"(\d{4})-(\d{2})", BASE_FILE_NAME)
    training_period = (
        f"Month of {datetime(int(month_match[1]), int(month_match[2]), 1):%b %Y}"
        if month_match
        else BASE_FILE_NAME
    )

    figures = {}
    for time_group in TimeControl.ALL.value:
        time_group_df = all_player_features[
            all_player_features["time_control"] == time_group
        ]
        if time_group_df.empty:
            continue
        ## number_of_games counts each game once for every player in it, so games between two players are counted twice
        training_data_summary = (
            f"{training_period}: {time_group_df['number_of_games'].sum():,} games (counted once per player), "
            f"{len(time_group_df):,} players with at least 30 games"
        )
        fig = go.Figure()
        rating_bin_labels = {}
        subtitles = []
        for rd_group, (rd_label, min_games, display_name) in rd_groups.items():
            rd_group_df = time_group_df[
                time_group_df[f"number_of_{rd_group}_games"] >= min_games
            ]
            rating_bins = (
                (rd_group_df[f"mean_rating_{rd_group}"] // 100 * 100).astype(int).values
            )
            subtitles.append(
                f"{display_name}: {len(rd_group_df):,} players with at least {min_games} games at {rd_label}"
            )
            for i, (rating_bin, rating_group) in enumerate(
                rd_group_df.groupby(rating_bins)
            ):
                rating_bin_label = f"{rating_bin}-{rating_bin+100}"
                rating_bin_labels[rating_bin] = rating_bin_label
                perf_diffs = rating_group[f"mean_perf_diff_{rd_group}"].values
                fig.add_trace(
                    go.Violin(
                        x=perf_diffs,
                        y=[rating_bin_label] * len(perf_diffs),
                        name=f"{display_name} games ({rd_label})",
                        meta=rd_group,
                        legendgroup=rd_group,
                        showlegend=i == 0,
                        line_color=RD_GROUP_COLORS[rd_group],
                        fillcolor=RD_GROUP_FILL_COLORS[rd_group],
                        line_width=1,
                        orientation="h",
                        side="positive",
                        width=3,
                        points=False,
                        hoveron="violins",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[perf_diffs.mean()],
                        y=[rating_bin_label],
                        mode="markers",
                        meta=rd_group,
                        legendgroup=rd_group,
                        showlegend=False,
                        marker={
                            "color": RD_GROUP_COLORS[rd_group],
                            "size": 9,
                            "symbol": "diamond",
                            "line": {"color": "black", "width": 1},
                        },
                        hovertemplate=f"{display_name.lower()} mean: %{{x:.3f}}<br>{len(rating_group)} players<extra></extra>",
                    )
                )

        ## the axes are fixed, so switching between settled and new games doesn't move the rating bins
        fig.update_layout(
            title_y=0.98,
            title_yanchor="top",
            title_text=(
                f"{time_group.capitalize()} Performance Difference by Rating Bin"
                + f"<br><span style='font-size:14px'>{training_data_summary}</span>"
                + "".join(f"<br><sup>{subtitle}</sup>" for subtitle in subtitles)
                + "<br><sup>Diamonds mark the bin mean.</sup>"
            ),
            violinmode="overlay",
            xaxis_title="Mean Performance Difference (actual score - expected score)",
            yaxis_title="Rating Bin",
            xaxis_range=[-0.50, 0.50],
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_categoryorder="array",
            yaxis_categoryarray=[
                rating_bin_labels[rating_bin] for rating_bin in sorted(rating_bin_labels)
            ],
            yaxis_range=[-0.5, len(rating_bin_labels) + 1],
            legend={"orientation": "h", "x": 0.0, "y": 1.01, "yanchor": "bottom"},
            plot_bgcolor="white",
            height=900,
            margin={"t": 160},
        )
        fig.add_vline(
            x=0.0, line_dash="dash", line_color=DASHED_LINE_COLOR, line_width=1.5
        )
        figures[(time_group.capitalize(),)] = fig

    ## the settled / new dropdown shows the traces whose meta is the chosen rd group
    write_dropdown_page(
        f"{Folders.EXPLORATORY_PLOTS.value}/{BASE_FILE_NAME}_perf_diff.html",
        "Performance Difference by Rating Bin",
        figures,
        dropdown_labels=["Time control"],
        trace_filter={
            "label": "Games",
            "options": {
                "Settled and new": ["settled", "new"],
                "Settled": ["settled"],
                "New": ["new"],
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create exploratory plots")
    parser.add_argument(
        "CSV_PLAYER_FEATURE_FILE_PATH",
        type=str,
        help="Path to the player features CSV file",
    )
    args = parser.parse_args()

    ## create features from the CSV file
    make_exploratory_plots(args.CSV_PLAYER_FEATURE_FILE_PATH)
