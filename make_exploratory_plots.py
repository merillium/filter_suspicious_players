import argparse
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from enums import Folders


def make_exploratory_plots(CSV_PLAYER_FEATURE_FILE_PATH):
    if not os.path.exists(Folders.EXPLORATORY_PLOTS.value):
        os.mkdir(Folders.EXPLORATORY_PLOTS.value)

    BASE_FILE_NAME = Path(CSV_PLAYER_FEATURE_FILE_PATH).stem.split(".")[0]

    ## load the player features dataframe
    all_player_features = pd.read_csv(CSV_PLAYER_FEATURE_FILE_PATH)
    all_player_features = all_player_features[
        all_player_features["time_control"].isin(["bullet", "blitz", "classical"])
    ]

    ## plot distribution of mean_perf_diff separately for settled (RD < 80) and new (RD >= 80) games
    ## players are binned by their mean rating within each group, and both plots share an x-axis range
    rd_groups = {
        "settled": ("RD < 80", 30, "settled players"),
        "new": ("RD >= 80", 10, "players new to the rating pool"),
    }
    for time_group, time_group_df in all_player_features.groupby("time_control"):
        for rd_group, (rd_label, min_games, description) in rd_groups.items():
            rd_group_df = time_group_df[
                time_group_df[f"number_of_{rd_group}_games"] >= min_games
            ]
            rating_bins = (
                (rd_group_df[f"mean_rating_{rd_group}"] // 100 * 100).astype(int).values
            )
            fig = go.Figure()
            for rating_bin, rating_group in rd_group_df.groupby(rating_bins):
                fig.add_trace(
                    go.Violin(
                        x=rating_group[f"mean_perf_diff_{rd_group}"].values,
                        name=f"{rating_bin}-{rating_bin+100}",
                        line_color="#3a6ea5",
                        fillcolor="rgba(58, 110, 165, 0.35)",
                        line_width=1,
                    )
                )
            ## the side='positive' argument to update_traces method
            ## is only valid for a figure containing only go.Violin plots,
            ## so we have to update_traces before we add any of the go.Scatter traces
            fig.update_traces(
                orientation="h", side="positive", width=3, points=False, showlegend=False
            )
            fig.update_layout(
                title=(
                    f"{time_group.capitalize()} Performance Difference by Rating Bin, Estimated {rd_label}"
                    f"<br><sup>{description}: {len(rd_group_df):,} players with at least {min_games} games "
                    f"at {rd_label}. Diamonds mark the bin mean.</sup>"
                ),
                xaxis_title="Mean Performance Difference (actual score - expected score)",
                yaxis_title="Rating Bin",
                xaxis_range=[-0.50, 0.50],
                xaxis_showgrid=False,
                xaxis_zeroline=False,
                plot_bgcolor="white",
                height=900,
            )

            for rating_bin, rating_group in rd_group_df.groupby(rating_bins):
                fig.add_trace(
                    go.Scatter(
                        x=[rating_group[f"mean_perf_diff_{rd_group}"].mean()],
                        y=[f"{rating_bin}-{rating_bin+100}"],
                        mode="markers",
                        showlegend=False,
                        marker={"color": "black", "size": 8},
                        marker_symbol="diamond",
                        hovertemplate=f"mean: %{{x:.3f}}<br>{len(rating_group)} players<extra></extra>",
                    )
                )

            fig.add_vline(x=0.0, line_dash="dash", line_color="#555555", line_width=1.5)
            fig.write_html(
                f"{Folders.EXPLORATORY_PLOTS.value}/{BASE_FILE_NAME}_{time_group}_perf_diff_{rd_group}.html"
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
