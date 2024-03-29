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

    ## plot the distribution of mean rating gain for each rating bin
    for time_group, time_group_df in all_player_features.groupby("time_control"):
        fig = go.Figure()
        for rating_bin, rating_group in time_group_df.groupby("rating_bin"):
            rating_bin_str = f"{rating_bin}-{rating_bin+100}"
            fig.add_trace(
                go.Violin(
                    x=rating_group["mean_rating_gain"].values,
                    name=rating_bin_str,
                    box_visible=False,
                    meanline_visible=False,
                    opacity=0.5,
                ),
            )

        ## the side='positive' argument to update_traces method
        ## is only valid for a figure containing only go.Violin plots,
        ## so we have to update_layout before we add any of the go.Scatter traces

        fig.update_traces(orientation="h", side="positive", width=3, points=False)
        fig.update_layout(
            title=f"{time_group.capitalize()} Rating Changes by Rating Bin",
            xaxis_title="Mean Rating Change",
            yaxis_title="Rating Bin",
            xaxis_showgrid=False,
            xaxis_zeroline=False,
        )

        ## add markers to indicate the mean rating gain for each rating bin
        for rating_bin, rating_group in time_group_df.groupby("rating_bin"):
            rating_bin_str = f"{rating_bin}-{rating_bin+100}"
            fig.add_trace(
                go.Scatter(
                    x=[rating_group["mean_rating_gain"].mean()],
                    y=[rating_bin_str],
                    mode="markers",
                    showlegend=False,
                    marker={"color": "black", "size": 5},
                    marker_symbol="diamond",
                )
            )

        fig.add_vline(
            x=0, line_dash="dash", line_color="blue", line_width=2, opacity=0.5
        )
        fig.write_html(
            f"{Folders.EXPLORATORY_PLOTS.value}/{BASE_FILE_NAME}_{time_group}_rating_gain.html"
        )

    ## plot distribution of mean_perf_diff
    for time_group, time_group_df in all_player_features.groupby("time_control"):
        fig = go.Figure()
        for rating_bin, rating_group in time_group_df.groupby("rating_bin"):
            rating_bin_str = f"{rating_bin}-{rating_bin+100}"
            fig.add_trace(
                go.Violin(
                    x=rating_group["mean_perf_diff"].values,
                    name=rating_bin_str,
                    box_visible=False,
                    meanline_visible=False,
                    opacity=0.5,
                ),
            )

        ## the side='positive' argument to update_traces method
        ## is only valid for a figure containing only go.Violin plots,
        ## so we have to update_layout before we add any of the go.Scatter traces

        ## add markers to indicate the mean rating gain for each rating bin
        fig.update_traces(orientation="h", side="positive", width=3, points=False)
        fig.update_layout(
            title=f"{time_group.capitalize()} Performance Difference by Rating Bin",
            xaxis_title="Mean Performance Difference",
            yaxis_title="Rating Bin",
            xaxis_range=[-1.00, 1.00],
            xaxis_showgrid=False,
            xaxis_zeroline=False,
        )

        for rating_bin, rating_group in time_group_df.groupby("rating_bin"):
            rating_bin_str = f"{rating_bin}-{rating_bin+100}"
            fig.add_trace(
                go.Scatter(
                    x=[rating_group["mean_perf_diff"].mean()],
                    y=[rating_bin_str],
                    mode="markers",
                    showlegend=False,
                    marker={"color": "black", "size": 5},
                    marker_symbol="diamond",
                )
            )

        fig.add_vline(
            x=0.0, line_dash="dash", line_color="blue", line_width=2, opacity=0.5
        )

        fig.write_html(
            f"{Folders.EXPLORATORY_PLOTS.value}/{BASE_FILE_NAME}_{time_group}_perf_diff.html"
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
