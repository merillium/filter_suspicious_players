import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dropdown_page import write_dropdown_page
from enums import TimeControl

## colors for the rate, its confidence interval, and the settled / new series
RATE_COLOR = "#2a78d6"
INTERVAL_FILL_COLOR = "rgba(42, 120, 214, 0.15)"
LOWER_BOUND_COLOR = "#184f95"
DASHED_LINE_COLOR = "#bbbbbb"
RD_GROUP_COLORS = {"settled": "#2a78d6", "new": "#eb6834"}


def save_model_plots(
    base_file_name, model_plots_folder, threshold_figures, overview_figure
):
    """Saves the model plots to one html file, with the overview at the top and dropdowns for the
    time control, settled or new games, and rating bin of the threshold plot below it, and to a second
    html file with only the threshold plot and its dropdowns.
    The figures are also saved to one json file of {"<time control>_<settled or new>_<rating bin>": figure json},
    and each figure can be loaded with plotly.io.from_json.

    threshold_figures maps (time control, settled or new, rating bin) to the threshold plot of that rating bin.
    """
    os.makedirs(model_plots_folder, exist_ok=True)
    sorted_keys = sorted(
        threshold_figures,
        key=lambda key: (
            TimeControl.ALL.value.index(key[0]),
            list(RD_GROUP_COLORS).index(key[1]),
            key[2],
        ),
    )
    dropdown_figures = {
        (
            time_control.capitalize(),
            rd_group.capitalize(),
            f"{rating_bin}-{rating_bin+100}",
        ): threshold_figures[(time_control, rd_group, rating_bin)]
        for time_control, rd_group, rating_bin in sorted_keys
    }
    dropdown_labels = ["Time control", "Games", "Rating bin"]
    write_dropdown_page(
        f"{model_plots_folder}/{base_file_name}_model_thresholds.html",
        "Model Thresholds",
        dropdown_figures,
        dropdown_labels=dropdown_labels,
        header_figure=overview_figure,
    )
    ## the overview is hard to read at a glance, so the threshold selection plots are also saved without it
    write_dropdown_page(
        f"{model_plots_folder}/{base_file_name}_model_threshold_selection.html",
        "Model Threshold Selection",
        dropdown_figures,
        dropdown_labels=dropdown_labels,
    )

    figures_json = {
        f"{time_control}_{rd_group}_{rating_bin}-{rating_bin+100}": threshold_figures[
            (time_control, rd_group, rating_bin)
        ].to_json()
        for time_control, rd_group, rating_bin in sorted_keys
    }
    if overview_figure is not None:
        figures_json["overview"] = overview_figure.to_json()
    with open(f"{model_plots_folder}/{base_file_name}_model_thresholds.json", "w") as f:
        json.dump(figures_json, f)


def make_model_threshold_figure(
    train_threshold_list,
    train_accuracy_list,
    train_accuracy_lower_bounds,
    train_accuracy_upper_bounds,
    train_metric_list,
    train_number_of_flagged_players,
    best_index,
    time_control,
    rd_group,
    rating_bin_key,
    metric_name,
):
    """Returns a model threshold plot for one rating bin, showing the accuracy X / N of each threshold
    with its 95% Wilson confidence interval, the number of flagged players, and the chosen threshold."""

    ## the wilson lower bound is drawn in the accuracy panel, any other metric gets its own panel
    show_metric_panel = metric_name != "wilson_lower_bound"
    rows = 3 if show_metric_panel else 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.3, 0.25] if show_metric_panel else [0.65, 0.35],
    )

    ## confidence interval band, then the observed accuracy and its lower bound
    fig.add_trace(
        go.Scatter(
            x=list(train_threshold_list) + list(train_threshold_list)[::-1],
            y=list(train_accuracy_upper_bounds) + list(train_accuracy_lower_bounds)[::-1],
            fill="toself",
            fillcolor=INTERVAL_FILL_COLOR,
            line={"width": 0},
            hoverinfo="skip",
            name="95% Wilson interval",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=train_threshold_list,
            y=train_accuracy_list,
            line={"color": RATE_COLOR, "width": 2},
            name="Accuracy (X / N)",
            customdata=list(zip(train_accuracy_lower_bounds, train_accuracy_upper_bounds)),
            hovertemplate="threshold %{x:.3f}<br>accuracy %{y:.1%}<br>95% interval %{customdata[0]:.1%} to %{customdata[1]:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=train_threshold_list,
            y=train_accuracy_lower_bounds,
            line={"color": LOWER_BOUND_COLOR, "width": 3},
            name="Wilson lower bound",
            hovertemplate="threshold %{x:.3f}<br>lower bound %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    best_threshold = train_threshold_list[best_index]
    best_number_flagged = int(train_number_of_flagged_players[best_index])
    best_metric = train_metric_list[best_index]
    fig.add_trace(
        go.Scatter(
            x=[best_threshold],
            y=[train_accuracy_lower_bounds[best_index]],
            mode="markers",
            marker={"color": LOWER_BOUND_COLOR, "size": 11, "line": {"color": "white", "width": 2}},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=best_threshold,
        y=train_accuracy_lower_bounds[best_index],
        text=(
            f"chosen threshold {best_threshold:.3f}<br>"
            f"lower bound {train_accuracy_lower_bounds[best_index]:.1%}, {best_number_flagged} flagged"
        ),
        showarrow=True,
        arrowhead=0,
        ax=60,
        ay=40,
        align="left",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=train_threshold_list,
            y=train_number_of_flagged_players,
            line={"color": RATE_COLOR, "width": 2},
            name="Flagged players",
            showlegend=False,
            hovertemplate="threshold %{x:.3f}<br>%{y} flagged<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if show_metric_panel:
        fig.add_trace(
            go.Scatter(
                x=train_threshold_list,
                y=train_metric_list,
                line={"color": RATE_COLOR, "width": 2},
                name=f"Metric ({metric_name})",
                showlegend=False,
                hovertemplate="threshold %{x:.3f}<br>metric %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    fig.add_vline(
        x=best_threshold, line_width=1.5, line_dash="dash", line_color=DASHED_LINE_COLOR
    )
    fig.update_layout(
        title=(
            f"Threshold Selection for {time_control.capitalize()}, {rd_group.capitalize()} Games: Rating Bin {rating_bin_key}"
            f"<br><sup>The chosen threshold maximizes the {metric_name.replace('_', ' ')} "
            f"({best_metric:.3f}). Accuracy X / N counts tosViolation as 1 and closed as 0.75.</sup>"
        ),
        plot_bgcolor="white",
        height=750 if show_metric_panel else 650,
        legend={"orientation": "h", "y": -0.12},
        hovermode="closest",
    )
    fig.update_yaxes(title_text="Accuracy (X / N)", tickformat=".0%", range=[0, 1], gridcolor="#eeeeee", row=1, col=1)
    fig.update_yaxes(title_text="Flagged players", type="log", gridcolor="#eeeeee", row=2, col=1)
    if show_metric_panel:
        fig.update_yaxes(title_text="Metric", gridcolor="#eeeeee", row=3, col=1)
    fig.update_xaxes(title_text="Threshold (mean performance difference)", row=rows, col=1)

    return fig


def make_model_thresholds_overview_figure(
    thresholds,
    threshold_metrics,
    threshold_number_of_flagged_players,
    metric_name,
):
    """Returns one plot of the chosen threshold, and the metric at that threshold, for every fitted
    (time control, settled or new, rating bin), with one column per time control."""

    time_controls = sorted(
        {
            time_control
            for (time_control, rd_group), metrics in threshold_metrics.items()
            if any(metric is not None for metric in metrics.values())
        }
    )
    if not time_controls:
        return None

    fig = make_subplots(
        rows=2,
        cols=len(time_controls),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.03,
        row_heights=[0.6, 0.4],
        subplot_titles=[time_control.capitalize() for time_control in time_controls],
    )
    for col, time_control in enumerate(time_controls, start=1):
        for rd_group, color in RD_GROUP_COLORS.items():
            fitted_bins = [
                (int(rating_bin_key.split("-")[0]), rating_bin_key)
                for rating_bin_key, metric in threshold_metrics[(time_control, rd_group)].items()
                if metric is not None
            ]
            rating_bins = [rating_bin for rating_bin, _ in fitted_bins]
            customdata = [
                (
                    rating_bin_key,
                    thresholds[(time_control, rd_group)][rating_bin_key],
                    threshold_metrics[(time_control, rd_group)][rating_bin_key],
                    threshold_number_of_flagged_players[(time_control, rd_group)][rating_bin_key],
                )
                for _, rating_bin_key in fitted_bins
            ]
            hovertemplate = (
                f"{rd_group} games, rating bin %{{customdata[0]}}<br>threshold %{{customdata[1]:.3f}}<br>"
                f"{metric_name.replace('_', ' ')} %{{customdata[2]:.3f}}<br>%{{customdata[3]}} flagged<extra></extra>"
            )
            fig.add_trace(
                go.Scatter(
                    x=rating_bins,
                    y=[c[1] for c in customdata],
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"color": color, "size": 8, "line": {"color": "white", "width": 2}},
                    name=f"{rd_group.capitalize()} games",
                    legendgroup=rd_group,
                    showlegend=col == 1,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=rating_bins,
                    y=[c[2] for c in customdata],
                    mode="lines+markers",
                    line={"color": color, "width": 2},
                    marker={"color": color, "size": 8, "line": {"color": "white", "width": 2}},
                    name=f"{rd_group.capitalize()} games",
                    legendgroup=rd_group,
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=2,
                col=col,
            )
        fig.update_xaxes(title_text="Rating bin", gridcolor="#eeeeee", row=2, col=col)
        fig.update_xaxes(gridcolor="#eeeeee", row=1, col=col)

    is_wilson = metric_name == "wilson_lower_bound"
    fig.update_yaxes(title_text="Chosen threshold", gridcolor="#eeeeee", row=1, col=1)
    fig.update_yaxes(
        title_text="Wilson lower bound" if is_wilson else "Metric",
        tickformat=".0%" if is_wilson else None,
        gridcolor="#eeeeee",
        row=2,
        col=1,
    )
    for col in range(2, len(time_controls) + 1):
        fig.update_yaxes(gridcolor="#eeeeee", row=1, col=col)
        fig.update_yaxes(gridcolor="#eeeeee", row=2, col=col)
    fig.update_layout(
        title=(
            "Chosen Thresholds by Rating Bin"
            f"<br><sup>Top: the threshold chosen for each rating bin. Bottom: the {metric_name.replace('_', ' ')} "
            "at that threshold. Hover over a point for the number of flagged players.</sup>"
        ),
        plot_bgcolor="white",
        height=700,
        legend={"orientation": "h", "y": -0.12},
    )
    return fig
