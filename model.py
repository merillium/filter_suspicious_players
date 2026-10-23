import copy
import os
import pickle
import numpy as np
import pandas as pd

from enums import TimeControl, Folders
from player_account_handler import PlayerAccountHandler
from model_plots import (
    generate_model_threshold_plots,
    generate_model_thresholds_overview_plot,
)

## settled (RD < 80) and new (RD >= 80) games are fitted separately, see make_player_features.py
## a player is only evaluated for a group if they have at least this many games in that group
RD_GROUP_MIN_GAMES = {"settled": 30, "new": 10}

## a legitimate strong player who is new to the rating pool will naturally overperform
## on the way up from 1500, so new players start with a higher default threshold
DEFAULT_THRESHOLDS = {"settled": 0.10, "new": 0.20}
THRESHOLD_STEP = 0.005

## the lichess API is limited to 120,000 users per day and asks that it not be used to download all users,
## so we only label the players at or above this quantile of performance difference within each rating bin,
## and only search for thresholds within that range
DEFAULT_LABEL_QUANTILE = 0.9

## we need to integrate this into the model logic properly
BASE_FILE_NAME = "test"


def log_flagged_metric(number_of_flagged_players, estimated_cheaters):
    """log(N+1) * X / N: rewards the accuracy X / N of a threshold, and the number of flagged players N, but not too much"""
    return (
        np.log(number_of_flagged_players + 1)
        * estimated_cheaters
        / number_of_flagged_players
    )


def wilson_interval(number_of_flagged_players, estimated_cheaters, z=1.96):
    """Returns the (lower, upper) bounds of the 95% Wilson confidence interval for the accuracy X / N"""
    n = number_of_flagged_players
    p = estimated_cheaters / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half_width, center + half_width


def wilson_lower_bound_metric(number_of_flagged_players, estimated_cheaters):
    """Lower bound of the 95% Wilson confidence interval for the accuracy X / N of a threshold:
    the accuracy we can be confident in, which penalizes thresholds that flag very few players"""
    return wilson_interval(number_of_flagged_players, estimated_cheaters)[0]


METRICS = {
    "log_flagged": log_flagged_metric,
    "wilson_lower_bound": wilson_lower_bound_metric,
}


def get_rd_group_data(data: pd.DataFrame, rd_group: str) -> pd.DataFrame:
    """Returns the players with enough games in the rd_group, with their rating bin and
    mean performance difference within that group."""
    rd_group_data = data[
        data["time_control"].isin(TimeControl.ALL.value)
        & (data[f"number_of_{rd_group}_games"] >= RD_GROUP_MIN_GAMES[rd_group])
    ]
    return pd.DataFrame(
        {
            "player": rd_group_data["player"],
            "time_control": rd_group_data["time_control"],
            "rating_bin": (rd_group_data[f"mean_rating_{rd_group}"] // 100 * 100).astype(
                int
            ),
            "mean_perf_diff": rd_group_data[f"mean_perf_diff_{rd_group}"],
        },
        index=rd_group_data.index,
    )


class PlayerAnomalyDetectionModel:
    """
    The PlayerAnomalyDetectionModel class returns a model with methods:
    .fit to tune the model's internal thresholds on training data
    .predict to make predictions on test data
    .load_model to load a predefined model from a pkl file
    .save_model to save the model to a file

    A separate threshold on mean performance difference is fitted for each
    (time control, settled or new, rating bin).
    """

    def __init__(
        self,
        player_account_handler,
        metric="wilson_lower_bound",
        label_quantile=DEFAULT_LABEL_QUANTILE,
    ):
        self.is_fitted = False
        self._thresholds = {
            (time_control, rd_group): {
                f"{rating_bin}-{rating_bin+100}": DEFAULT_THRESHOLDS[rd_group]
                for rating_bin in np.arange(0, 4000, 100)
            }
            for time_control in TimeControl.ALL.value
            for rd_group in RD_GROUP_MIN_GAMES
        }
        self._player_account_handler = player_account_handler
        self._account_status_score_map = {
            "open": 0,
            "tosViolation": 1,
            "closed": 0.75,  # weight closed account as closer to a tosViolation
        }
        self._metric = metric
        self._label_quantile = label_quantile
        self._threshold_metrics = {
            (time_control, rd_group): {
                f"{rating_bin}-{rating_bin+100}": None
                for rating_bin in np.arange(0, 4000, 100)
            }
            for time_control in TimeControl.ALL.value
            for rd_group in RD_GROUP_MIN_GAMES
        }
        self._threshold_number_of_flagged_players = copy.deepcopy(
            self._threshold_metrics
        )

    def load_model(
        self,
        model_name: str = "player_anomaly_detection_model",
        saved_models_folder=Folders.SAVED_MODELS.value,
    ):
        """
        Loads the thresholds of a model saved with .save_model, and marks the model as fitted.
        """
        with open(f"{saved_models_folder}/{model_name}.pkl", "rb") as f:
            thresholds = pickle.load(f)
        ## models saved before settled and new games were fitted separately are keyed by time control only
        if not all(key in thresholds for key in self._thresholds):
            raise ValueError(
                f"{model_name} was saved by an older version of the model, and must be refitted"
            )
        self._thresholds = thresholds
        self.is_fitted = True

    def fit(self, train_data: pd.DataFrame, generate_plots=True):
        if not self.is_fitted:
            self._set_thresholds(train_data, generate_plots)
            self.is_fitted = True
        else:
            print("Warning: model is already fitted")
            pass

    def _set_thresholds(self, train_data, generate_plots):
        ## only players at or above the label quantile (and above 0) of their rating bin are labelled
        all_rd_group_data = {}
        for rd_group in RD_GROUP_MIN_GAMES:
            rd_group_data = get_rd_group_data(train_data, rd_group)
            rd_group_data["label_cutoff"] = (
                rd_group_data.groupby(["time_control", "rating_bin"])["mean_perf_diff"]
                .transform(lambda s: s.quantile(self._label_quantile))
                .clip(lower=0)
            )
            all_rd_group_data[rd_group] = rd_group_data[
                rd_group_data["mean_perf_diff"] >= rd_group_data["label_cutoff"]
            ].copy()

        ## get the account status of every player to be labelled up front, in as few API requests as possible
        self._player_account_handler.update_player_account_statuses(
            pd.concat([df["player"] for df in all_rd_group_data.values()]).unique()
        )
        metric_function = METRICS[self._metric]

        for rd_group, rd_group_data in all_rd_group_data.items():
            ## players whose accounts were not found have no label, so they are excluded
            rd_group_data["score"] = (
                rd_group_data["player"]
                .map(self._player_account_handler._account_statuses)
                .map(self._account_status_score_map)
            )
            rd_group_data = rd_group_data.dropna(subset=["score"])

            for (time_control, rating_bin), train_rating_bin_df in rd_group_data.groupby(
                ["time_control", "rating_bin"]
            ):
                rating_bin_key = f"{rating_bin}-{rating_bin+100}"

                ## 1D grid search over thresholds from the label cutoff up to the largest performance difference
                ## N (flagged players) and X (estimated cheaters) for every threshold, using cumulative sums
                order = np.argsort(train_rating_bin_df["mean_perf_diff"].values)
                sorted_perf_diffs = train_rating_bin_df["mean_perf_diff"].values[order]
                sorted_scores = train_rating_bin_df["score"].values[order]
                label_cutoff = train_rating_bin_df["label_cutoff"].iloc[0]
                train_thresholds = np.arange(
                    np.ceil(label_cutoff / THRESHOLD_STEP) * THRESHOLD_STEP,
                    sorted_perf_diffs[-1],
                    THRESHOLD_STEP,
                ).round(3)
                if len(train_thresholds) == 0:
                    continue

                first_flagged_index = np.searchsorted(
                    sorted_perf_diffs, train_thresholds, side="right"
                )
                number_of_flagged_players = len(sorted_perf_diffs) - first_flagged_index
                estimated_cheaters = np.append(np.cumsum(sorted_scores[::-1])[::-1], 0)[
                    first_flagged_index
                ]
                train_accuracies = estimated_cheaters / number_of_flagged_players
                train_metrics = metric_function(
                    number_of_flagged_players, estimated_cheaters
                )

                best_index = np.argmax(train_metrics)
                self._thresholds[(time_control, rd_group)][rating_bin_key] = float(
                    train_thresholds[best_index]
                )
                self._threshold_metrics[(time_control, rd_group)][rating_bin_key] = float(
                    train_metrics[best_index]
                )
                self._threshold_number_of_flagged_players[(time_control, rd_group)][
                    rating_bin_key
                ] = int(number_of_flagged_players[best_index])

                ## generate plots by default
                if generate_plots:
                    accuracy_lower_bounds, accuracy_upper_bounds = wilson_interval(
                        number_of_flagged_players, estimated_cheaters
                    )
                    generate_model_threshold_plots(
                        BASE_FILE_NAME,
                        Folders.MODEL_PLOTS.value,
                        train_thresholds,
                        train_accuracies,
                        accuracy_lower_bounds,
                        accuracy_upper_bounds,
                        train_metrics,
                        number_of_flagged_players,
                        best_index,
                        time_control,
                        rd_group,
                        rating_bin_key,
                        self._metric,
                    )

        if generate_plots:
            generate_model_thresholds_overview_plot(
                BASE_FILE_NAME,
                Folders.MODEL_PLOTS.value,
                self._thresholds,
                self._threshold_metrics,
                self._threshold_number_of_flagged_players,
                self._metric,
            )

    def predict(self, test_data: pd.DataFrame):
        """Returns test_data for the known time controls, with columns is_anomaly_settled and
        is_anomaly_new (whether the player's settled or new games are above the threshold),
        is_anomaly (either), and account_status (if known)
        """
        if not self.is_fitted:
            print("Warning: model is not fitted and will use default thresholds")

        ## predictions are only made on known time controls
        predictions = test_data[
            test_data["time_control"].isin(TimeControl.ALL.value)
        ].copy()

        for rd_group in RD_GROUP_MIN_GAMES:
            rd_group_data = get_rd_group_data(predictions, rd_group)
            thresholds = [
                self._thresholds[(time_control, rd_group)][
                    f"{rating_bin}-{rating_bin+100}"
                ]
                for time_control, rating_bin in zip(
                    rd_group_data["time_control"], rd_group_data["rating_bin"]
                )
            ]
            predictions[f"is_anomaly_{rd_group}"] = False
            predictions.loc[rd_group_data.index, f"is_anomaly_{rd_group}"] = (
                rd_group_data["mean_perf_diff"].values > np.array(thresholds)
            )

        predictions["is_anomaly"] = (
            predictions["is_anomaly_settled"] | predictions["is_anomaly_new"]
        )
        predictions["account_status"] = predictions["player"].map(
            self._player_account_handler._account_statuses
        )

        return predictions

    def save_model(
        self,
        model_name: str = "player_anomaly_detection_model",
        saved_models_folder=Folders.SAVED_MODELS.value,
    ):
        if not os.path.exists(saved_models_folder):
            os.mkdir(saved_models_folder)
        if os.path.exists(f"{saved_models_folder}/{model_name}"):
            model_name = model_name + "_"

        with open(f"{saved_models_folder}/{model_name}.pkl", "wb") as f:
            pickle.dump(self._thresholds, f)
