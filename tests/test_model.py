import copy
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from unittest import mock
from model import PlayerAnomalyDetectionModel
from player_account_handler import PlayerAccountHandler


# fixture for sample training data
@pytest.fixture(scope="class")
def get_sample_train_data():
    sample_train_data = pd.DataFrame(
        {
            "player": [f"test_player{i}" for i in range(1, 6 + 1)] * 2,
            "time_control": ["blitz"] * 6 + ["bullet"] * 6,
            "number_of_games": [100] * 12,
            "mean_perf_diff": [0.155, 0.16, 0.17, 0.18, 0.19, 0.25]
            + [0.16, 0.17, 0.18, 0.19, 0.20, 0.26],
            "number_of_settled_games": [90] * 6 + [100] * 6,
            "mean_rating_settled": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1610, 1620, 1630, 1640, 1650, 1660],
            "mean_perf_diff_settled": [0.155, 0.16, 0.17, 0.18, 0.19, 0.25]
            + [0.16, 0.17, 0.18, 0.19, 0.20, 0.26],
            ## blitz players also have enough new games to be fitted separately, bullet players do not
            "number_of_new_games": [10] * 6 + [0] * 6,
            "mean_rating_new": [1510, 1520, 1530, 1540, 1550, 1560] + [None] * 6,
            "mean_perf_diff_new": [0.255, 0.26, 0.27, 0.28, 0.29, 0.35] + [None] * 6,
            "std_perf_diff": [0.005] * 12,
            "mean_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "median_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "std_rating": [10] * 12,
            "mean_opponent_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "std_opponent_rating": [10] * 12,
            "mean_rating_gain": [1.00, -1.00, 1.00, -1.00, 1.00, -1.00]
            + [1.00, -1.00, 1.00, -1.00, 1.00, -1.00],
            "std_rating_gain": [0.01] * 12,
            "proportion_increment_games": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
            + [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "rating_bin": [1500] * 6 + [1600] * 6,
        }
    )

    return sample_train_data


# fixture for sample test data
# players 7-12 where players 10, 11, 12 perform above expected
@pytest.fixture(scope="class")
def get_sample_test_data():
    sample_test_data = pd.DataFrame(
        {
            "player": [f"test_player{i}" for i in range(7, 6 + 7)] * 2,
            "time_control": ["blitz"] * 6 + ["bullet"] * 6,
            "number_of_games": [100] * 12,
            "mean_perf_diff": [0.04, 0.06, 0.15, 0.161, 0.17, 0.25]
            + [0.15, 0.15, 0.16, 0.171, 0.18, 0.25],
            "number_of_settled_games": [100] * 12,
            "mean_rating_settled": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1610, 1620, 1630, 1640, 1650, 1660],
            "mean_perf_diff_settled": [0.04, 0.06, 0.15, 0.161, 0.17, 0.25]
            + [0.15, 0.15, 0.16, 0.171, 0.18, 0.25],
            "number_of_new_games": [0] * 12,
            "mean_rating_new": [None] * 12,
            "mean_perf_diff_new": [None] * 12,
            "std_perf_diff": [0.005] * 12,
            "mean_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "median_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "std_rating": [10] * 12,
            "mean_opponent_rating": [1510, 1520, 1530, 1540, 1550, 1560]
            + [1510, 1520, 1530, 1540, 1550, 1560],
            "std_opponent_rating": [10] * 12,
            "mean_rating_gain": [1.00, -1.00, 1.00, -1.00, 1.00, -1.00]
            + [1.00, -1.00, 1.00, -1.00, 1.00, -1.00],
            "std_rating_gain": [0.01] * 12,
            "proportion_increment_games": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
            + [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "rating_bin": [1500] * 6 + [1600] * 6,
        }
    )

    return sample_test_data


@pytest.mark.usefixtures("get_sample_train_data", "build_training_data")
@pytest.mark.usefixtures("get_sample_test_data", "build_test_data")
class TestPlayerAnomalyDetectionModel(unittest.TestCase):
    @mock.patch(
        "player_account_handler.PlayerAccountHandler.update_player_account_status"
    )
    def setUp(self, mock_update_player_account_status):
        player_account_handler = PlayerAccountHandler()
        ## label every player, since each rating bin in the sample data only has 6 players
        self.model = PlayerAnomalyDetectionModel(
            player_account_handler, label_quantile=0.0
        )

    @pytest.fixture(autouse=True)
    def build_training_data(self, get_sample_train_data):
        self.sample_train_data = get_sample_train_data

    @pytest.fixture(autouse=True)
    def build_test_data(self, get_sample_test_data):
        self.sample_test_data = get_sample_test_data

    def test_fit(self):
        ## this is a workaround to avoid calling get_player_account_status
        self.model._player_account_handler._account_statuses = {
            "test_player1": "open",
            "test_player2": "open",
            "test_player3": "tosViolation",
            "test_player4": "tosViolation",
            "test_player5": "tosViolation",
            "test_player6": "closed",
        }

        ## expected settled thresholds are 0.16, 0.17, and the expected new threshold is 0.26
        ## and NOT the defaults initially set within the model
        ## e.g. for blitz settled, a threshold of 0.16 flags 4 players with X = 3.75,
        ## the highest wilson lower bound (0.450) of any threshold in that rating bin
        expected_thresholds = copy.deepcopy(self.model._thresholds)
        expected_thresholds[("blitz", "settled")]["1500-1600"] = 0.16
        expected_thresholds[("bullet", "settled")]["1600-1700"] = 0.17
        expected_thresholds[("blitz", "new")]["1500-1600"] = 0.26
        self.model.fit(self.sample_train_data, generate_plots=False)
        assert expected_thresholds == self.model._thresholds

    def test_predict(self):
        self.model._player_account_handler._account_statuses = {
            "test_player7": "open",
            "test_player8": "open",
            "test_player9": "open",
            "test_player10": "closed",
            "test_player11": "tosViolation",
            "test_player12": "tosViolation",
        }
        self.model.fit(self.sample_test_data, generate_plots=False)
        expected_predictions = self.sample_test_data.copy()
        expected_predictions["is_anomaly_settled"] = [
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        expected_predictions["is_anomaly_new"] = [False] * 12
        expected_predictions["is_anomaly"] = [
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
        ]
        expected_predictions["account_status"] = [
            "open",
            "open",
            "open",
            "closed",
            "tosViolation",
            "tosViolation",
            "open",
            "open",
            "open",
            "closed",
            "tosViolation",
            "tosViolation",
        ]
        test_predictions = self.model.predict(self.sample_test_data)
        assert_frame_equal(expected_predictions, test_predictions)

    def test_save_model(self):
        pass

    def test_load_model(self):
        pass
