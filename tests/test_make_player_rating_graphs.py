import pandas as pd
from make_player_rating_graphs import (
    ACCOUNT_STATUS_COLORS,
    UNKNOWN_STATUS_COLOR,
    make_player_rating_graph,
)


def test_make_player_rating_graph():
    ## 12 flagged players, already in order of mean performance difference, with 3 games each
    players = [f"test_player{i}" for i in range(1, 13)]
    statuses = ["open", "closed", "tosViolation", "not found"] * 3
    flagged_players = pd.DataFrame(
        {
            "player": players,
            "time_control": "blitz",
            "mean_perf_diff": [0.3 - 0.01 * i for i in range(12)],
            "account_status": statuses,
        }
    )
    player_games = pd.DataFrame(
        {
            "player": [player for player in players for _ in range(3)],
            "time_control": "blitz",
            "utc_datetime": pd.to_datetime(
                ["2018-01-03", "2018-01-01", "2018-01-02"] * 12
            ),
            "ratings": [1500, 1480, 1490] * 12,
            "opponent_ratings": [1500] * 36,
            "actual_scores": [1.0, 1.0, 1.0] * 12,
            "rating_gains": [10, 10, 10] * 12,
            "estimated_rd": [60.0] * 36,
        }
    ).set_index(["player", "time_control"]).sort_index()

    fig = make_player_rating_graph(flagged_players, player_games, title="test")

    ## each player has a trace colored by account status, and only the first 10 are visible
    assert [trace.name.split(" ")[1] for trace in fig.data] == players
    assert [trace.line.color for trace in fig.data] == [
        ACCOUNT_STATUS_COLORS.get(status, UNKNOWN_STATUS_COLOR) for status in statuses
    ]
    assert [trace.visible for trace in fig.data] == [True] * 10 + [False] * 2

    ## games are plotted in time order, at the rating after each game
    assert list(fig.data[0].y) == [1490, 1500, 1510]

    ## the dropdown pages through the players 10 at a time
    buttons = fig.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == ["Players 1-10", "Players 11-12"]
    assert buttons[1].args[0]["visible"] == [False] * 10 + [True] * 2
