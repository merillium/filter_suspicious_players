import pandas as pd
from make_player_rating_graphs import (
    ACCOUNT_STATUS_COLORS,
    UNKNOWN_STATUS_COLOR,
    get_player_pages,
    make_player_rating_graph,
)


def get_flagged_players():
    ## 12 flagged players, already in order of mean performance difference
    return pd.DataFrame(
        {
            "player": [f"test_player{i}" for i in range(1, 13)],
            "time_control": "blitz",
            "mean_perf_diff": [0.3 - 0.01 * i for i in range(12)],
            "account_status": ["open", "closed", "tosViolation", "not found"] * 3,
        }
    )


def test_get_player_pages():
    pages = get_player_pages(get_flagged_players(), players_per_page=10)
    assert [label for label, _, _ in pages] == ["Players 1-10", "Players 11-12"]
    assert [first_rank for _, first_rank, _ in pages] == [1, 11]
    assert list(pages[1][2]["player"]) == ["test_player11", "test_player12"]


def test_make_player_rating_graph():
    ## the last two flagged players, with 3 games each except the last player, who has 4
    flagged_players = get_flagged_players().iloc[10:]
    player_games = (
        pd.DataFrame(
            {
                "player": ["test_player11"] * 3 + ["test_player12"] * 4,
                "time_control": "blitz",
                "utc_datetime": pd.to_datetime(
                    ["2018-01-03", "2018-01-01", "2018-01-02"] * 2 + ["2018-01-04"]
                ),
                "ratings": [1500, 1480, 1490] * 2 + [1510],
                "opponent_ratings": [1500] * 7,
                "actual_scores": [1.0] * 7,
                "rating_gains": [10] * 7,
                "estimated_rd": [60.0] * 7,
            }
        )
        .set_index(["player", "time_control"])
        .sort_index()
    )

    fig = make_player_rating_graph(
        flagged_players, player_games, title="test", first_rank=11
    )

    ## each player has a trace colored by account status, numbered from the first rank
    assert [trace.name.split(" ")[:2] for trace in fig.data] == [
        ["11.", "test_player11"],
        ["12.", "test_player12"],
    ]
    assert [trace.line.color for trace in fig.data] == [
        ACCOUNT_STATUS_COLORS["tosViolation"],
        UNKNOWN_STATUS_COLOR,
    ]

    ## games are plotted in time order, at the rating after each game, against the number of games
    assert list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].y) == [1490, 1500, 1510]

    ## the x-axis ends at the largest number of games played by any of the players
    assert list(fig.layout.xaxis.range) == [1, 4]
