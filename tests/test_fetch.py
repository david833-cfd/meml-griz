from meml_griz import fetch
import pytest
from datetime import datetime


@pytest.fixture
def mem_vs_hou():
    game_id = "0022501194"
    game_date = datetime(year=2026, month=4, day=12)
    matchup = "MEM vs. HOU"
    return fetch.GameMeta(game_id=game_id, game_date=game_date, matchup=matchup)


@pytest.fixture
def my_team():
    return fetch.NbaTeams.MEM


@pytest.fixture
def game_meta(mem_vs_hou, my_team):
    game_data = fetch.GameData.from_game_id(
        id_game=mem_vs_hou.game_id,
        team=my_team,
        game_date=mem_vs_hou.game_date,
        matchup=mem_vs_hou.matchup,
    )
    return game_data


@pytest.fixture
def season(my_team):
    return fetch.SeasonData.from_date_range(
        team=my_team, start_date=datetime(2025, 1, 1), end_date=datetime(2025, 1, 3)
    )


def test_home_away(mem_vs_hou, my_team):

    game_data = fetch.GameData.from_game_id(
        id_game=mem_vs_hou.game_id,
        team=my_team,
        game_date=mem_vs_hou.game_date,
        matchup=mem_vs_hou.matchup,
    )

    assert game_data.is_home is True


def test_game_meta(game_meta, mem_vs_hou):
    assert game_meta.game_date == mem_vs_hou.game_date
    assert game_meta.game_id == mem_vs_hou.game_id
    assert game_meta.matchup == mem_vs_hou.matchup


def test_season_meta(season, my_team):
    assert isinstance(season.games, list)
    assert my_team == season.team
