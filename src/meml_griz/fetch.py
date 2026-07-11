from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.static import teams
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import time
from enum import Enum


def main():

    pass


class NbaTeams(Enum):
    """Enumeration of NBA teams mapping abbreviations to full names and IDs.

    Each tuple contains: (Abbreviation, Full Name, Team ID)
    """

    MEM = ("MEM", "Memphis Grizzlies", "1610612763")

    @property
    def abbreviation(self) -> str:
        return self.value[0]

    @property
    def full_name(self) -> str:
        return self.value[1]

    @property
    def team_id(self) -> str:
        return self.value[2]


@dataclass
class GameData:
    players: pd.DataFrame
    game_id: str
    game_date: datetime

    @classmethod
    def from_game_id(cls, id_game: str, team: NbaTeams, game_date: datetime):

        # Get box-score
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=id_game, timeout=30)

        # Only player data
        players_df = box.player_stats.get_data_frame()

        # Only my team
        df = players_df[players_df["teamTriCode"] == team.abbreviation]

        # Create data class
        return cls(players=df, game_id=id_game, game_date=game_date)


@dataclass
class SeasonData:
    games: list[GameData]
    team: NbaTeams

    @classmethod
    def from_year(cls, team: NbaTeams, year_start: int):
        # Get the NBA team and their games played so far
        griz = teams.find_team_by_abbreviation(team.abbreviation)
        games = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=griz["id"], timeout=30
        )

        # Get the df
        games_df = games.get_data_frames()[0]

        # Filter by year
        games_df["date_time"] = pd.to_datetime(games_df["GAME_DATE"])
        df = games_df[games_df["date_time"].dt.year > year_start]

        # Get GameData objects
        results = []
        for tup in df.itertuples():
            results.append(
                GameData.from_game_id(
                    id_game=tup.GAME_ID, team=team, game_date=tup.GAME_DATE
                )
            )

        return cls(games=results, team=team)


if __name__ == "__main__":
    main()
