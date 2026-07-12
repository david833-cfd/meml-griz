from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.static import teams
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import time
from enum import Enum
from loguru import logger
from tqdm import tqdm
import random
from collections import namedtuple


def main():
    logger.info("Testing data ingestion loop...")
    # Fetch 2024 season entries for the Grizzlies
    griz_season = SeasonData.from_date_range(
        team=NbaTeams.MEM, start_date=datetime(2026, 3, 31)
    )

    logger.info(f"Successfully loaded {len(griz_season.games)} games!")
    if griz_season.games:
        logger.info("\nFirst game roster extract:")
        logger.info(
            griz_season.games[0].players[["familyName", "points", "assists"]].head()
        )


# Define the data shape once at the top of your test file
GameMeta = namedtuple("GameMeta", ["game_id", "game_date", "matchup"])


class NbaTeams(Enum):
    """Enumeration of NBA teams mapping abbreviations to full names and IDs.

    Each tuple contains: (Abbreviation, Full Name, Team ID)
    """

    _TEAM_META = {
        "ATL": ("Atlanta Hawks", "1610612737"),
        "BOS": ("Boston Celtics", "1610612738"),
        "BKN": ("Brooklyn Nets", "1610612751"),
        "CHA": ("Charlotte Hornets", "1610612766"),
        "CHI": ("Chicago Bulls", "1610612741"),
        "CLE": ("Cleveland Cavaliers", "1610612739"),
        "DAL": ("Dallas Mavericks", "1610612742"),
        "DEN": ("Denver Nuggets", "1610612743"),
        "DET": ("Detroit Pistons", "1610612744"),
        "GSW": ("Golden State Warriors", "1610612744"),
        "HOU": ("Houston Rockets", "1610612745"),
        "IND": ("Indiana Pacers", "1610612754"),
        "LAC": ("LA Clippers", "1610612746"),
        "LAL": ("Los Angeles Lakers", "1610612747"),
        "MEM": ("Memphis Grizzlies", "1610612763"),
        "MIA": ("Miami Heat", "1610612748"),
        "MIL": ("Milwaukee Bucks", "1610612749"),
        "MIN": ("Minnesota Timberwolves", "1610612750"),
        "NOP": ("New Orleans Pelicans", "1610612740"),
        "NYK": ("New York Knicks", "1610612752"),
        "OKC": ("Oklahoma City Thunder", "1610612760"),
        "ORL": ("Orlando Magic", "1610612753"),
        "PHI": ("Philadelphia 76ers", "1610612755"),
        "PHX": ("Phoenix Suns", "1610612756"),
        "POR": ("Portland Trail Blazers", "1610612757"),
        "SAC": ("Sacramento Kings", "1610612758"),
        "SAS": ("San Antonio Spurs", "1610612759"),
        "TOR": ("Toronto Raptors", "1610612761"),
        "UTA": ("Utah Jazz", "1610612762"),
        "WAS": ("Washington Wizards", "1610612764"),
    }

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
    matchup: str
    is_home: bool = False

    @classmethod
    def from_game_id(
        cls, id_game: str, team: NbaTeams, game_date: datetime, matchup: str
    ):

        # Get box-score
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=id_game, timeout=30)

        # Only player data
        players_df = box.player_stats.get_data_frame()

        # Only my team
        df = players_df[players_df["teamTricode"] == team.abbreviation]

        instance = cls(
            players=df, game_id=id_game, game_date=game_date, matchup=matchup
        )

        # Make player full name
        instance.add_full_name()

        # Is game home or away
        instance.get_home_away()

        # Create data class
        return instance

    def add_full_name(self):
        self.players["fullName"] = (
            self.players["firstName"] + " " + self.players["familyName"]
        )

    def get_home_away(self):
        self.is_home = "vs" in self.matchup


@dataclass
class SeasonData:
    games: list[GameData]
    team: NbaTeams

    @classmethod
    def from_date_range(
        cls, team: NbaTeams, start_date: datetime, end_date: datetime = datetime.now()
    ):
        # Get the NBA team and their games played so far
        griz = teams.find_team_by_abbreviation(team.abbreviation)
        games = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=griz["id"], timeout=30
        )

        # Get the df
        games_df = games.get_data_frames()[0]

        # Filter by date
        games_df["date_time"] = pd.to_datetime(games_df["GAME_DATE"])

        df = games_df[games_df["date_time"].between(start_date, end_date)]

        # Get GameData objects
        results = []
        for tup in tqdm(df.itertuples(), desc="Pulling game data", total=len(df)):
            # Is the game home or away

            results.append(
                GameData.from_game_id(
                    id_game=tup.GAME_ID,
                    team=team,
                    game_date=tup.GAME_DATE,
                    matchup=tup.MATCHUP,
                )
            )

            # Random jitter to not get flagged by API.
            time.sleep(random.uniform(0.5, 2.5))

        return cls(games=results, team=team)


if __name__ == "__main__":
    main()
