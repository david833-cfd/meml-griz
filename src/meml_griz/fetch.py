from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3
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

    # Fetch 2026 season entries for the Grizzlies
    griz_season = SeasonData.from_date_range(
        team=NbaTeams.MEM,
        start_date=datetime(2026, 3, 31),
        end_date=datetime(2026, 4, 3),
    )

    logger.info(f"Successfully loaded {len(griz_season.games)} games!")
    if griz_season.games:
        logger.info("\nFirst game roster extract:")
        logger.info(
            griz_season.games[0].players[["familyName", "points", "assists"]].head()
        )


# Define the data shape once at the top of your test file
GameMeta = namedtuple("GameMeta", ["game_id", "game_date", "matchup"])


class SeasonType(Enum):
    PRESEASON = "1"
    REGULAR_SEASON = "2"
    PLAYOFFS = "4"


class NbaTeams(Enum):
    """Enumeration of NBA teams mapping abbreviations to full names and IDs.

    Each tuple contains: (Abbreviation, Full Name, Team ID)
    """

    ATL = ("ATL", "Atlanta Hawks", "1610612737")
    BOS = ("BOS", "Boston Celtics", "1610612738")
    BKN = ("BKN", "Brooklyn Nets", "1610612751")
    CHA = ("CHA", "Charlotte Hornets", "1610612766")
    CHI = ("CHI", "Chicago Bulls", "1610612741")
    CLE = ("CLE", "Cleveland Cavaliers", "1610612739")
    DAL = ("DAL", "Dallas Mavericks", "1610612742")
    DEN = ("DEN", "Denver Nuggets", "1610612743")
    DET = ("DET", "Detroit Pistons", "1610612744")
    GSW = ("GSW", "Golden State Warriors", "1610612744")
    HOU = ("HOU", "Houston Rockets", "1610612745")
    IND = ("IND", "Indiana Pacers", "1610612754")
    LAC = ("LAC", "LA Clippers", "1610612746")
    LAL = ("LAL", "Los Angeles Lakers", "1610612747")
    MEM = ("MEM", "Memphis Grizzlies", "1610612763")
    MIA = ("MIA", "Miami Heat", "1610612748")
    MIL = ("MIL", "Milwaukee Bucks", "1610612749")
    MIN = ("MIN", "Minnesota Timberwolves", "1610612750")
    NOP = ("NOP", "New Orleans Pelicans", "1610612740")
    NYK = ("NYK", "New York Knicks", "1610612752")
    OKC = ("OKC", "Oklahoma City Thunder", "1610612760")
    ORL = ("ORL", "Orlando Magic", "1610612753")
    PHI = ("PHI", "Philadelphia 76ers", "1610612755")
    PHX = ("PHX", "Phoenix Suns", "1610612756")
    POR = ("POR", "Portland Trail Blazers", "1610612757")
    SAC = ("SAC", "Sacramento Kings", "1610612758")
    SAS = ("SAS", "San Antonio Spurs", "1610612759")
    TOR = ("TOR", "Toronto Raptors", "1610612761")
    UTA = ("UTA", "Utah Jazz", "1610612762")
    WAS = ("WAS", "Washington Wizards", "1610612764")

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

    def __post_init__(self):
        self._wrangle_data()

    def _wrangle_data(self):
        # Make player full name
        self.add_full_name()

        # Change minutes to timedelta data type
        minutes = self.players["minutes"].replace(r"^\s*$", "00:00", regex=True)
        self.players["minutes"] = pd.to_timedelta("00:" + minutes)

        # Get true/false if player was active
        self.get_is_active()

    @classmethod
    def from_game_id(
        cls, id_game: str, team: NbaTeams, game_date: datetime, matchup: str
    ):

        # Get traditional box-score
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=id_game, timeout=30)
        box_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=id_game, timeout=30)

        # Only player data. Merge on team ID, player family name, player first name
        adv_df = box_adv.player_stats.get_data_frame()
        box_df = box.player_stats.get_data_frame()
        merge_cols = list(adv_df.columns.intersection(box_df.columns))
        df = box_df.merge(
            right=adv_df,
            on=merge_cols,
        )

        # Only my team
        team_df = df[df["teamTricode"] == team.abbreviation].reset_index(drop=True)

        instance = cls(
            players=team_df, game_id=id_game, game_date=game_date, matchup=matchup
        )

        # Create data class
        return instance

    def add_full_name(self):
        self.players["fullName"] = (
            self.players["firstName"] + " " + self.players["familyName"]
        )

    @property
    def is_home(self):
        return "vs" in self.matchup

    @property
    def opponent(self):
        opponent = self.matchup.split(" ")[-1]
        return NbaTeams[opponent]

    def get_is_active(self):
        self.players["isActive"] = self.players["minutes"] > pd.Timedelta(0)


@dataclass
class SeasonData:
    games: list[GameData]
    team: NbaTeams
    season_df: pd.DataFrame = None

    @classmethod
    def from_date_range(
        cls,
        team: NbaTeams,
        start_date: datetime,
        end_date: datetime = datetime.now(),
        season_type: SeasonType = SeasonType.REGULAR_SEASON,
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

        # Filter by season type (pre, regular, playoffs)
        df = df[df["SEASON_ID"].str.startswith(season_type.value)]

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
            time.sleep(random.uniform(0.1, 3.0))

        return cls(games=results, team=team)

    def games_to_df(self):
        df = pd.concat([game.players for game in self.games], ignore_index=True)
        id_to_date = dict(
            zip(
                [game.game_id for game in self.games],
                [game.game_date for game in self.games],
            )
        )
        df["gameDate"] = df["gameId"].map(id_to_date)
        self.season_df = df

    def get_participation_rate(self):
        ngames = len(self.season_df["gameId"].unique())
        self.season_df["gameParticipationRate"] = (
            self.season_df.groupby("playerSlug")["gameId"].transform("count")
            / ngames
            * 100
        )


if __name__ == "__main__":
    main()
