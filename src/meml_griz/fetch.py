from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3
from nba_api.stats.static import teams
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
import time
from enum import Enum
from loguru import logger
from tqdm import tqdm
import random
from collections import namedtuple
from meml_griz.settings import CACHE_DIR, SLEEP_MAX, SLEEP_MIN
from pathlib import Path
import os
from nba_api.stats.library.http import NBAStatsHTTP

# Force nba_api to masquerade as a normal web browser
custom_headers = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://nba.com",
    "Connection": "keep-alive",
}

NBAStatsHTTP.default_headers = custom_headers


def main():
    logger.info("Testing data ingestion loop...")

    # Fetch 2026 season entries for the Grizzlies
    griz_season = SeasonData.from_date_range(
        teams=[
            NbaTeams.MEM,
        ],
        start_date=datetime(2025, 3, 31),
        end_date=datetime(2026, 4, 3),
    )

    logger.info(f"Successfully loaded {len(griz_season.games)} games!")
    if griz_season.games:
        logger.info("\nFirst game roster extract:")
        logger.info(griz_season.games[0].players[["familyName", "points", "assists"]].head())

    griz_season.get_team_stat()


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
    DET = ("DET", "Detroit Pistons", "1610612765")
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

        # What season is this game
        self.add_season_year()

        # Add home/away columns
        self.add_home_away()

    @classmethod
    def from_game_id(cls, id_game: str, game_date: datetime, matchup: str):

        cache_file = CACHE_DIR / f"{id_game}.parquet"

        if not cache_file.exists():
            # If not in cache, then write to file for later.
            df = pull_game_data(game_id=id_game)
            df.to_parquet(cache_file)
        else:
            df = pd.read_parquet(cache_file)

        instance = cls(players=df, game_id=id_game, game_date=game_date, matchup=matchup)

        # Create data class
        return instance

    def add_full_name(self):
        self.players["fullName"] = self.players["firstName"] + " " + self.players["familyName"]

    @property
    def is_home(self):
        return "vs" in self.matchup

    @property
    def opponent(self):
        opponent = self.matchup.split(" ")[-1]
        return NbaTeams[opponent]

    @property
    def home_team(self):
        return NbaTeams[self.matchup.split(" ")[0]]

    def add_season_year(self):
        self.players["seasonYear"] = pd.to_datetime(
            self.players["gameId"].str[3:5], format="%y"
        ).dt.year

    def add_home_away(self):
        self.players[["home_team", "away_team"]] = (
            self.players["matchup"].str.split(" ", expand=True).iloc[:, [0, 2]]
        )

    def get_is_active(self):
        self.players["isActive"] = self.players["minutes"] > pd.Timedelta(0)


@dataclass
class SeasonData:
    games: list[GameData]
    teams: list[NbaTeams]
    season_df: pd.DataFrame = field(default=None, init=False)

    SLEEP_MIN: float = field(default=0.1)
    SLEEP_MAX: float = field(default=10)

    def __post_init__(self):
        self.games_to_df()
        self.get_participation_rate()
        self.get_team_stat()

    @classmethod
    def from_date_range(
        cls,
        teams: list[NbaTeams],
        start_date: datetime,
        end_date: datetime = datetime.now(),
        season_type: SeasonType = SeasonType.REGULAR_SEASON,
    ):
        # Get the NBA games played so far
        games = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=start_date.strftime("%m/%d/%Y"),
            date_to_nullable=end_date.strftime("%m/%d/%Y"),
            timeout=30,
        )

        # Get the df
        games_df = games.get_data_frames()[0]

        # Filter by date & season type
        games_df["date_time"] = pd.to_datetime(games_df["GAME_DATE"])
        filter_games = games_df[
            (games_df["date_time"].between(start_date, end_date))
            & (games_df["SEASON_ID"].str.startswith(season_type.value))
            & (games_df["TEAM_ABBREVIATION"].isin([team.abbreviation for team in teams]))
            & (games_df.duplicated(subset=["GAME_ID"], keep="first"))  # Drop duplicate IDs
        ]

        results = []

        # Map game id to date
        id_to_date = dict(zip(filter_games["GAME_ID"], filter_games["date_time"]))
        id_to_matchup = dict(zip(filter_games["GAME_ID"], filter_games["MATCHUP"]))

        # Get path to cached files
        for tup in tqdm(
            filter_games.itertuples(), desc="Pulling game data", total=len(filter_games)
        ):
            # Get df paths
            results.append(file_from_game_id(game_id=tup.GAME_ID))

        # Read the data
        df = pd.read_parquet(results, engine="pyarrow")
        df["gameDate"] = df["gameId"].map(id_to_date)
        df["matchup"] = df["gameId"].map(id_to_matchup)

        games_data = []
        for (gameid, date), group in df.groupby(["gameId", "gameDate"]):
            games_data.append(
                GameData(
                    players=group,
                    game_id=gameid,
                    game_date=date,
                    matchup=id_to_matchup[gameid],
                )
            )

        return cls(games=games_data, teams=teams)

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
            self.season_df.groupby("playerSlug")["gameId"].transform("count") / ngames * 100
        )

    def get_team_stat(self):
        # Sum the counting stats
        count_stats = [
            "fieldGoalsMade",
            "fieldGoalsAttempted",
            "threePointersMade",
            "threePointersAttempted",
            "freeThrowsMade",
            "freeThrowsAttempted",
            "reboundsOffensive",
            "reboundsDefensive",
            "reboundsTotal",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "foulsPersonal",
            "points",
        ]
        id_stats = [
            "gameId",
            "teamId",
            "teamCity",
            "teamName",
            "teamTricode",
            "teamSlug",
            "gameDate",
            "seasonYear",
            "matchup",
        ]

        # Sum team stats for each game
        df = self.season_df.groupby(id_stats)[count_stats].sum().reset_index()

        # Get home/away team from matchup
        df[["home_team", "away_team"]] = df["matchup"].str.split(" ", expand=True).loc[:, [0, 2]]

        # Mark who won the game
        df["win"] = df["points"] == df.groupby(["gameId"])["points"].transform("max")

        # Reset index
        df = df.reset_index()

        # Make gameId number
        df["gameId"] = df["gameId"].astype(int)

        # Make data for home teams and away teams
        df["home_game"] = df["teamTricode"] == df["home_team"]
        home = df[df["home_game"]]
        away = df[~(df["home_game"])]

        # Add attribute to object
        self.team_stats_df = pd.concat([home, away], ignore_index=True)


def pull_game_data(game_id: str, sleep_min: float = SLEEP_MIN, sleep_max: float = SLEEP_MAX):
    # Get traditional box-score
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)
    box_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=30)

    # Random jitter to not get flagged by API.
    time.sleep(random.uniform(sleep_min, sleep_max))

    # Only player data. Merge on team ID, player family name, player first name
    adv_df = box_adv.player_stats.get_data_frame()
    box_df = box.player_stats.get_data_frame()
    merge_cols = list(adv_df.columns.intersection(box_df.columns))
    return box_df.merge(
        right=adv_df,
        on=merge_cols,
    )


def fill_nba_cache(from_date: datetime):
    logger.info("Filling the cache takes a while...")

    # Download all team games since 2022-2023 season
    for team in NbaTeams:
        logger.info(f"Filling cache for {team} in {CACHE_DIR}")
        _ = SeasonData.from_date_range(
            team=team,
            start_date=from_date,
            end_date=datetime.now(),
        )


def file_from_game_id(game_id: str) -> os.PathLike:

    cache_path = CACHE_DIR / f"{game_id}.parquet"

    if not cache_path.exists():
        df = pull_game_data(game_id=game_id)
        df.to_parquet(cache_path)

    return cache_path


if __name__ == "__main__":
    main()
