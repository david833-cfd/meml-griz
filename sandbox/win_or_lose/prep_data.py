from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from meml_griz import fetch


def get_nba_dataset() -> pd.DataFrame:
    # Fetch all NBA team data
    seasons = []
    nba_teams = [team for team in fetch.NbaTeams]
    seasons = fetch.SeasonData.from_date_range(
        teams=nba_teams,
        start_date=datetime(2021, 3, 1),
        end_date=datetime(2026, 5, 1),
        season_type=fetch.SeasonType.REGULAR_SEASON,
    )

    # Get all NBA games in one df
    df = seasons.team_stats_df.copy().drop_duplicates()
    print(f"Data shape after removing duplicates: {df.shape}")

    # Sort by game data and team
    df = df.sort_values(["teamSlug", "gameDate"]).reset_index(drop=True)

    # Create rolling mean of stats
    roll_df = df.groupby("teamSlug").rolling(window=5, closed="left").mean(numeric_only=True)

    # Add data to main df
    roll_df = roll_df.reset_index(level=0, drop=True)
    df[roll_df.columns] = roll_df
    df = df.dropna().reset_index(drop=True)
    return df


def get_preprocessor(
    df: pd.DataFrame, number_cols: list[str], cat_cols: list[str], bool_cols: list[str]
) -> ColumnTransformer:

    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    str_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numbers", num_trans, number_cols),
            ("categories", str_trans, cat_cols),
            ("bool_cols", "passthrough", bool_cols),
        ],
        remainder="drop",
    )


def get_features(
    number_cols: list[str], bool_cols: list[str], cat_cols: list[str], label_cols: list[str]
) -> list[str]:
    return list(set(number_cols + bool_cols + cat_cols) - set(label_cols))


def get_number_cols(
    df: pd.DataFrame, bools: set[str], cats: set[str], labels: str[str]
) -> tuple[list[str], list[str], list[str]]:
    drops = {
        "gameId",
        "teamId",
        "teamCity",
        "teamName",
        "gameDate",
        "teamTricode",
        "seasonYear",
        "matchup",
        "home_team",
        "away_team",
    }
    bools = {"home_game"}
    cats = {"teamSlug"}
    labels = {"win"}
    return list(set(df.columns.tolist()) - cats - drops - bools - labels)


def get_nba_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Make training data set (all seasons before the 2025 season)
    train = df.query("seasonYear < 2025")

    # Create testing data for the 2025 season
    test = df[(df["seasonYear"] == 2025)]

    return train, test
