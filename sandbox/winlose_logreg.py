from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from meml_griz import fetch, snifcheck


def main():
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

    # Clean data
    columns = df.columns.tolist()

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
    targets = {"win"}
    nums = set(columns) - cats - drops - bools - targets

    #  Sort by game data and team
    df = df.sort_values(["teamSlug", "gameDate"]).reset_index(drop=True)

    # Create rolling mean of stats
    roll_df = df.groupby("teamSlug").rolling(window=5, closed="left").mean(numeric_only=True)
    df[roll_df.columns] = roll_df.reset_index(level=0, drop=True)

    # Make training data set (all seasons before the 2025 season)
    train = df.query("seasonYear < 2025")

    # Create testing data for the grizzlies
    test = df[(df["seasonYear"] == 2025) & (df["teamSlug"] == "grizzlies")]

    # Make sklearn pipeline
    number_cols = list(nums)
    bool_cols = list(bools)
    cat_cols = list(cats)
    features = number_cols + bool_cols + cat_cols

    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    str_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", num_trans, number_cols),
            ("categories", str_trans, cat_cols),
            ("bool_cols", "passthrough", bool_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(random_state=99, class_weight="balanced", max_iter=100)),
        ]
    )

    # Fit the data on win or lose
    X = train[features]
    y = train[list(targets)].to_numpy().ravel()
    model = pipeline.fit(X, y)

    # Serialize model
    test_data = snifcheck.DataSamples(data=test, x_names=features, y_names=list(targets))
    train_data = snifcheck.DataSamples(data=train, x_names=features, y_names=list(targets))
    snifcheck.MLArtifact(model, test_data, train_data).save_model("winlose_logreg.pkl")


if __name__ == "__main__":
    main()
