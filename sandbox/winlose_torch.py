from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn, optim

from meml_griz import fetch, hugtorch, snifcheck


class ClassNet(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=out_features),
        )

    def forward(self, x):
        return self.layers(x)


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

    # Sort by game data and team
    df = df.sort_values(["teamSlug", "gameDate"]).reset_index(drop=True)

    # Create rolling mean of stats
    roll_df = df.groupby("teamSlug").rolling(window=5, closed="left").mean(numeric_only=True)

    # Add data to main df
    roll_df = roll_df.reset_index(level=0, drop=True)
    df[roll_df.columns] = roll_df
    df = df.dropna().reset_index(drop=True)

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
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
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

    # Fit the data on win or lose
    X = preprocessor.fit_transform(train[features])
    y = train[list(targets)].to_numpy(dtype=float)

    # Train torch model
    model = ClassNet(in_features=X.shape[1], out_features=y.shape[1])
    hugtorch.train_model(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        loss_func=nn.BCEWithLogitsLoss(),
        x_train=X,
        y_train=y,
        epochs=100,
    )

    # Serialize model
    test_data = snifcheck.DataSamples(data=test, x_names=features, y_names=list(targets))
    train_data = snifcheck.DataSamples(data=train, x_names=features, y_names=list(targets))
    artifact = snifcheck.MLArtifact(
        model,
        test_data,
        train_data,
        snifcheck.MLibrary.TORCH,
        preprocessor,
    )
    artifact.save_to_pickle("winlose_torch.pkl")


if __name__ == "__main__":
    main()
