from pathlib import Path

import numpy as np
from loguru import logger
from prep_data import (
    get_features,
    get_nba_dataset,
    get_nba_train_test,
    get_number_cols,
    get_preprocessor,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from meml_griz import snifcheck

PWD = Path(__file__).parent


def train_logreg():
    df = get_nba_dataset()

    # Get train & test data
    train, test = get_nba_train_test(df=df)

    # Set column types
    bool_cols = ["home_game"]
    cat_cols = ["teamSlug", "seasonYear", "matchup", "home_team", "away_team"]
    label_cols = ["win"]

    # Set the numeric, boolean, and categorical cols
    number_cols = get_number_cols(
        df=df, bools=bool_cols, cats=cat_cols, labels=label_cols
    )
    logger.info(f"{number_cols=}")
    features = get_features(
        number_cols=number_cols,
        bool_cols=bool_cols,
        cat_cols=cat_cols,
        label_cols=label_cols,
    )
    # Get the preprocessor/col transformer
    preprocessor = get_preprocessor(
        df=df, number_cols=number_cols, cat_cols=cat_cols, bool_cols=bool_cols
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(0.9)),
            # ("model", LogisticRegression(random_state=99, class_weight="balanced", max_iter=500)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=99, min_samples_leaf=5
                ),
            ),
        ]
    )

    # Fit the data on win or lose
    X = train[features]
    y = train[list(label_cols)].to_numpy().ravel()

    # Random search for hyper params
    param_distributions = {
        "preprocessor__numbers__scaler__with_mean": [True, False],
        "model__n_estimators": np.arange(50, 250),
        "model__max_depth": [None, 5, 10, 20, 50],
        "model__min_samples_split": np.arange(3, 11),
        "model__min_samples_leaf": np.arange(3, 11),
    }

    # K-fold splits
    kfold = 5

    # Train the model with rand search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,  # Number of random parameter settings sampled
        cv=kfold,  # 5-fold cross-validation
        scoring="accuracy",
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,
        verbose=1,
    )
    model = random_search.fit(X, y)

    # Evaluate the best model configuration
    print("\n--- Hyperparameter Tuning Results ---")
    print(f"Best Parameters found: {random_search.best_params_}")
    print(f"Best Cross-Validation Score: {random_search.best_score_:.4f}")

    # model = pipeline.fit(X, y)

    # Serialize model
    test_data = snifcheck.DataSamples(
        data=test, x_names=features, y_names=list(label_cols)
    )
    train_data = snifcheck.DataSamples(
        data=train, x_names=features, y_names=list(label_cols)
    )
    snifcheck.MLArtifact(
        model, test_data, train_data, snifcheck.MLibrary.SKLEARN
    ).save_to_pickle(path_stem="winlose_sklearn")


def main():
    train_logreg()


if __name__ == "__main__":
    main()
