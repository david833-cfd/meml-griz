from pathlib import Path

from prep_data import (
    get_features,
    get_nba_dataset,
    get_nba_train_test,
    get_number_cols,
    get_preprocessor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from meml_griz import snifcheck

PWD = Path(__file__).parent


def train_logreg():
    df = get_nba_dataset()

    # Get train & test data
    train, test = get_nba_train_test(df=df)

    # Set column types
    bool_cols = ["home_game"]
    cat_cols = ["teamSlug"]
    label_cols = ["win"]

    # Set the numeric, boolean, and categorical cols
    number_cols = get_number_cols(df=df, bools=bool_cols, cats=cat_cols, labels=label_cols)
    features = get_features(
        number_cols=number_cols, bool_cols=bool_cols, cat_cols=cat_cols, label_cols=label_cols
    )
    # Get the preprocessor/col transformer
    preprocessor = get_preprocessor(
        df=df, number_cols=number_cols, cat_cols=cat_cols, bool_cols=bool_cols
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(random_state=99, class_weight="balanced", max_iter=250)),
        ]
    )

    # Fit the data on win or lose
    X = train[features]
    y = train[list(label_cols)].to_numpy().ravel()
    model = pipeline.fit(X, y)

    # Serialize model
    test_data = snifcheck.DataSamples(data=test, x_names=features, y_names=list(label_cols))
    train_data = snifcheck.DataSamples(data=train, x_names=features, y_names=list(label_cols))
    snifcheck.MLArtifact(model, test_data, train_data, snifcheck.MLibrary.SKLEARN).save_to_pickle(
        path_stem="winlose_sklearn"
    )


if __name__ == "__main__":
    main()
