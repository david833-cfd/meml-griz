from pathlib import Path

from prep_data import (
    get_features,
    get_nba_dataset,
    get_nba_train_test,
    get_number_cols,
    get_preprocessor,
)
from torch import nn, optim

from meml_griz import hugtorch, snifcheck

PWD = Path(__file__).parent


class ClassNet(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=out_features),
        )

    def forward(self, x):
        return self.layers(x)


def train_torch():
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

    # Fit the data on win or lose
    X = preprocessor.fit_transform(train[features])
    y = train[list(label_cols)].to_numpy(dtype=float)

    # Train torch model
    model = ClassNet(in_features=X.shape[1], out_features=y.shape[1])
    hugtorch.train_model(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        loss_func=nn.BCEWithLogitsLoss(),
        x_train=X,
        y_train=y,
        epochs=10,
    )

    # Serialize model
    test_data = snifcheck.DataSamples(data=test, x_names=features, y_names=list(label_cols))
    train_data = snifcheck.DataSamples(data=train, x_names=features, y_names=list(label_cols))
    artifact = snifcheck.MLArtifact(
        model,
        test_data,
        train_data,
        snifcheck.MLibrary.TORCH,
        preprocessor,
    )
    artifact.save_to_pickle("winlose_torch.pkl")


if __name__ == "__main__":
    train_torch()
