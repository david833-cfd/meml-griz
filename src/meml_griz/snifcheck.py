"""
Pseudo code for sniffcheck.py

1a. Write a method to take in the model object, test-X, and test-Y data.
1b. Write a method to take in the serialized model, test-X, and test-Y data.
    - Maybe have test data as a dataclass that is immutable?

2. Calculate metrics using sklearn function.
    - store these in attributes and methods.
    - use properties to compute them automatcially.

3. Create plotting methods. These are regular instance methods that the user can call.


"""

import os
import pickle
from abc import ABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    root_mean_squared_error,
)


@dataclass(frozen=True)
class DataSamples:
    data: pd.DataFrame
    x_names: list[str]
    y_names: list[str]

    @property
    def X(self):
        return self.data[self.x_names]

    @property
    def y(self):
        return self.data[self.y_names]


@dataclass
class MLArtifact:
    model: object
    test: DataSamples
    train: DataSamples

    @classmethod
    def load_model(cls, model_path: os.PathLike):
        with open(model_path, "rb") as fio:
            ml_model = pickle.load(fio)

        return cls(
            model=ml_model["model"],
            test=ml_model["test"],
            train=ml_model["train"],
        )

    def save_model(self, model_path: os.PathLike):

        out_path = Path(model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: save pytorch, gpytorch, and tensorflow models.

        with open(out_path, "wb") as fio:
            pickle.dump(
                {
                    "model": self.model,
                    "test": self.test,
                    "train": self.train,
                },
                fio,
            )


class BaseSniff(ABC):
    def __init__(self, model):
        self.model = model.model
        self.test_data = model.test
        self.train_data = model.train

    @classmethod
    def from_model_artifact(cls, components: MLArtifact):
        return cls(components)

    @classmethod
    def from_serialized_model(cls, model_path: os.PathLike):
        # Load in the serialized model
        model = MLArtifact.load_model(model_path)
        return cls.from_model_artifact(model)

    def get_prediction(self):
        return self.model.predict(self.test_data.X)

    @property
    def y_prediction(self) -> np.ndarray:
        return self.get_prediction()

    @property
    def y_true(self) -> np.ndarray:
        return np.array(self.test_data.y).flatten()


class ClassifierSniff(BaseSniff):
    @property
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_prediction)

    @property
    def precision(self):
        return precision_score(self.y_true, self.y_prediction, average="weighted")

    @property
    def classification_report(self):
        return classification_report(self.y_true, self.y_prediction)


class RegressorSniff(BaseSniff):
    @property
    def r2_score(self):
        return r2_score(self.y_true, self.y_prediction)

    @property
    def mean_absolute_error(self):
        return mean_absolute_error(self.y_true, self.y_prediction)

    @property
    def mean_squared_error(self):
        return mean_squared_error(self.y_true, self.y_prediction)

    @property
    def root_mean_squared_error(self):
        return root_mean_squared_error(self.y_true, self.y_prediction)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    x = np.linspace(0, 10, 51).reshape(-1, 1)
    y = np.sin(x) + 2 * x + np.random.rand(x.shape[0], 1) * 5

    model = LinearRegression()
    model.fit(x, y)
    pred_x = np.linspace(0, 10, 10).reshape(-1, 1)
    pred_y = model.predict(pred_x)

    build = MLArtifact(
        model=model,
        train=DataSamples(
            data=pd.DataFrame({"x": x.flatten(), "y": y.flatten()}),
            x_names=["x"],
            y_names=["y"],
        ),
        test=DataSamples(
            data=pd.DataFrame({"x": pred_x.flatten(), "y": pred_y.flatten()}),
            x_names=["x"],
            y_names=["y"],
        ),
    )

    sniff = RegressorSniff.from_model_artifact(components=build)

    # Extract properties into a list of dictionaries
    df = pd.DataFrame(
        [
            {
                # "accuracy": sniff.accuracy,
                # "precision": sniff.precision,
                "r2_score": sniff.r2_score,
                "mae": sniff.mean_absolute_error,
                "mse": sniff.mean_squared_error,
                "rmse": sniff.root_mean_squared_error,
            }
        ]
    )
    print(df)

    plt.plot(x, y, "o")
    plt.plot(pred_x, pred_y, "r", label="Predicted")
    plt.show()


if __name__ == "__main__":
    main()
