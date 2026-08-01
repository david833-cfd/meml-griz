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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import dump, load
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
class TestData:
    X: np.ndarray
    y: np.ndarray


class ModelLoader(ABC):
    def __init__(self, model, model_path: Path):
        self.model = model
        self.model_path = model_path

    @abstractmethod
    def deserialize(self):
        pass

    @abstractmethod
    def serialize(self):
        pass


class SklearnLoader(ModelLoader):
    def deserialize(self):
        return load(self.model_path)

    def serialize(self):
        return dump(self.model, self.model_path)


class BaseSniff(ABC):
    def __init__(self, model, test_data: TestData):
        self.model = model
        self.test_data = test_data

    @classmethod
    def from_model_object(cls, model, test_data: TestData):
        return cls(model, test_data)

    @classmethod
    def from_serialized_model(cls, model_loader: ModelLoader, test_data: TestData):
        # Load in the serialized model
        model = model_loader.deserialize()
        return cls.from_model_object(model, test_data)

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

    sniff = RegressorSniff.from_model_object(model=model, test_data=TestData(X=x, y=y))

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
