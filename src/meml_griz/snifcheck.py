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
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
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


class Sniff:
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
        pass
