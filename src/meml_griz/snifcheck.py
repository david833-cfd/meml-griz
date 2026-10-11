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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
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

from meml_griz.settings import logger


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


class BaseSerializer(ABC):
    suffix: str = ""

    @abstractmethod
    def save(self, model, path: os.PathLike):
        pass

    @abstractmethod
    def load(self, path: os.PathLike):
        pass


class TorchSerializer(BaseSerializer):
    suffix: str = ".model.pt"

    def save(self, model, path: os.PathLike):
        import torch

        torch.save(model, path.with_suffix(self.suffix))

    def load(self, path: os.PathLike):
        import torch

        return torch.load(path.with_suffix(self.suffix), weights_only=False)


class SklearnSerializer(BaseSerializer):
    suffix: str = ".skmodel.pickle"

    def save(self, model, model_path: os.PathLike):

        with open(model_path.with_suffix(self.suffix), "wb") as fio:
            pickle.dump(model, fio)

    def load(self, model_path: os.PathLike):
        with open(model_path.with_suffix(self.suffix), "rb") as fio:
            return pickle.load(fio)


class MLibrary(str, Enum):
    SKLEARN = "sklearn"
    TORCH = "torch"

    @property
    def serializer(self):
        lib_to_serializer = {
            MLibrary.SKLEARN: SklearnSerializer(),
            MLibrary.TORCH: TorchSerializer(),
        }
        return lib_to_serializer[self]


@dataclass
class MLArtifact:
    model: object
    test: DataSamples
    train: DataSamples
    ml_lib: MLibrary
    preprocessor: object = None

    @property
    def serializer(self):
        logger.debug(f"{self.ml_lib=}")
        logger.debug(f"{self.ml_lib.serializer=}")
        return self.ml_lib.serializer

    @staticmethod
    def get_meta_path(path: os.PathLike):
        return Path(path).with_suffix(".meta.pickle")

    @classmethod
    def load_from_pickle(cls, path_stem: os.PathLike):
        # Make sure path is a pathlib obj
        path_stem = Path(path_stem)

        # Get the meta data path
        metadata_path = cls.get_meta_path(path_stem)

        with open(metadata_path, "rb") as f:
            meta_data = pickle.load(f)

        artifact = cls(
            model=None,
            ml_lib=MLibrary(meta_data["ml_lib"]),
            test=meta_data["test"],
            train=meta_data["train"],
            preprocessor=meta_data.get("preprocessor", None),
        )

        # Load the model with serializer
        artifact.model = artifact.serializer.load(path_stem)

        return artifact

    def save_to_pickle(self, path_stem: os.PathLike):
        # Define path to write
        out_path = Path(path_stem)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        logger.debug(f"Serializing model to: {out_path=}")
        self.serializer.save(self.model, out_path)

        # Define meta dict
        meta_data = {
            "ml_lib": self.ml_lib.value,
            "test": self.test,
            "train": self.train,
            "preprocessor": self.preprocessor,
        }

        # Pickle the data
        metadata_path = self.get_meta_path(path_stem)
        with open(metadata_path, "wb") as f:
            pickle.dump(meta_data, f)


class BaseSniff(ABC):
    def __init__(self, model):
        self.model = model.model
        self.test_data = model.test
        self.train_data = model.train
        self.ml_lib = model.ml_lib
        self.preprocessor = getattr(model, "preprocessor", None)

    @classmethod
    def from_model_artifact(cls, components: MLArtifact):
        return cls(components)

    @classmethod
    def from_serialized_model(cls, path_stem: os.PathLike):
        # Load in the serialized model
        model = MLArtifact.load_from_pickle(path_stem)
        return cls.from_model_artifact(model)

    def get_prediction(self):

        if self.ml_lib == MLibrary.SKLEARN:
            return self.model.predict(self.test_data.X)
        elif self.ml_lib == MLibrary.TORCH:
            import torch

            # Use GPU if available
            device = "mps" if torch.mps.is_available() else "cpu"

            # Pre-process the data
            if self.preprocessor is not None:
                x_input = self.preprocessor.transform(self.test_data.X)
            else:
                x_input = self.test_data.X.to_numpy()

            # Convert test X data to torch tensor.
            x_tensor = torch.from_numpy(x_input.astype(np.float32))

            # Put model and X on same device
            self.model.to(device)
            x_tensor = x_tensor.to(device)
            logger.debug(f"Using {device=} for foward pass.")

            # Put into eval model
            self.model.eval()

            # Turn off grad tracking. Get prediction as np array.
            with torch.no_grad():
                y_pred = self.model(x_tensor).cpu().numpy().flatten()

            logger.debug("Made the torch prediction.")

            # If classifier, then convert data to int
            if isinstance(self, ClassifierSniff):
                return (y_pred >= 0.0).astype(int)
            return y_pred

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
        ml_lib=MLibrary.SKLEARN,
    )

    sniff = RegressorSniff.from_model_artifact(components=build)

    # Extract properties into a list of dictionaries
    df = pd.DataFrame(
        [
            {
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
