from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from meml_griz import snifcheck


@pytest.fixture
def x_test():
    return np.linspace(0, 10, 25).reshape(-1, 1)


@pytest.fixture
def y_test(x_test):
    return 2 * x_test + 4


@pytest.fixture
def test_dataframe(x_test, y_test):
    return pd.DataFrame({"x": x_test.flatten(), "y": y_test.flatten()})


@pytest.fixture
def example_model(x_test, y_test):
    # Build simple linear model
    linreg = LinearRegression()
    linreg.fit(x_test, y_test)
    return linreg


@pytest.fixture
def serial_path(tmp_path: Path, example_model) -> Path:

    file_path = tmp_path / "example_model.pickle"

    joblib.dump(example_model, file_path)

    return file_path


def test_model_serialization(serial_path, example_model, test_dataframe):
    test_data = snifcheck.DataSamples(data=test_dataframe, x_names=["x"], y_names=["y"])

    # Serialize the model
    loader = snifcheck.MLArtifact(
        example_model, train=test_data, test=test_data, ml_lib=snifcheck.MLibrary.SKLEARN
    )
    loader.save_to_pickle(serial_path)

    # Deserialize the model
    loaded_model = loader.load_from_pickle(serial_path).model

    # Check that the loaded model is the same as the original
    assert np.allclose(loaded_model.coef_, example_model.coef_)
    assert np.allclose(loaded_model.intercept_, example_model.intercept_)


def test_snif_factory(serial_path, example_model, test_dataframe):

    test_data = snifcheck.DataSamples(data=test_dataframe, x_names=["x"], y_names=["y"])

    # Serialize the model
    loader = snifcheck.MLArtifact(
        example_model, train=test_data, test=test_data, ml_lib=snifcheck.MLibrary.SKLEARN
    )
    loader.save_to_pickle(model_path=serial_path)

    # Use Sniff to load the model from serialized path
    serial = snifcheck.BaseSniff.from_serialized_model(serial_path)

    # Load just the model
    memory = snifcheck.BaseSniff.from_model_artifact(loader)

    # Check that the loaded model is the same as the original
    assert np.allclose(serial.model.coef_, example_model.coef_)
    assert np.allclose(serial.model.intercept_, example_model.intercept_)
    assert np.allclose(serial.model.coef_, memory.model.coef_)
    assert np.allclose(serial.model.intercept_, memory.model.intercept_)
