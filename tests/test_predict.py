import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import ALL_FEATURES, predict_heart_disease


@pytest.fixture
def sample_input():
    """Create a valid sample using a row from the UCI test data."""
    return pd.DataFrame([{
        "age": 63,
        "sex": 1,
        "cp": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }], columns=ALL_FEATURES)


def test_prediction_output_structure(sample_input):
    result = predict_heart_disease(sample_input)

    assert len(result) == 1
    assert list(result.columns) == [
        "Predicted", "Disease_Probability"
    ]


def test_prediction_values(sample_input):
    result = predict_heart_disease(sample_input)

    assert result["Predicted"].isin([0, 1]).all()
    assert result["Disease_Probability"].between(0, 1).all()


def test_missing_feature_is_rejected(sample_input):
    incomplete_input = sample_input.drop(columns=["age"])

    with pytest.raises(ValueError, match="age"):
        predict_heart_disease(incomplete_input)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError, match="DataFrame"):
        predict_heart_disease([1, 2, 3])
