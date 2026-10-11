from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_heart_disease_model.joblib"

NUMERICAL_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_model():
    """Load the saved, trusted scikit-learn pipeline."""
    return joblib.load(MODEL_PATH)


def predict_heart_disease(input_data, model=None):
    """Return predicted classes and estimated disease probabilities."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame.")

    missing_features = set(ALL_FEATURES) - set(input_data.columns)
    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )

    input_data = input_data[ALL_FEATURES].copy()

    if model is None:
        model = load_model()

    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    return pd.DataFrame({
        "Predicted": predictions,
        "Disease_Probability": probabilities
    }, index=input_data.index)
