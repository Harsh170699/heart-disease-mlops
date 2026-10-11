import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import ALL_FEATURES, load_model, predict_heart_disease

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Academic ML demonstration using the UCI Cleveland "
        "Heart Disease dataset. Not intended for clinical use."
    ),
    version="1.0.0",
)


class PatientFeatures(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model could not be loaded."
        )


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        patient_data = pd.DataFrame(
            [patient.model_dump()],
            columns=ALL_FEATURES
        )
        result = predict_heart_disease(patient_data)

        predicted_class = int(result["Predicted"].iloc[0])
        probability = float(
            result["Disease_Probability"].iloc[0]
        )

        return {
            "predicted_class": predicted_class,
            "prediction_label": (
                "heart_disease" if predicted_class == 1
                else "no_heart_disease"
            ),
            "estimated_probability": round(probability, 4),
            "disclaimer": (
                "For educational demonstration only; "
                "not a medical diagnosis."
            ),
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )
