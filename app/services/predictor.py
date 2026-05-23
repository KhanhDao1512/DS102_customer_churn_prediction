from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from src.features.features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "churn_model.joblib"


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@lru_cache(maxsize=1)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def _payload_to_frame(payload: CustomerInput) -> pd.DataFrame:
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload.dict()
    return pd.DataFrame([data])


def _heuristic_probability(df: pd.DataFrame) -> float:
    score = 0.18

    if int(df.loc[0, "SeniorCitizen"]) == 1:
        score += 0.06
    if df.loc[0, "Contract"] == "Month-to-month":
        score += 0.20
    if df.loc[0, "InternetService"] == "Fiber optic":
        score += 0.10
    if df.loc[0, "PaperlessBilling"] == "Yes":
        score += 0.04
    if df.loc[0, "PaymentMethod"] == "Electronic check":
        score += 0.10
    if df.loc[0, "TechSupport"] == "No":
        score += 0.05
    if df.loc[0, "OnlineSecurity"] == "No":
        score += 0.05
    if int(df.loc[0, "tenure"]) <= 12:
        score += 0.12
    if float(df.loc[0, "MonthlyCharges"]) >= 80:
        score += 0.05

    return max(0.02, min(score, 0.95))


def predict_churn(payload: CustomerInput) -> dict:
    raw_df = _payload_to_frame(payload)
    feature_df = build_features(raw_df)
    model = load_model()

    if model is not None:
        proba = float(model.predict_proba(feature_df)[:, 1][0])
        source = "model"
    else:
        proba = _heuristic_probability(feature_df)
        source = "heuristic"

    label = "Churn Risk" if proba >= 0.5 else "Retain"
    confidence = round(proba * 100, 1)

    return {
        "prediction": label,
        "probability": round(proba, 4),
        "confidence": confidence,
        "source": source,
    }
