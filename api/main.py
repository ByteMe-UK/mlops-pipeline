"""
FastAPI prediction endpoint for the Wine Quality classifier.

Loads the trained model from models/classifier.joblib and
serves predictions via a REST API.
"""

import os
import joblib
import subprocess
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

MODEL_PATH = "models/classifier.joblib"

app = FastAPI(
    title="Wine Quality Classifier API",
    description="Predicts whether a red wine is good (quality >= 6) or bad (quality < 6).",
    version="1.0.0",
)


def _load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found — running pipeline...")
        subprocess.run([sys.executable, "run_pipeline.py"], check=True)
    return joblib.load(MODEL_PATH)


model = _load_model()


class WineFeatures(BaseModel):
    fixed_acidity:        float = Field(..., example=7.4, description="Fixed acidity (g/dm³)")
    volatile_acidity:     float = Field(..., example=0.70, description="Volatile acidity (g/dm³)")
    citric_acid:          float = Field(..., example=0.0,  description="Citric acid (g/dm³)")
    residual_sugar:       float = Field(..., example=1.9,  description="Residual sugar (g/dm³)")
    chlorides:            float = Field(..., example=0.076, description="Chlorides (g/dm³)")
    free_sulfur_dioxide:  float = Field(..., example=11.0, description="Free SO₂ (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., example=34.0, description="Total SO₂ (mg/dm³)")
    density:              float = Field(..., example=0.9978, description="Density (g/cm³)")
    ph:                   float = Field(..., example=3.51, description="pH")
    sulphates:            float = Field(..., example=0.56, description="Sulphates (g/dm³)")
    alcohol:              float = Field(..., example=9.4,  description="Alcohol (% vol)")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    features: dict


@app.get("/", response_class=HTMLResponse)
def root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict(wine: WineFeatures):
    """Predict wine quality from physicochemical features."""
    features = np.array([[
        wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
        wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
        wine.total_sulfur_dioxide, wine.density, wine.ph,
        wine.sulphates, wine.alcohol,
    ]])

    prediction = int(model.predict(features)[0])
    confidence = float(model.predict_proba(features)[0][prediction])
    label = "good" if prediction == 1 else "bad"

    return PredictionResponse(
        prediction=prediction,
        label=label,
        confidence=round(confidence, 4),
        features=wine.model_dump(),
    )
