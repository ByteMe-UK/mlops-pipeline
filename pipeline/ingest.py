"""
Data ingestion — downloads the UCI Wine Quality dataset and prepares features.

Dataset: Red Wine Quality (UCI ML Repository)
Task:    Binary classification — quality >= 6 → good (1), quality < 6 → bad (0)
"""

import io
import os

import pandas as pd
import requests

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/wine-quality/winequality-red.csv"
)
DATA_PATH = "data/winequality-red.csv"

FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET_COL = "quality"


def download_data() -> pd.DataFrame:
    """Download raw dataset from UCI. Caches locally to data/."""
    if os.path.exists(DATA_PATH):
        print(f"Using cached data at {DATA_PATH}")
        return pd.read_csv(DATA_PATH, sep=";")

    print("Downloading Wine Quality dataset from UCI...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()

    os.makedirs("data", exist_ok=True)
    with open(DATA_PATH, "w") as f:
        f.write(resp.text)

    return pd.read_csv(io.StringIO(resp.text), sep=";")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and binarise the target.
    quality >= 6 → 1 (good), quality < 6 → 0 (bad)
    """
    X = df[FEATURE_COLS].copy()
    y = (df[TARGET_COL] >= 6).astype(int)
    return X, y


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Full ingestion: download + prepare. Returns (X, y)."""
    df = download_data()
    X, y = prepare_features(df)
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class balance: {y.mean():.1%} good wines")
    return X, y
