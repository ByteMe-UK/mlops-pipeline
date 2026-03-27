"""
Model training with MLflow experiment tracking.

Trains a Random Forest classifier on the Wine Quality dataset.
Logs parameters, metrics, and model artifact to MLflow.
Saves model to models/classifier.joblib for the API.
"""

import os
import joblib

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pipeline.ingest import load

MODEL_PATH = "models/classifier.joblib"
EXPERIMENT_NAME = "wine-quality-classifier"

# Hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH = 10
RANDOM_STATE = 42


def train() -> dict:
    """
    Train the model, track with MLflow, save artifact.
    Returns evaluation metrics dict.
    """
    os.makedirs("models", exist_ok=True)

    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        })

        # Build pipeline: scaler + classifier
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc  = model.score(X_test, y_test)

        # Feature importances (from the RF inside the pipeline)
        importances = model.named_steps["clf"].feature_importances_
        top_feature = X.columns[np.argmax(importances)]

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy":  round(test_acc, 4),
            "top_feature":    top_feature,
        }

        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy":  test_acc,
        })
        mlflow.log_param("top_feature", top_feature)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save locally for the API
        joblib.dump(model, MODEL_PATH)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run: {run_id}")
        print(f"Train accuracy: {train_acc:.1%}")
        print(f"Test accuracy:  {test_acc:.1%}")
        print(f"Top feature:    {top_feature}")
        print(f"Model saved to: {MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train()
