"""
This module trains three regression models on the Diabetes dataset
and tracks every experiment with MLflow Experiment Tracking.

For each model, the script logs hyperparameters, evaluation metrics
and the serialized model artifact so runs can be compared in MLflow UI.
"""

# MLOps concept:
# Experiment Tracking records what we tried and how well each model performed.
# MLflow stores hyperparameters, metrics, tags and model artifacts.
# Run `mlflow ui` to compare the experiments.


import logging
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "train_features_v1.csv"
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "diabetes-progression"
TARGET = "target"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_features(path: Path) -> pd.DataFrame:
    """Load the processed Diabetes dataset from CSV."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first:\n"
            "python src/data/load_data.py\n"
            "python src/data/preprocess.py"
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in the dataset."
        )

    if df.empty:
        raise ValueError("The input dataset is empty.")

    logger.info(
        "Loaded %d rows and %d columns from %s",
        len(df),
        df.shape[1],
        path,
    )

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate input features (X) from target progression values (y)
    and split the dataset into training and testing sets.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    logger.info("X represents the patient features.")
    logger.info("y represents the diabetes progression measure.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger.info("Training samples: %d", len(X_train))
    logger.info("Test samples: %d", len(X_test))

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# MLflow experiment
# ---------------------------------------------------------------------------

def run_experiment(
    model,
    model_type: str,
    params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Train one regression model and log its parameters,
    metrics, tags and model artifact to MLflow.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"

    with mlflow.start_run(run_name=run_name):

        # Tags
        mlflow.set_tags({
            "model_type": model_type,
            "dataset": INPUT_PATH.name,
            "stage": "training",
            "target": TARGET,
        })

        # Hyperparameters
        mlflow.log_params(params)

        # Training
        logger.info("[%s] Training model...", model_type)

        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Evaluation
        metrics = compute_metrics(y_test, predictions)

        mlflow.log_metrics(metrics)

        logger.info(
            "[%s] RMSE=%.4f | MAE=%.4f | R2=%.4f",
            model_type,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

        # Model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        logger.info("[%s] Model artifact logged.", model_type)

    return {
        "Model": model_type,
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R2": metrics["r2"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run and compare three regression models."""

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment: %s", EXPERIMENT_NAME)

    # Load dataset
    df = load_features(INPUT_PATH)

    # Feature and target split
    X_train, X_test, y_train, y_test = split_data(df)

    # Define models
    models = {
        "LinearRegression": (
            LinearRegression(),
            {"fit_intercept": True},
        ),

        "RandomForest": (
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
            ),
            {
                "n_estimators": 100,
                "random_state": RANDOM_STATE,
            },
        ),

        "GradientBoosting": (
            GradientBoostingRegressor(
                random_state=RANDOM_STATE,
            ),
            {
                "random_state": RANDOM_STATE,
            },
        ),
    }

    # Train and log all models
    for model_type, (model, params) in models.items():
        run_experiment(
            model=model,
            model_type=model_type,
            params=params,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
    
    logger.info(
        "Training completed. Open MLflow UI with:\n"
        "mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )


if __name__ == "__main__":
    main()