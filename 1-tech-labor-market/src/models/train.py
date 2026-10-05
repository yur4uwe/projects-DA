from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .metrics import compute_regression_metrics
from .pipeline import create_preprocessor, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate(
    data_path: Path,
    models_dir: Path,
    target_col: str = "salary_mid_usd",
) -> dict[str, Any]:
    """Load labeled dataset, evaluate baselines vs LightGBM, and export the best model."""
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run builder first.")

    df = pd.read_parquet(data_path)
    # Ensure target exists and is strictly valid
    df = df[df[target_col].notnull()].copy()

    if len(df) == 0:
        raise ValueError("No labeled salary rows found in dataset.")

    logger.info(f"Loaded {len(df)} labeled salary records for training.")

    num_cols, cat_cols, skill_cols = get_feature_columns(df)
    feature_cols = num_cols + cat_cols + skill_cols

    X = df[feature_cols]
    y = df[target_col].values

    # Define candidates: Median Baseline -> Linear Regression (Ridge) -> LightGBM
    models: dict[str, Pipeline] = {
        "Dummy (Median)": Pipeline(
            steps=[
                ("prep", create_preprocessor(num_cols, cat_cols, skill_cols, scale_numeric=False)),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "Linear Regression (Ridge)": Pipeline(
            steps=[
                ("prep", create_preprocessor(num_cols, cat_cols, skill_cols, scale_numeric=True)),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "LightGBM Regressor": Pipeline(
            steps=[
                ("prep", create_preprocessor(num_cols, cat_cols, skill_cols, scale_numeric=False)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=15,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }

    # Cross-validation setup
    n_samples = len(df)
    n_splits = min(5, n_samples) if n_samples >= 4 else 2
    logger.info(f"Running {n_splits}-fold cross-validation...")

    cv_results = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, pipe in models.items():
        try:
            preds = cross_val_predict(pipe, X, y, cv=kf)
            metrics = compute_regression_metrics(y, preds)
            cv_results[name] = metrics
            logger.info(f"[{name}] -> MAE: ${metrics['MAE']} | RMSE: ${metrics['RMSE']} | R2: {metrics['R2']}")
        except Exception as exc:
            logger.warning(f"Failed CV for {name}: {exc}")
            cv_results[name] = {"MAE": float("inf"), "RMSE": float("inf"), "R2": -1.0}

    # Select best model by lowest MAE
    best_name = min(cv_results, key=lambda m: cv_results[m]["MAE"])
    best_pipeline = models[best_name]
    logger.info(f"Best performing model: {best_name} (MAE: ${cv_results[best_name]['MAE']})")

    # Fit best model on entire dataset
    best_pipeline.fit(X, y)

    # Save model artifact and metadata
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "salary_model.joblib"
    meta_path = models_dir / "metadata.json"

    joblib.dump(best_pipeline, model_path)

    metadata = {
        "best_model": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_splits": n_splits,
        "cv_metrics": cv_results,
        "features": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "skills": skill_cols,
        },
        "target": target_col,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Saved metadata to: {meta_path}")

    return metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    data_file = project_root / "data" / "processed" / "salary_labeled_features.parquet"
    models_folder = project_root / "models"

    train_and_evaluate(data_file, models_folder)
