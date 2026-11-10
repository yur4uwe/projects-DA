"""Market baseline pricing model and deal tier thresholding."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

BASELINE_CATEGORICAL = [
    "Company",
    "TypeName",
    "Cpu_Tier",
    "Gpu_Tier",
    "Primary_Storage_Type",
    "OpSys",
]

BASELINE_NUMERICAL = [
    "Inches",
    "Ram_GB",
    "Weight_kg",
    "Cpu_Freq_GHz",
    "Primary_Storage_GB",
    "Total_Storage_GB",
    "PPI",
    "Is_Touchscreen",
    "Is_IPS",
    "Has_Dedicated_Gpu",
    "Has_SSD",
]


class MarketBaselineModel:
    """Estimates fair secondary market baseline price given hardware specifications."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    BASELINE_CATEGORICAL,
                ),
                ("num", "passthrough", BASELINE_NUMERICAL),
            ]
        )
        self.pipeline = Pipeline(
            steps=[
                ("prep", self.preprocessor),
                (
                    "reg",
                    GradientBoostingRegressor(
                        n_estimators=120,
                        max_depth=4,
                        learning_rate=0.08,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarketBaselineModel":
        self.pipeline.fit(X[BASELINE_CATEGORICAL + BASELINE_NUMERICAL], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.round(
            self.pipeline.predict(X[BASELINE_CATEGORICAL + BASELINE_NUMERICAL]), 2
        )

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        logger.info("Saved baseline market model to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "MarketBaselineModel":
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance


def compute_out_of_fold_baseline(
    df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> tuple[np.ndarray, MarketBaselineModel]:
    """Compute unbiased out-of-fold market baseline estimates for historical training data."""
    X = df[BASELINE_CATEGORICAL + BASELINE_NUMERICAL]
    y = df["asking_price"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(df))

    for train_idx, val_idx in kf.split(X):
        fold_model = MarketBaselineModel(random_state=random_state)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_predictions[val_idx] = fold_model.predict(X.iloc[val_idx])

    # Fit final baseline model on all records for deployment inference
    final_model = MarketBaselineModel(random_state=random_state)
    final_model.fit(X, y)

    return np.round(oof_predictions, 2), final_model


def assign_deal_tier(
    price_ratio: float, great_threshold: float = 0.85, fair_threshold: float = 1.15
) -> str:
    """Classify listing into deal tiers based on asking price vs market baseline ratio."""
    if price_ratio <= great_threshold:
        return "Great Deal"
    if price_ratio <= fair_threshold:
        return "Fair Price"
    return "Overpriced"
