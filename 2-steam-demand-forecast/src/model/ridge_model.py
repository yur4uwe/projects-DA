"""Autoregressive regularized linear model (M2 Ridge)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AutoregressiveRidgeModel:
    """M2 Autoregressive Ridge Regression with standardized feature scaling."""

    DEFAULT_FEATURES = [
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_30",
        "sales_rolling_7",
        "sales_rolling_std_7",
        "sales_rolling_30",
        "sales_momentum_7_30",
        "is_weekend",
        "sin_day_of_week",
        "cos_day_of_week",
        "is_steam_sale",
        "discount_depth",
    ]

    def __init__(
        self, alpha: float = 10.0, feature_cols: list[str] | None = None
    ) -> None:
        self.alpha = alpha
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=42)),
            ]
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> "AutoregressiveRidgeModel":
        available_cols = [c for c in self.feature_cols if c in X.columns]
        self.pipeline.fit(X[available_cols], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        available_cols = [c for c in self.feature_cols if c in X.columns]
        preds = self.pipeline.predict(X[available_cols])
        return np.clip(preds, 0.0, None)
