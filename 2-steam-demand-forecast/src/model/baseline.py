"""Baseline models: naive persistence and seasonal lag-7."""

import numpy as np
import pandas as pd


class SeasonalPersistenceBaseline:
    """M1 Baseline: Seasonal persistence predicting y_t = y_{t-7} (or y_{t-1})."""

    def __init__(self, seasonal_lag: int = 7) -> None:
        self.seasonal_lag = seasonal_lag

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> "SeasonalPersistenceBaseline":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using 7-day lag, falling back to 1-day lag if missing."""
        lag7_col = f"sales_lag_{self.seasonal_lag}"
        lag1_col = "sales_lag_1"

        if lag7_col in df.columns:
            preds = df[lag7_col].copy()
            if lag1_col in df.columns:
                preds = preds.fillna(df[lag1_col])
            return preds.fillna(0.0).to_numpy()
        elif lag1_col in df.columns:
            return df[lag1_col].fillna(0.0).to_numpy()
        return np.zeros(len(df))
