"""Autoregressive lag feature extraction conforming to course Listing 5."""

import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    prefix: str = "sales",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create autoregressive lag features for time series.

    Strictly shifts backwards so that feature at time t uses values from t-k.
    """
    out = df.copy()
    lag_steps = lags or [1, 2, 7, 14, 30]

    for lag in lag_steps:
        col_name = f"{prefix}_lag_{lag}"
        out[col_name] = out[target_col].shift(lag)

    return out
