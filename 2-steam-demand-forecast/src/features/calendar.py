"""Calendar and cyclical trigonometric feature extraction."""

import numpy as np
import pandas as pd


def create_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate calendar and cyclical trigonometric features."""
    out = df.copy()
    dates = pd.to_datetime(out[date_col])

    # Day of week
    dow = dates.dt.dayofweek
    out["day_of_week"] = dow
    out["is_weekend"] = dow.isin([4, 5, 6]).astype(
        int
    )  # Fri, Sat, Sun are peak gaming days

    # Month and Day of Year
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["quarter"] = dates.dt.quarter

    # Cyclical trigonometric transformations
    out["sin_day_of_week"] = np.sin(2 * np.pi * dow / 7.0)
    out["cos_day_of_week"] = np.cos(2 * np.pi * dow / 7.0)

    out["sin_day_of_year"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["cos_day_of_year"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    return out
