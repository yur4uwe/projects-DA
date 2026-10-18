"""Unit tests for feature engineering, lags, and calendar transformations."""

import pandas as pd

from src.features.calendar import create_calendar_features
from src.features.cleaner import align_daily_series
from src.features.lags import create_lag_features
from src.features.rolling import create_rolling_features


def test_align_daily_series_gap_fill() -> None:
    # Intentionally missing dates 2024-01-02 and 2024-01-03
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-04"],
            "avg_players": [100.0, 400.0],
        }
    )
    aligned = align_daily_series(df, date_col="date")
    assert len(aligned) == 4
    assert pd.to_datetime(aligned["date"].iloc[1]) == pd.to_datetime("2024-01-02")
    assert pd.to_datetime(aligned["date"].iloc[2]) == pd.to_datetime("2024-01-03")
    # Linear interpolation check
    assert aligned["avg_players"].iloc[1] == 200.0
    assert aligned["avg_players"].iloc[2] == 300.0


def test_lag_features_shift_integrity() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "demand": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )
    with_lags = create_lag_features(
        df, target_col="demand", prefix="sales", lags=[1, 7]
    )

    # lag 1 at index 1 should be value of index 0
    assert pd.isna(with_lags["sales_lag_1"].iloc[0])
    assert with_lags["sales_lag_1"].iloc[1] == 10
    assert with_lags["sales_lag_1"].iloc[5] == 50

    # lag 7 at index 7 should be value of index 0
    assert pd.isna(with_lags["sales_lag_7"].iloc[6])
    assert with_lags["sales_lag_7"].iloc[7] == 10


def test_rolling_features_no_target_leakage() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "demand": [100.0] * 10,
        }
    )
    with_roll = create_rolling_features(
        df, target_col="demand", prefix="sales", windows=[7]
    )

    # Day 0 shifted target is NaN, rolling mean at index 0 should be NaN or 0
    # Day 1 shifted target is 100.0, rolling mean at index 1 is 100.0
    assert with_roll["sales_rolling_7"].iloc[1] == 100.0


def test_calendar_cyclical_bounds() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=365, freq="D"),
        }
    )
    with_cal = create_calendar_features(df, date_col="date")

    assert with_cal["sin_day_of_week"].min() >= -1.0
    assert with_cal["sin_day_of_week"].max() <= 1.0
    assert with_cal["cos_day_of_year"].min() >= -1.0
    assert with_cal["cos_day_of_year"].max() <= 1.0
    assert set(with_cal["is_weekend"].unique()).issubset({0, 1})
