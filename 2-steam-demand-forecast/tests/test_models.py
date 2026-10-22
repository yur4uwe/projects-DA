"""Unit tests for metrics, baseline, and model registry."""

import numpy as np
import pandas as pd

from src.model.baseline import SeasonalPersistenceBaseline
from src.model.metrics import calculate_metrics
from src.model.registry import ModelRegistry


def test_metrics_calculation() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == round(20.0 / 3.0, 2)
    # Total error: 10 + 10 + 0 = 20. Total actual: 600. WAPE = 20 / 600 = 0.0333
    assert abs(metrics["wape"] - (20.0 / 600.0)) < 1e-3
    assert metrics["r2"] > 0.95


def test_persistence_baseline() -> None:
    df = pd.DataFrame(
        {
            "sales_lag_7": [150.0, 250.0],
            "sales_lag_1": [140.0, 240.0],
        }
    )
    m1 = SeasonalPersistenceBaseline(seasonal_lag=7)
    preds = m1.predict(df)
    assert np.array_equal(preds, np.array([150.0, 250.0]))


def test_model_registry_crud(tmp_path) -> None:
    test_db = tmp_path / "test_experiments.db"
    reg = ModelRegistry(db_path=test_db)

    metrics = {"mae": 50.0, "rmse": 75.0, "wape": 0.08, "r2": 0.92}
    reg.record_run("M_TEST", "Test Model", metrics, description="Testing CRUD")

    exp = reg.get_experiment("M_TEST")
    assert exp is not None
    assert exp["milestone"] == "M_TEST"
    assert exp["mae"] == 50.0

    df = reg.list_experiments()
    assert len(df) == 1
    assert df["milestone"].iloc[0] == "M_TEST"


def test_simulate_promotional_campaign_continuity() -> None:
    import joblib
    from pathlib import Path
    from src.model.predict import simulate_promotional_campaign

    model_path = Path("models/indie_demand_model.joblib")
    data_path = Path("data/processed/367520_features.parquet")
    if not model_path.exists() or not data_path.exists():
        return

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    res = simulate_promotional_campaign(
        history_df=df,
        model=model,
        horizon_days=21,
        start_date="2026-10-25",
        discount_depth=0.50,
        sale_duration_days=7,
    )

    fdf = res["forecast_df"]
    # Peak CCU for Hollow Knight must remain bounded and not explode past 20k
    assert res["peak_promo_ccu"] < 18000.0
    assert res["peak_promo_ccu"] > res["recent_baseline_ccu"]

    # Post-sale tail must decay back to organic baseline
    post_sale_days = fdf[fdf["date"] > pd.to_datetime("2026-10-31").date()]
    assert not post_sale_days.empty
    # The last simulated day should be close to organic baseline (< 1.25x recent_baseline)
    assert post_sale_days["predicted_ccu"].iloc[-1] < res["recent_baseline_ccu"] * 1.25


def test_forecast_scale_invariance() -> None:
    """Verify that multiplying historical demand by k scales forecast demand by k."""
    import joblib
    from pathlib import Path
    from src.model.predict import simulate_promotional_campaign

    model_path = Path("models/indie_demand_model.joblib")
    data_path = Path("data/processed/367520_features.parquet")
    if not model_path.exists() or not data_path.exists():
        return

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    res1 = simulate_promotional_campaign(
        history_df=df,
        model=model,
        horizon_days=7,
        start_date="2026-10-21",
        discount_depth=0.50,
        sale_duration_days=5,
    )

    df2 = df.copy()
    scale_factor = 2.0
    for col in [
        "demand",
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_30",
        "sales_rolling_3",
        "sales_rolling_std_3",
        "sales_rolling_7",
        "sales_rolling_std_7",
        "sales_rolling_14",
        "sales_rolling_std_14",
        "sales_rolling_30",
        "sales_rolling_std_30",
    ]:
        if col in df2.columns:
            df2[col] = df2[col] * scale_factor

    res2 = simulate_promotional_campaign(
        history_df=df2,
        model=model,
        horizon_days=7,
        start_date="2026-10-21",
        discount_depth=0.50,
        sale_duration_days=5,
    )

    p1 = res1["forecast_df"]["predicted_ccu"].values
    p2 = res2["forecast_df"]["predicted_ccu"].values
    ratios = p2 / p1
    assert (abs(ratios - scale_factor) < 0.05).all()


def test_elasticity_monotonicity() -> None:
    """Verify that peak promotional CCU is non-decreasing with increasing discount depth."""
    import joblib
    from pathlib import Path
    from src.model.predict import simulate_promotional_campaign

    model_path = Path("models/indie_demand_model.joblib")
    data_path = Path("data/processed/367520_features.parquet")
    if not model_path.exists() or not data_path.exists():
        return

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    discounts = [0.10, 0.20, 0.33, 0.50, 0.66, 0.75]
    peaks = []
    for d in discounts:
        res = simulate_promotional_campaign(
            history_df=df,
            model=model,
            horizon_days=7,
            start_date="2026-10-21",
            discount_depth=d,
            sale_duration_days=5,
        )
        peaks.append(res["peak_promo_ccu"])

    assert all(peaks[i] <= peaks[i + 1] for i in range(len(peaks) - 1))


def test_forecast_no_early_organic_surge() -> None:
    """Verify that forward organic predictions maintain continuous decay without artificial transient surges."""
    import joblib
    from pathlib import Path
    from src.model.predict import simulate_promotional_campaign

    model_path = Path("models/indie_demand_model.joblib")
    data_path = Path("data/processed/367520_features.parquet")
    if not model_path.exists() or not data_path.exists():
        return

    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)

    # 1. Zero-discount simulation starting immediately
    res = simulate_promotional_campaign(
        history_df=df,
        model=model,
        horizon_days=7,
        start_date="2026-10-19",
        discount_depth=0.0,
        sale_duration_days=7,
    )

    fdf = res["forecast_df"]
    last_hist_ccu = float(df.sort_values("date")["demand"].iloc[-1])
    first_5_days = fdf["predicted_ccu"].head(5).values

    # No early day should surge past 1.05x the last historical demand
    assert (first_5_days <= last_hist_ccu * 1.05).all()
    # Predictions must not have active promo flags on zero discount
    assert not fdf["is_promo_active"].any()
