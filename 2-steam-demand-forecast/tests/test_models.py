"""Unit tests for metrics, baseline, and model registry."""

import numpy as np
import pandas as pd

from src.models.baseline import SeasonalPersistenceBaseline
from src.models.metrics import calculate_metrics
from src.models.registry import ModelRegistry


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
