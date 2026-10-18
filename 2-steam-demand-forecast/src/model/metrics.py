"""Evaluation metrics for time-series demand forecasting."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]
) -> dict[str, float]:
    """Calculate comprehensive time-series forecasting metrics."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    # Ensure non-negative demand predictions
    y_p = np.clip(y_p, 0.0, None)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = float(r2_score(y_t, y_p))

    # WAPE (Weighted Absolute Percentage Error: sum(|y - y_hat|) / sum(y))
    total_actual = float(np.sum(y_t))
    wape = float(np.sum(np.abs(y_t - y_p)) / total_actual) if total_actual > 0 else 0.0

    # Non-zero safe MAPE
    mask = y_t > 0
    mape = (
        float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])))
        if np.any(mask)
        else 0.0
    )

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape": round(wape, 4),
        "wape_percent": round(wape * 100.0, 2),
        "mape_percent": round(mape * 100.0, 2),
        "r2": round(r2, 4),
    }
