import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute standard regression metrics on original scale (USD)."""
    # Ensure positive predictions for MAPE
    y_pred_clipped = np.clip(y_pred, 1.0, None)
    y_true_clipped = np.clip(y_true, 1.0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true_clipped - y_pred_clipped) / y_true_clipped)) * 100.0
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MedAE": round(float(medae), 2),
        "MAPE%": round(float(mape), 2),
        "R2": round(float(r2), 4),
    }
