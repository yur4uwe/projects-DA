"""Unit tests for model pipeline, metrics, and inference."""

import numpy as np

from src.features.schema import DEAL_TIERS
from src.models.metrics import compute_multiclass_metrics
from src.models.pipeline import (
    create_calibrated_rf_classifier,
    create_logistic_baseline,
    create_rf_classifier,
)
from src.models.predict import predict_deal


def test_pipeline_instantiation() -> None:
    rf_pipe = create_rf_classifier()
    assert "prep" in rf_pipe.named_steps
    assert "clf" in rf_pipe.named_steps

    cal_pipe = create_calibrated_rf_classifier()
    assert "prep" in cal_pipe.named_steps
    assert "clf" in cal_pipe.named_steps

    log_pipe = create_logistic_baseline()
    assert "prep" in log_pipe.named_steps
    assert "clf" in log_pipe.named_steps


def test_compute_multiclass_metrics() -> None:
    y_true = ["Fair Price", "Great Deal", "Overpriced", "Fair Price"]
    y_pred = ["Fair Price", "Great Deal", "Overpriced", "Overpriced"]
    y_prob = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.3, 0.4],
        ]
    )
    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob, labels=DEAL_TIERS)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.75


def test_predict_deal_inference() -> None:
    sample_listing = {
        "Company": "Dell",
        "TypeName": "Notebook",
        "Inches": 15.6,
        "Ram_GB": 8,
        "Weight_kg": 2.1,
        "Cpu_Brand": "Intel",
        "Cpu_Tier": "Core i5",
        "Cpu_Freq_GHz": 2.5,
        "Primary_Storage_GB": 256,
        "Primary_Storage_Type": "SSD",
        "Total_Storage_GB": 256,
        "Gpu_Brand": "Intel",
        "Gpu_Tier": "Integrated",
        "OpSys": "Windows",
        "Screen_Width": 1920,
        "Screen_Height": 1080,
        "Is_Touchscreen": False,
        "Is_IPS": True,
        "Has_Dedicated_Gpu": False,
        "Has_SSD": True,
        "asking_price": 500.0,
    }

    result = predict_deal(sample_listing)
    assert "deal_tier" in result
    assert result["deal_tier"] in DEAL_TIERS
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3
    assert "asking_price" in result
    assert "market_baseline_price" in result
    assert "price_delta_euros" in result
    assert "price_ratio" in result
    assert "relative_savings_pct" in result
