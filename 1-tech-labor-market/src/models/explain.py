import json
import logging
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_feature_name(name: str) -> str:
    """Format raw transformer column names into human-readable labels."""
    if name.startswith("num__"):
        name = name[5:]
    elif name.startswith("cat__"):
        name = name[5:]
    elif name.startswith("skills__"):
        name = name[8:]

    name = name.replace("skill_", "Skill: ")
    name = name.replace("is_", "Flag: ")
    name = name.replace("_", " ").title()
    return name


class SalaryExplainer:
    """TreeSHAP explainability engine for the salary prediction pipeline."""

    def __init__(self, pipeline_path: Optional[Path] = None):
        if pipeline_path is None:
            pipeline_path = (
                Path(__file__).resolve().parent.parent.parent
                / "models"
                / "salary_model.joblib"
            )

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Model not found at {pipeline_path}")

        self.pipeline = joblib.load(pipeline_path)
        self.preprocessor = self.pipeline.named_steps["prep"]
        self.model = self.pipeline.named_steps["model"]

        # Only Tree models (e.g. LightGBM) support TreeExplainer
        if hasattr(self.model, "booster_"):
            self.explainer = shap.TreeExplainer(self.model)
            self.expected_value = float(np.ravel(self.explainer.expected_value)[0])
        else:
            self.explainer = None
            self.expected_value = 0.0

    def explain_candidate(
        self, candidate_df: pd.DataFrame, top_n: int = 5
    ) -> dict[str, Any]:
        """Compute local SHAP attribution for a single candidate profile."""
        if self.explainer is None:
            return {"error": "Active model is not a tree booster. SHAP unavailable."}

        # Transform candidate input features
        X_trans = self.preprocessor.transform(candidate_df)
        feature_names = self.preprocessor.get_feature_names_out()

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_trans)[0]

        # Pair features with their dollar contributions
        contributions = []
        for raw_name, val in zip(feature_names, shap_values):
            if abs(val) > 0.01:
                contributions.append(
                    {
                        "feature_raw": raw_name,
                        "feature": clean_feature_name(raw_name),
                        "contribution_usd": round(float(val), 2),
                    }
                )

        # Sort by impact
        positive_drivers = sorted(
            [c for c in contributions if c["contribution_usd"] > 0],
            key=lambda x: x["contribution_usd"],
            reverse=True,
        )[:top_n]

        negative_drivers = sorted(
            [c for c in contributions if c["contribution_usd"] < 0],
            key=lambda x: x["contribution_usd"],
        )[:top_n]

        predicted_val = float(self.pipeline.predict(candidate_df)[0])

        return {
            "base_market_salary_usd": round(self.expected_value, 2),
            "predicted_salary_usd": round(predicted_val, 2),
            "top_positive_drivers": positive_drivers,
            "top_negative_drivers": negative_drivers,
        }

    def compute_global_importance(
        self, X_df: pd.DataFrame, output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Compute mean absolute SHAP feature importance across the dataset."""
        if self.explainer is None:
            return pd.DataFrame()

        X_trans = self.preprocessor.transform(X_df)
        feature_names = self.preprocessor.get_feature_names_out()

        shap_values = self.explainer.shap_values(X_trans)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": [clean_feature_name(f) for f in feature_names],
                "feature_raw": feature_names,
                "mean_abs_shap_usd": np.round(mean_abs_shap, 2),
            }
        ).sort_values("mean_abs_shap_usd", ascending=False)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            importance_records = importance_df.head(25).to_dict(orient="records")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(importance_records, f, indent=2)
            logger.info(f"Saved global feature importance to {output_path}")

        return importance_df
