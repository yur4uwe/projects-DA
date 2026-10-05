from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
import joblib
import pandas as pd

from ..features.composites import compute_skill_composites


def predict_salary(
    candidate: dict[str, Any],
    model_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run salary prediction on a single candidate feature profile."""
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "salary_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train model first.")

    pipeline = joblib.load(model_path)

    # 1. Parse and derive candidate skill composites
    skills_list = candidate.get("skills", [])
    composites = compute_skill_composites(skills_list)
    comp_dict = asdict(composites)
    skill_flags = comp_dict.pop("flags")
    comp_dict.update(skill_flags)

    # 2. Assemble candidate feature row
    candidate_row = {
        "experience_years": float(candidate.get("experience_years", 3.0)),
        "seniority_level": int(candidate.get("seniority_level", 2)),
        "english_proficiency_rank": int(candidate.get("english_proficiency_rank", 3)),
        "is_remote": int(candidate.get("is_remote", 1)),
        "category": str(candidate.get("category", "Python")),
        "title_char_length": len(candidate.get("title", "Software Engineer")),
        "desc_word_count": len(candidate.get("description", "").split()),
        **comp_dict,
    }

    input_df = pd.DataFrame([candidate_row])

    # 3. Model inference
    pred_val = float(pipeline.predict(input_df)[0])
    # Sanity clip to reasonable bounds
    predicted_monthly_usd = max(500.0, round(pred_val, 2))

    return {
        "predicted_monthly_usd": predicted_monthly_usd,
        "estimated_annual_usd": round(predicted_monthly_usd * 12.0, 2),
        "range_p10_usd": round(predicted_monthly_usd * 0.85, 2),
        "range_p90_usd": round(predicted_monthly_usd * 1.15, 2),
        "extracted_skills": skills_list,
        "is_fullstack": bool(comp_dict.get("is_fullstack", 0)),
        "is_cloud_native": bool(comp_dict.get("is_cloud_native", 0)),
        "is_ai_ml": bool(comp_dict.get("is_ai_ml", 0)),
    }
