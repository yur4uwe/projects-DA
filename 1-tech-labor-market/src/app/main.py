import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path when running via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.features.skills import SKILL_GROUPS
from src.models.predict import predict_salary

MODEL_PATH = PROJECT_ROOT / "models" / "salary_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
IMPORTANCE_PATH = PROJECT_ROOT / "models" / "global_feature_importance.json"

st.set_page_config(
    page_title="Tech Labor Market & Compensation Intelligence",
    layout="wide",
)

st.title("Tech Labor Market & Compensation Intelligence")
st.markdown(
    "Data-driven salary benchmark estimation and TreeSHAP explainability based on real scraped listings."
)


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_global_importance() -> list[dict[str, Any]]:
    if IMPORTANCE_PATH.exists():
        with IMPORTANCE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


# -------------------------------------------------------------
# Sidebar: Candidate Profile Configuration
# -------------------------------------------------------------
st.sidebar.header("Candidate Profile")

category_options = [
    "Python",
    "Java",
    "JavaScript",
    "Data Science",
    "DevOps",
    "QA",
    "Ruby",
    "Go",
    "Other",
]
selected_category = st.sidebar.selectbox("Primary Domain", category_options, index=0)

exp_years = st.sidebar.slider(
    "Years of Experience", min_value=0.0, max_value=15.0, value=4.0, step=0.5
)

seniority_map = {
    "Trainee / Intern": 0,
    "Junior": 1,
    "Middle": 2,
    "Senior": 3,
    "Lead / Staff / Architect": 4,
}
selected_seniority = st.sidebar.selectbox(
    "Seniority Level", list(seniority_map.keys()), index=2
)

english_map = {
    "None / Unspecified": 0,
    "Pre-Intermediate (A2)": 1,
    "Intermediate (B1)": 2,
    "Upper-Intermediate (B2)": 3,
    "Advanced / Fluent (C1/C2)": 4,
}
selected_english = st.sidebar.selectbox(
    "English Level", list(english_map.keys()), index=3
)

is_remote = st.sidebar.radio("Work Preference", ["Remote", "On-site / Hybrid"], index=0)

st.sidebar.subheader("Technical Skills")

# Multi-selects by skill group
languages = st.sidebar.multiselect(
    "Languages",
    sorted(SKILL_GROUPS["languages"].keys()),
    default=["python", "sql"] if "python" in SKILL_GROUPS["languages"] else [],
)

cloud_devops = st.sidebar.multiselect(
    "Cloud & DevOps",
    sorted(list(SKILL_GROUPS["cloud"].keys()) + list(SKILL_GROUPS["devops"].keys())),
    default=["docker", "aws"],
)

databases = st.sidebar.multiselect(
    "Databases & Streaming",
    sorted(SKILL_GROUPS["databases"].keys()),
    default=["postgresql"],
)

frameworks = st.sidebar.multiselect(
    "Frameworks & Web",
    sorted(
        list(SKILL_GROUPS["frontend"].keys()) + list(SKILL_GROUPS["backend"].keys())
    ),
    default=["fastapi"] if "fastapi" in SKILL_GROUPS["backend"] else [],
)

ai_data = st.sidebar.multiselect(
    "AI & Data Tools",
    sorted(list(SKILL_GROUPS["data_ml"].keys()) + list(SKILL_GROUPS["genai"].keys())),
    default=[],
)

all_selected_skills = sorted(
    set(languages + cloud_devops + databases + frameworks + ai_data)
)

# -------------------------------------------------------------
# Inference Execution
# -------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Model artifact not found. Please train the model via `uv run python -m src.models.train`."
    )
    st.stop()

candidate_input = {
    "experience_years": exp_years,
    "seniority_level": seniority_map[selected_seniority],
    "english_proficiency_rank": english_map[selected_english],
    "category": selected_category,
    "is_remote": 1 if is_remote == "Remote" else 0,
    "skills": all_selected_skills,
    "title": f"{selected_seniority} {selected_category} Engineer",
    "description": "Standard market responsibilities and technical deliverables.",
}

try:
    prediction = predict_salary(
        candidate_input, model_path=MODEL_PATH, include_explanation=True
    )
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

# -------------------------------------------------------------
# Main Dashboard Layout
# -------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Predicted Monthly Salary (USD)",
        value=f"${prediction['predicted_monthly_usd']:,.2f}",
        delta=f"${prediction['predicted_monthly_usd'] - prediction.get('explanation', {}).get('base_market_salary_usd', prediction['predicted_monthly_usd']):,.2f} vs Market Base",
    )

with col2:
    st.metric(
        label="Estimated Annual Salary (USD)",
        value=f"${prediction['estimated_annual_usd']:,.2f}",
    )

with col3:
    st.metric(
        label="80% Market Confidence Band",
        value=f"${prediction['range_p10_usd']:,.0f} - ${prediction['range_p90_usd']:,.0f}",
    )

st.markdown("---")

# Profile attributes badges
badge_cols = st.columns(4)
with badge_cols[0]:
    st.info(f"**Total Skills:** {len(all_selected_skills)}")
with badge_cols[1]:
    st.info(f"**Cloud Native:** {'Yes' if prediction['is_cloud_native'] else 'No'}")
with badge_cols[2]:
    st.info(f"**Full Stack:** {'Yes' if prediction['is_fullstack'] else 'No'}")
with badge_cols[3]:
    st.info(f"**AI / ML:** {'Yes' if prediction['is_ai_ml'] else 'No'}")

# Tabs: Explainability, Skill Upskilling ROI, Market Intelligence
tab_explain, tab_roi, tab_market = st.tabs(
    [
        "Local SHAP Attribution",
        "Skill ROI & Upskilling Simulator",
        "Global Market Trends",
    ]
)

# -------------------------------------------------------------
# Tab 1: Local SHAP Attribution
# -------------------------------------------------------------
with tab_explain:
    st.subheader("Why Did the Model Predict This Salary?")
    st.markdown(
        "TreeSHAP breaks down the prediction additively into the base market starting point and the exact dollar amount added or subtracted by each feature."
    )

    exp_data = prediction.get("explanation", {})
    if "error" in exp_data:
        st.warning(exp_data["error"])
    else:
        base_val = exp_data.get("base_market_salary_usd", 0.0)
        pos_drivers = exp_data.get("top_positive_drivers", [])
        neg_drivers = exp_data.get("top_negative_drivers", [])

        # Plotly horizontal bar chart of drivers
        driver_items = []
        for d in pos_drivers:
            driver_items.append(
                {
                    "Feature": d["feature"],
                    "Impact ($/mo)": d["contribution_usd"],
                    "Type": "Positive",
                }
            )
        for d in neg_drivers:
            driver_items.append(
                {
                    "Feature": d["feature"],
                    "Impact ($/mo)": d["contribution_usd"],
                    "Type": "Negative / Penalty",
                }
            )

        if driver_items:
            driver_df = pd.DataFrame(driver_items).sort_values(
                "Impact ($/mo)", ascending=True
            )
            fig = px.bar(
                driver_df,
                x="Impact ($/mo)",
                y="Feature",
                color="Type",
                orientation="h",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Negative / Penalty": "#e74c3c",
                },
                title=f"Feature Dollar Attribution (Base Expected Salary: ${base_val:,.2f})",
            )
            fig.update_layout(height=400, margin={"l": 20, "r": 20, "t": 40, "b": 20})
            st.plotly_chart(fig, use_container_width=True)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**Top Positive Value Drivers**")
            for d in pos_drivers:
                st.write(f"+ **{d['feature']}**: +${d['contribution_usd']:,.2f}/mo")
        with col_neg:
            st.markdown("**Top Negative / Below-Average Drivers**")
            for d in neg_drivers:
                st.write(
                    f"- **{d['feature']}**: -${abs(d['contribution_usd']):,.2f}/mo"
                )

# -------------------------------------------------------------
# Tab 2: Skill ROI & Upskilling Simulator
# -------------------------------------------------------------
with tab_roi:
    st.subheader("Simulate Skill Addition & Estimated Salary Uplift")
    st.markdown(
        "This tool evaluates skills currently missing from your profile and estimates the incremental monthly salary gain if you acquire them."
    )

    unselected_skills = [
        s
        for s in [
            "kubernetes",
            "aws",
            "terraform",
            "go",
            "kafka",
            "postgresql",
            "fastapi",
            "react",
            "spark",
            "llm",
            "rag",
        ]
        if s not in all_selected_skills
    ]

    simulated_results = []
    base_pred = prediction["predicted_monthly_usd"]

    for extra_skill in unselected_skills:
        sim_candidate = dict(candidate_input)
        sim_candidate["skills"] = all_selected_skills + [extra_skill]
        sim_res = predict_salary(
            sim_candidate, model_path=MODEL_PATH, include_explanation=False
        )
        delta = sim_res["predicted_monthly_usd"] - base_pred
        simulated_results.append(
            {
                "Skill": extra_skill.upper(),
                "New Monthly Salary ($)": sim_res["predicted_monthly_usd"],
                "Estimated Uplift ($/mo)": delta,
                "Annual Uplift ($/yr)": delta * 12.0,
            }
        )

    if simulated_results:
        sim_df = pd.DataFrame(simulated_results).sort_values(
            "Estimated Uplift ($/mo)", ascending=False
        )
        st.dataframe(
            sim_df.style.format(
                {
                    "New Monthly Salary ($)": "${:,.2f}",
                    "Estimated Uplift ($/mo)": "+${:,.2f}",
                    "Annual Uplift ($/yr)": "+${:,.2f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("You have already selected all evaluated high-impact skills.")

# -------------------------------------------------------------
# Tab 3: Global Market Trends
# -------------------------------------------------------------
with tab_market:
    st.subheader("Global Feature Importance (Across Entire Market)")
    st.markdown(
        "Mean absolute SHAP dollar values across the analyzed tech job market dataset."
    )

    importance_list = load_global_importance()
    if importance_list:
        imp_df = (
            pd.DataFrame(importance_list)
            .head(15)
            .sort_values("mean_abs_shap_usd", ascending=True)
        )
        fig_imp = px.bar(
            imp_df,
            x="mean_abs_shap_usd",
            y="feature",
            orientation="h",
            labels={
                "mean_abs_shap_usd": "Mean Absolute Impact (USD/mo)",
                "feature": "Market Driver",
            },
            title="Top 15 Market Salary Determinants",
            color="mean_abs_shap_usd",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(height=500, margin={"l": 20, "r": 20, "t": 40, "b": 20})
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Global importance metadata not generated yet.")

    meta = load_metadata()
    if meta:
        st.markdown("**Underlying Model Configuration:**")
        st.json(
            {
                "Selected Model": meta.get("best_model"),
                "Training Samples (with Salary)": meta.get("n_samples"),
                "Cross-Validation Folds": meta.get("n_splits"),
                "CV Metrics": meta.get("cv_metrics", {}).get(meta.get("best_model")),
                "Trained At": meta.get("trained_at"),
            }
        )
