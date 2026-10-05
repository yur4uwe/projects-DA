from typing import Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CORE_NUMERICAL_COLS = [
    "experience_years",
    "seniority_level",
    "english_proficiency_rank",
    "is_remote",
    "num_skills",
    "num_languages",
    "num_frontend",
    "num_backend",
    "num_databases",
    "num_cloud",
    "num_devops",
    "num_data_ml",
    "num_genai",
    "num_mobile",
    "is_fullstack",
    "is_cloud_native",
    "is_ai_ml",
    "is_devops_oriented",
    "title_char_length",
    "desc_word_count",
]

CATEGORICAL_COLS = ["category"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Identify present numerical, categorical, and binary skill columns."""
    num_cols = [c for c in CORE_NUMERICAL_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    return num_cols, cat_cols, skill_cols


def create_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    skill_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a sklearn ColumnTransformer for tabular preprocessing."""
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    transformers = [
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
        ("skills", "passthrough", skill_cols),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")
