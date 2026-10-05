import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from .metrics import compute_regression_metrics
from .pipeline import get_feature_columns
from .registry import (
    DEFAULT_DB_PATH,
    ExperimentRecord,
    get_experiment,
    init_db,
    list_experiments,
    log_experiment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


HISTORICAL_MILESTONES = [
    {
        "run_id": "milestone_1_cold_start",
        "timestamp": "2026-10-05T14:30:00+00:00",
        "name": "M1: Cold-Start Prototype",
        "description": "Initial scraper proof-of-concept on 2 pages of general Djinni listings. Only 15 jobs had public salaries. LightGBM severely underfit due to small sample size.",
        "model_name": "LIGHTGBM (L1)",
        "target_transform": "none",
        "sample_size": 15,
        "cv_folds": 3,
        "feature_count": 84,
        "features_used": ["experience_years", "seniority_level", "english_proficiency_rank"],
        "mae": 2877.0,
        "rmse": 3410.0,
        "r2": -0.1200,
        "medae": 2100.0,
        "mape": 68.5,
        "hyperparams": {"n_estimators": 50, "leaves": 15},
    },
    {
        "run_id": "milestone_2_multipages",
        "timestamp": "2026-10-05T15:45:00+00:00",
        "name": "M2: Multi-Page Ingestion",
        "description": "Expanded crawl across Djinni and RemoteOK without salary filters. Yielded 49 labeled salary rows. Positive predictive correlation emerged, but skill addition was noisy.",
        "model_name": "LIGHTGBM (L1)",
        "target_transform": "none",
        "sample_size": 49,
        "cv_folds": 5,
        "feature_count": 84,
        "features_used": ["experience_years", "seniority_level", "num_skills", "category"],
        "mae": 2657.0,
        "rmse": 3120.0,
        "r2": 0.3200,
        "medae": 1850.0,
        "mape": 59.2,
        "hyperparams": {"n_estimators": 80, "leaves": 15},
    },
    {
        "run_id": "milestone_3_targeted_crawl",
        "timestamp": "2026-10-05T19:42:00+00:00",
        "name": "M3: Targeted Salary Crawl",
        "description": "Applied Djinni editorial=has_public_salary query parameter across 25 pages, expanding labeled training sample by 8x from 49 to 388 rows. Error dropped by over 55%.",
        "model_name": "LIGHTGBM (L1)",
        "target_transform": "none",
        "sample_size": 388,
        "cv_folds": 5,
        "feature_count": 84,
        "features_used": ["experience_years", "seniority_level", "num_skills", "num_cloud", "category"],
        "mae": 1137.62,
        "rmse": 2257.55,
        "r2": 0.3531,
        "medae": 598.02,
        "mape": 52.25,
        "hyperparams": {"n_estimators": 80, "leaves": 15, "objective": "regression_l1"},
    },
    {
        "run_id": "milestone_4_log_target",
        "timestamp": "2026-10-05T20:14:00+00:00",
        "name": "M4: Log-Salary Transformation",
        "description": "Modeled log-normal tech wage distribution using TransformedTargetRegressor(func=log1p). Prevented extreme high-earner contracts from skewing junior/mid predictions.",
        "model_name": "LIGHTGBM (LOG1P)",
        "target_transform": "log1p",
        "sample_size": 388,
        "cv_folds": 5,
        "feature_count": 84,
        "features_used": ["experience_years", "seniority_level", "num_skills", "num_cloud", "category"],
        "mae": 1131.38,
        "rmse": 2263.70,
        "r2": 0.3495,
        "medae": 584.98,
        "mape": 49.64,
        "hyperparams": {"n_estimators": 80, "target_transform": "log1p"},
    },
    {
        "run_id": "milestone_5_ensemble",
        "timestamp": "2026-10-05T20:14:30+00:00",
        "name": "M5: Production Ensemble Blend",
        "description": "Blended Ridge (linear experience gradient, R2=0.41) and LightGBM (non-linear category and skill splits) on log1p salary. Achieved best overall MAE, RMSE, and R2 across all milestones.",
        "model_name": "ENSEMBLE (RIDGE+LGBM)",
        "target_transform": "log1p",
        "sample_size": 388,
        "cv_folds": 5,
        "feature_count": 84,
        "features_used": ["experience_years", "seniority_level", "num_skills", "num_cloud", "category"],
        "mae": 1099.35,
        "rmse": 2150.94,
        "r2": 0.4127,
        "medae": 559.32,
        "mape": 49.38,
        "hyperparams": {"weights": [0.4, 0.6], "ridge_alpha": 10.0, "lgbm_leaves": 15},
    },
]


def ensure_history_seeded() -> None:
    """Ensure baseline historical milestones exist in the database."""
    init_db()
    existing = list_experiments()
    existing_ids = {r.run_id for r in existing}
    for m in HISTORICAL_MILESTONES:
        if m["run_id"] not in existing_ids:
            rec = ExperimentRecord(
                run_id=m["run_id"],
                timestamp=m["timestamp"],
                name=m["name"],
                description=m["description"],
                model_name=m["model_name"],
                target_transform=m["target_transform"],
                sample_size=m["sample_size"],
                cv_folds=m["cv_folds"],
                feature_count=m["feature_count"],
                features_used=m["features_used"],
                mae=m["mae"],
                rmse=m["rmse"],
                r2=m["r2"],
                medae=m["medae"],
                mape=m["mape"],
                hyperparams=m["hyperparams"],
            )
            log_experiment(rec)


def display_model_lineage() -> None:
    """Print the complete chronological evolution narrative and metric lineage table."""
    ensure_history_seeded()
    runs = list_experiments()

    print("\n" + "=" * 88)
    print("           MODEL LINEAGE & EVOLUTION REPORT (TECH LABOR MARKET)")
    print("=" * 88)
    print("Tracking the empirical progression of salary modeling from cold-start to production.\n")

    # 1. Summary Lineage Table
    headers = [
        "Step",
        "Milestone Name",
        "Architecture",
        "N",
        "MAE ($)",
        "Delta MAE",
        "RMSE ($)",
        "R2 Score",
        "MAPE%",
    ]
    rows = []
    prev_mae = None

    for idx, r in enumerate(runs, 1):
        if prev_mae is None:
            delta_mae_str = "Baseline"
        else:
            diff = r.mae - prev_mae
            sign = "+" if diff >= 0 else ""
            delta_mae_str = f"{sign}${diff:,.1f}"
        prev_mae = r.mae

        rows.append(
            [
                f"M{idx}",
                r.name[:25],
                r.model_name[:20],
                str(r.sample_size),
                f"${r.mae:,.1f}",
                delta_mae_str,
                f"${r.rmse:,.1f}",
                f"{r.r2:.3f}",
                f"{r.mape:.1f}%",
            ]
        )

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt_row(vals: list[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)

    # 2. Cumulative Progress Highlights
    first_run = runs[0]
    last_run = runs[-1]
    mae_drop = first_run.mae - last_run.mae
    mae_drop_pct = (mae_drop / first_run.mae) * 100.0
    r2_gain = last_run.r2 - first_run.r2

    print(
        f"\nOVERALL PROGRESS: MAE reduced from ${first_run.mae:,.2f} -> ${last_run.mae:,.2f} "
        f"(-${mae_drop:,.2f} / -{mae_drop_pct:.1f}%) | R2 increased from {first_run.r2:.2f} -> {last_run.r2:.2f} ({r2_gain:+.2f})"
    )

    # 3. Detailed Chronological Milestone Diary
    print("\n" + "-" * 88)
    print("CHRONOLOGICAL MILESTONE LOG")
    print("-" * 88)

    for idx, r in enumerate(runs, 1):
        print(f"\n[{idx}] {r.name} ({r.timestamp[:10]})")
        print(f"    Architecture: {r.model_name} | Target: {r.target_transform} | Sample Size: N={r.sample_size} | Features: {r.feature_count}")
        print(f"    CV Performance: MAE=${r.mae:,.2f} | RMSE=${r.rmse:,.2f} | R2={r.r2:.4f} | MedAE=${r.medae:,.2f} | MAPE={r.mape:.1f}%")
        print(f"    Summary: {r.description}")

    print("\n" + "=" * 88 + "\n")


def record_current_model(description: str, name: Optional[str] = None) -> None:
    """Benchmark current training pipeline in codebase and append as the latest milestone."""
    import joblib

    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "salary_labeled_features.parquet"
    model_path = project_root / "models" / "salary_model.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run feature builder first.")

    df = pd.read_parquet(data_path)
    df = df[df["salary_mid_usd"].notnull()].copy()
    y_true = df["salary_mid_usd"].values

    num_cols, cat_cols, skill_cols = get_feature_columns(df)
    features_used = num_cols + cat_cols + skill_cols
    X = df[features_used]

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")

    pipeline = joblib.load(model_path)

    n_samples = len(df)
    actual_folds = min(5, n_samples) if n_samples >= 4 else 2
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

    logger.info(f"Running {actual_folds}-fold cross validation to record milestone snapshot...")
    preds = cross_val_predict(pipeline, X, y_true, cv=kf)
    preds = np.clip(preds, 50.0, None)
    metrics = compute_regression_metrics(y_true, preds)

    model_step = pipeline.named_steps["model"]
    model_name = type(model_step).__name__.upper()

    existing = list_experiments()
    step_num = len(existing) + 1
    run_name = name or f"M{step_num}: Snapshot Iteration"
    run_id = f"milestone_{step_num}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    record = ExperimentRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        name=run_name,
        description=description,
        model_name=model_name,
        target_transform="none",
        sample_size=n_samples,
        cv_folds=actual_folds,
        feature_count=len(features_used),
        features_used=features_used,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        r2=metrics["R2"],
        medae=metrics["MedAE"],
        mape=metrics["MAPE%"],
        hyperparams={"pipeline_steps": list(pipeline.named_steps.keys())},
    )

    log_experiment(record)
    logger.info(f"Successfully recorded milestone '{run_name}' to lineage database.")
    display_model_lineage()


def diff_two_milestones(id1: str, id2: str) -> None:
    """Print side-by-side metric comparison between two specific milestones."""
    r1 = get_experiment(id1)
    r2 = get_experiment(id2)
    if not r1:
        print(f"Error: Milestone '{id1}' not found in database.")
        return
    if not r2:
        print(f"Error: Milestone '{id2}' not found in database.")
        return

    print("\n" + "=" * 78)
    print(f"MILESTONE DIFF: [{r1.name}] vs [{r2.name}]")
    print("=" * 78)
    print(f"Baseline:  {r1.name} ({r1.model_name}, N={r1.sample_size})")
    print(f"Candidate: {r2.name} ({r2.model_name}, N={r2.sample_size})")
    print("-" * 78)

    metrics_meta = [
        ("MAE ($)", r1.mae, r2.mae, False),
        ("RMSE ($)", r1.rmse, r2.rmse, False),
        ("MedAE ($)", r1.medae, r2.medae, False),
        ("MAPE (%)", r1.mape, r2.mape, False),
        ("R2 Score", r1.r2, r2.r2, True),
    ]

    print(f"{'Metric':<12} | {'Baseline':<12} | {'Candidate':<12} | {'Delta':<12} | {'Delta %':<10} | {'Status'}")
    print("-" * 78)
    for m_label, v1, v2, higher_better in metrics_meta:
        diff = v2 - v1
        pct = (diff / abs(v1) * 100.0) if abs(v1) > 1e-6 else 0.0
        sign = "+" if diff >= 0 else ""
        diff_s = f"{sign}{diff:,.2f}"
        pct_s = f"{sign}{pct:.1f}%"
        status = "IMPROVED" if ((higher_better and diff > 0) or (not higher_better and diff < 0)) else "DEGRADED"
        if abs(diff) < 1e-4:
            status = "NEUTRAL"
        print(f"{m_label:<12} | {v1:<12,.2f} | {v2:<12,.2f} | {diff_s:<12} | {pct_s:<10} | {status}")

    print("-" * 78)
    s1, s2 = set(r1.features_used), set(r2.features_used)
    added = sorted(s2 - s1)
    removed = sorted(s1 - s2)
    if added:
        print(f"  + Added Features ({len(added)}): {', '.join(added[:10])}{'...' if len(added) > 10 else ''}")
    if removed:
        print(f"  - Removed Features ({len(removed)}): {', '.join(removed[:10])}{'...' if len(removed) > 10 else ''}")
    if not added and not removed:
        print("  Features: Identical feature sets")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tech Labor Market Model Lineage & Milestone Evolution Tracker",
        epilog="Run without arguments to display the complete historical progression.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="DESCRIPTION",
        help="Record a new milestone snapshot from the currently saved model in codebase",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Custom name for the new milestone (optional with --record)",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("MILESTONE_A", "MILESTONE_B"),
        help="Compare two milestones side-by-side by run_id or name",
    )

    args = parser.parse_args()

    if args.record:
        record_current_model(description=args.record, name=args.name)
    elif args.diff:
        diff_two_milestones(args.diff[0], args.diff[1])
    else:
        display_model_lineage()


if __name__ == "__main__":
    main()
