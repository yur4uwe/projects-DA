import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .metrics import compute_regression_metrics
from .pipeline import create_preprocessor, get_feature_columns
from .registry import (
    DEFAULT_DB_PATH,
    ExperimentRecord,
    get_experiment,
    get_latest_runs,
    list_experiments,
    log_experiment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_experiment_model(
    model_type: str,
    target_transform: str,
    num_cols: list[str],
    cat_cols: list[str],
    skill_cols: list[str],
    alpha: float = 10.0,
    n_estimators: int = 80,
    learning_rate: float = 0.05,
    num_leaves: int = 15,
) -> tuple[Any, dict[str, Any]]:
    """Construct pipeline and hyperparameter record."""
    scale_numeric = model_type in ("ridge", "ensemble")
    prep = create_preprocessor(num_cols, cat_cols, skill_cols, scale_numeric=scale_numeric)

    hyperparams: dict[str, Any] = {
        "model_type": model_type,
        "target_transform": target_transform,
    }

    if model_type == "dummy":
        base_model = DummyRegressor(strategy="median")
        hyperparams["strategy"] = "median"
    elif model_type == "ridge":
        base_model = Ridge(alpha=alpha, random_state=42)
        hyperparams["alpha"] = alpha
    elif model_type == "lgbm":
        base_model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
        hyperparams.update(
            {
                "objective": "regression_l1",
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
            }
        )
    elif model_type == "ensemble":
        ridge = Ridge(alpha=alpha, random_state=42)
        lgbm = LGBMRegressor(
            objective="regression_l1",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
        base_model = VotingRegressor(
            estimators=[("ridge", ridge), ("lgbm", lgbm)],
            weights=[0.4, 0.6],
        )
        hyperparams.update(
            {
                "weights": [0.4, 0.6],
                "ridge_alpha": alpha,
                "lgbm_leaves": num_leaves,
            }
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if target_transform == "log1p":
        # Log target transformation for log-normal wage distribution
        model_step = TransformedTargetRegressor(
            regressor=base_model,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        model_step = base_model

    pipe = Pipeline(steps=[("prep", prep), ("model", model_step)])
    return pipe, hyperparams


def run_benchmark_experiment(
    name: str,
    description: str,
    model_type: str = "lgbm",
    target_transform: str = "none",
    data_path: Optional[Path] = None,
    cv_folds: int = 5,
    db_path: Optional[Path] = None,
    **kwargs: Any,
) -> ExperimentRecord:
    """Execute cross-validation for a specific configuration and register to experiment DB."""
    project_root = Path(__file__).resolve().parent.parent.parent
    if data_path is None:
        data_path = project_root / "data" / "processed" / "salary_labeled_features.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Run builder first.")

    df = pd.read_parquet(data_path)
    df = df[df["salary_mid_usd"].notnull()].copy()
    y_true = df["salary_mid_usd"].values

    num_cols, cat_cols, skill_cols = get_feature_columns(df)
    features_used = num_cols + cat_cols + skill_cols

    X = df[features_used]

    pipeline, hyperparams = build_experiment_model(
        model_type=model_type,
        target_transform=target_transform,
        num_cols=num_cols,
        cat_cols=cat_cols,
        skill_cols=skill_cols,
        **kwargs,
    )

    n_samples = len(df)
    actual_folds = min(cv_folds, n_samples) if n_samples >= 4 else 2
    kf = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

    logger.info(
        f"Evaluating experiment '{name}' | Model: {model_type} | "
        f"Target Transform: {target_transform} | Features: {len(features_used)} | N: {n_samples}"
    )

    preds = cross_val_predict(pipeline, X, y_true, cv=kf)
    # Clip negative predictions if any
    preds = np.clip(preds, 50.0, None)
    metrics = compute_regression_metrics(y_true, preds)

    timestamp_str = datetime.now(timezone.utc).isoformat()
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"

    record = ExperimentRecord(
        run_id=run_id,
        timestamp=timestamp_str,
        name=name,
        description=description,
        model_name=model_type.upper(),
        target_transform=target_transform,
        sample_size=n_samples,
        cv_folds=actual_folds,
        feature_count=len(features_used),
        features_used=features_used,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        r2=metrics["R2"],
        medae=metrics["MedAE"],
        mape=metrics["MAPE%"],
        hyperparams=hyperparams,
    )

    log_experiment(record, db_path=db_path)
    logger.info(
        f"Registered Run: {record.run_id} | MAE: ${record.mae:,.2f} | "
        f"RMSE: ${record.rmse:,.2f} | R2: {record.r2:.4f} | MedAE: ${record.medae:,.2f} | MAPE: {record.mape:.1f}%"
    )
    return record


def print_runs_table(runs: list[ExperimentRecord]) -> None:
    """Print an aligned ASCII summary table of experimental runs."""
    if not runs:
        print("No experiments logged yet.")
        return

    headers = [
        "Run ID",
        "Timestamp",
        "Name",
        "Model",
        "Target",
        "Feats",
        "MAE ($)",
        "RMSE ($)",
        "R2",
        "MedAE ($)",
        "MAPE%",
    ]
    rows = []
    for r in runs:
        ts = r.timestamp[:19].replace("T", " ")
        rows.append(
            [
                r.run_id,
                ts,
                r.name[:18],
                r.model_name,
                r.target_transform,
                str(r.feature_count),
                f"${r.mae:,.1f}",
                f"${r.rmse:,.1f}",
                f"{r.r2:.3f}",
                f"${r.medae:,.1f}",
                f"{r.mape:.1f}%",
            ]
        )

    # Determine column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(val))

    def fmt_row(vals: list[str]) -> str:
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)

    print("\n" + "=" * len(sep))
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("=" * len(sep) + "\n")


def compare_two_runs(run1: ExperimentRecord, run2: ExperimentRecord) -> None:
    """Print side-by-side metric comparison and feature differences between two runs."""
    print("\n" + "=" * 78)
    print(f"EXPERIMENT COMPARISON: [{run1.run_id}] vs [{run2.run_id}]")
    print("=" * 78)
    print(f"Baseline (Run 1): {run1.name} ({run1.model_name}, target={run1.target_transform})")
    print(f"Candidate (Run 2): {run2.name} ({run2.model_name}, target={run2.target_transform})")
    print(f"Candidate Description: {run2.description or 'None'}")
    print("-" * 78)

    # 1. Metric Diffs
    def evaluate_metric_change(m_name: str, v1: float, v2: float, higher_is_better: bool = False) -> tuple[str, str, str]:
        diff = v2 - v1
        pct = (diff / abs(v1) * 100.0) if abs(v1) > 1e-6 else 0.0
        sign = "+" if diff >= 0 else ""
        diff_str = f"{sign}{diff:,.2f}"
        pct_str = f"{sign}{pct:.1f}%"

        if abs(diff) < 1e-4:
            status = "NO CHANGE"
        elif (higher_is_better and diff > 0) or (not higher_is_better and diff < 0):
            status = "IMPROVED"
        else:
            status = "DEGRADED"
        return diff_str, pct_str, status

    metrics_meta = [
        ("MAE ($)", run1.mae, run2.mae, False),
        ("RMSE ($)", run1.rmse, run2.rmse, False),
        ("MedAE ($)", run1.medae, run2.medae, False),
        ("MAPE (%)", run1.mape, run2.mape, False),
        ("R2 Score", run1.r2, run2.r2, True),
    ]

    print(f"{'Metric':<12} | {'Run 1':<12} | {'Run 2':<12} | {'Delta':<12} | {'Delta %':<10} | {'Status'}")
    print("-" * 78)
    for m_label, v1, v2, higher_better in metrics_meta:
        diff_s, pct_s, status = evaluate_metric_change(m_label, v1, v2, higher_better)
        v1_s = f"{v1:,.2f}"
        v2_s = f"{v2:,.2f}"
        print(f"{m_label:<12} | {v1_s:<12} | {v2_s:<12} | {diff_s:<12} | {pct_s:<10} | {status}")
    print("-" * 78)

    # 2. Feature & Architecture Diffs
    s1 = set(run1.features_used)
    s2 = set(run2.features_used)
    added = sorted(s2 - s1)
    removed = sorted(s1 - s2)

    print(f"Feature Count Change: {run1.feature_count} -> {run2.feature_count} ({len(s2) - len(s1):+d})")
    if added:
        print(f"  + Added Features ({len(added)}): {', '.join(added[:15])}{'...' if len(added) > 15 else ''}")
    if removed:
        print(f"  - Removed Features ({len(removed)}): {', '.join(removed[:15])}{'...' if len(removed) > 15 else ''}")
    if not added and not removed:
        print("  Features: Identical feature sets")

    if run1.target_transform != run2.target_transform:
        print(f"Target Transform Changed: {run1.target_transform} -> {run2.target_transform}")

    if run1.model_name != run2.model_name:
        print(f"Model Architecture Changed: {run1.model_name} -> {run2.model_name}")

    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech Labor Market Experiment Tracking & Model Benchmarking")
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Subcommand: list
    subparsers.add_parser("list", help="List all logged experiments")

    # Subcommand: compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two experiments")
    cmp_parser.add_argument("--run1", type=str, default=None, help="Baseline run ID or name (default: second-latest)")
    cmp_parser.add_argument("--run2", type=str, default=None, help="Candidate run ID or name (default: latest)")

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run a new benchmark experiment and register it")
    run_parser.add_argument("--name", type=str, required=True, help="Experiment name")
    run_parser.add_argument("--desc", type=str, default="", help="Detailed description of hypothesis/changes")
    run_parser.add_argument(
        "--model",
        choices=["lgbm", "ridge", "dummy", "ensemble"],
        default="lgbm",
        help="Model architecture (default: lgbm)",
    )
    run_parser.add_argument(
        "--target-transform",
        choices=["none", "log1p"],
        default="none",
        help="Target transformation (none vs log1p for log-normal salary modeling)",
    )
    run_parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    run_parser.add_argument("--alpha", type=float, default=10.0, help="Ridge alpha parameter")
    run_parser.add_argument("--leaves", type=int, default=15, help="LightGBM num_leaves")
    run_parser.add_argument("--lr", type=float, default=0.05, help="LightGBM learning_rate")
    run_parser.add_argument("--trees", type=int, default=80, help="LightGBM n_estimators")

    args = parser.parse_args()

    if args.action == "list":
        runs = list_experiments()
        print_runs_table(runs)

    elif args.action == "compare":
        if args.run1 and args.run2:
            r1 = get_experiment(args.run1)
            r2 = get_experiment(args.run2)
            if not r1:
                logger.error(f"Run 1 not found: {args.run1}")
                return
            if not r2:
                logger.error(f"Run 2 not found: {args.run2}")
                return
        else:
            latest = get_latest_runs(n=2)
            if len(latest) < 2:
                logger.error("At least 2 runs are required to compare. Run another experiment first.")
                return
            r1, r2 = latest[0], latest[1]

        compare_two_runs(r1, r2)

    elif args.action == "run":
        new_run = run_benchmark_experiment(
            name=args.name,
            description=args.desc,
            model_type=args.model,
            target_transform=args.target_transform,
            cv_folds=args.cv,
            alpha=args.alpha,
            num_leaves=args.leaves,
            learning_rate=args.lr,
            n_estimators=args.trees,
        )

        # Check if there is a previous run to compare against automatically
        latest = get_latest_runs(n=2)
        if len(latest) >= 2:
            prev_run = latest[0]
            compare_two_runs(prev_run, new_run)


if __name__ == "__main__":
    main()
