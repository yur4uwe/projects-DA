"""Training pipeline evaluating milestones M1-M4 and saving production hybrid model."""

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.model.baseline import SeasonalPersistenceBaseline
from src.model.hybrid import HybridProphetLGBMModel
from src.model.lgbm_model import LightGBMDemandModel
from src.model.metrics import calculate_metrics
from src.model.registry import ModelRegistry
from src.model.ridge_model import AutoregressiveRidgeModel

logger = logging.getLogger("model.train")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def train_eval_pipeline(
    data_path: str = "data/processed/daily_features.parquet",
    holdout_days: int = 30,
) -> None:
    """Train and evaluate M1, M2, M3, and M4 on strict temporal train/test split."""
    data_file = Path(data_path)
    if not data_file.exists():
        logger.error(f"Features file {data_path} not found. Run feature builder first!")
        return

    df = pd.read_parquet(data_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["appid", "date"]).reset_index(drop=True)

    # 1. Temporal train/test split per game: holdout the last 30 days of each game
    train_dfs = []
    test_dfs = []

    for appid, group in df.groupby("appid"):
        split_point = len(group) - holdout_days
        if split_point > 0:
            train_dfs.append(group.iloc[:split_point])
            test_dfs.append(group.iloc[split_point:])
        else:
            train_dfs.append(group)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    logger.info(
        f"Split dataset: {len(train_df):,} train samples, {len(test_df):,} test samples ({holdout_days} days holdout per game)."
    )

    registry = ModelRegistry()

    # -------------------------------------------------------------------------
    # Milestone 1: Seasonal Persistence Baseline
    # -------------------------------------------------------------------------
    logger.info("Training M1: Seasonal 7-day Persistence Baseline...")
    m1 = SeasonalPersistenceBaseline(seasonal_lag=7)
    m1.fit(train_df, train_df["demand"])
    m1_preds = m1.predict(test_df)
    m1_metrics = calculate_metrics(test_df["demand"], m1_preds)
    logger.info(
        f"M1 Metrics -> MAE: {m1_metrics['mae']}, WAPE: {m1_metrics['wape_percent']}%, RMSE: {m1_metrics['rmse']}, R2: {m1_metrics['r2']}"
    )
    registry.record_run(
        milestone="M1",
        model_name="Seasonal Persistence Baseline (lag-7)",
        metrics=m1_metrics,
        params={"seasonal_lag": 7},
        description="Naive seasonal persistence baseline predicting y_t = y_{t-7}.",
    )

    # -------------------------------------------------------------------------
    # Milestone 2: Autoregressive Ridge Regression
    # -------------------------------------------------------------------------
    logger.info("Training M2: Autoregressive Ridge Regression...")
    m2 = AutoregressiveRidgeModel(alpha=10.0)
    m2.fit(train_df, train_df["demand"])
    m2_preds = m2.predict(test_df)
    m2_metrics = calculate_metrics(test_df["demand"], m2_preds)
    logger.info(
        f"M2 Metrics -> MAE: {m2_metrics['mae']}, WAPE: {m2_metrics['wape_percent']}%, RMSE: {m2_metrics['rmse']}, R2: {m2_metrics['r2']}"
    )
    registry.record_run(
        milestone="M2",
        model_name="Autoregressive Ridge Regression",
        metrics=m2_metrics,
        params={"alpha": 10.0},
        description="L2-regularized linear model trained on standardized lags and calendar indicators.",
    )

    # -------------------------------------------------------------------------
    # Milestone 3: LightGBM Regressor (Listing 5)
    # -------------------------------------------------------------------------
    logger.info("Training M3: LightGBM Regressor with promotional interactions...")
    m3 = LightGBMDemandModel(n_estimators=500, learning_rate=0.05, num_leaves=31)
    m3.fit(train_df, train_df["demand"])
    m3_preds = m3.predict(test_df)
    m3_metrics = calculate_metrics(test_df["demand"], m3_preds)
    logger.info(
        f"M3 Metrics -> MAE: {m3_metrics['mae']}, WAPE: {m3_metrics['wape_percent']}%, RMSE: {m3_metrics['rmse']}, R2: {m3_metrics['r2']}"
    )
    registry.record_run(
        milestone="M3",
        model_name="LightGBM Demand Regressor",
        metrics=m3_metrics,
        params={"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31},
        description="Gradient-boosted decision trees with lag, rolling, and discount interaction features.",
    )

    # -------------------------------------------------------------------------
    # Milestone 4: Hybrid Prophet + LightGBM Ensemble (Listing 5 in syllabus)
    # -------------------------------------------------------------------------
    logger.info("Training M4: Hybrid Prophet + LightGBM Ensemble...")
    m4 = HybridProphetLGBMModel(
        yearly_seasonality=True,
        weekly_seasonality=True,
        lgb_n_estimators=400,
        lgb_learning_rate=0.03,
    )
    m4.fit(train_df, target_col="demand")
    m4_preds = m4.predict(test_df)
    m4_metrics = calculate_metrics(test_df["demand"], m4_preds)
    logger.info(
        f"M4 Metrics (Global) -> MAE: {m4_metrics['mae']}, WAPE: {m4_metrics['wape_percent']}%, RMSE: {m4_metrics['rmse']}, R2: {m4_metrics['r2']}"
    )

    registry.record_run(
        milestone="M4",
        model_name="Hybrid Prophet + LightGBM Ensemble",
        metrics=m4_metrics,
        params={
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "changepoint_prior_scale": 0.05,
            "lgb_n_estimators": 400,
            "lgb_learning_rate": 0.03,
        },
        description="Hybrid Prophet baseline on macro normalized demand + LightGBM residual correction across catalog.",
    )

    # Export metadata and save production artifacts
    registry.export_metadata_json("models/metadata.json")

    # Save models
    models_dir = Path("models")
    joblib.dump(m4, models_dir / "indie_demand_model.joblib")
    joblib.dump(m3, models_dir / "lgbm_global_model.joblib")
    logger.info(
        "Exported trained model artifacts to models/indie_demand_model.joblib and models/lgbm_global_model.joblib"
    )


if __name__ == "__main__":
    train_eval_pipeline()
