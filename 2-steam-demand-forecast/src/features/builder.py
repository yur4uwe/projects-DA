"""Pipeline builder compiling raw time-series into processed feature matrices."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.calendar import create_calendar_features
from src.features.cleaner import align_daily_series
from src.features.lags import create_lag_features
from src.features.promotions import create_promotional_features
from src.features.rolling import create_rolling_features
from src.ingestion.sales_calendar import SteamSalesCalendar
from src.ingestion.storage import IngestionStorage

logger = logging.getLogger("features.builder")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


class FeaturePipeline:
    """End-to-end feature engineering pipeline for Steam time-series."""

    def __init__(
        self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"
    ) -> None:
        self.storage = IngestionStorage(raw_dir=raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.calendar = SteamSalesCalendar()

    def process_game(self, appid: int) -> pd.DataFrame:
        """Process daily time series and metadata for a single game into features."""
        meta = self.storage.load_metadata(appid)
        records = self.storage.load_records_jsonl(f"{appid}_daily_series.jsonl")

        if not records:
            logger.warning(f"No daily records found for AppID {appid}")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        # Map target variable (concurrent player demand)
        df["demand"] = df["avg_players"]

        # 1. Calendar alignment (unbroken daily grid)
        aligned = align_daily_series(df, date_col="date")

        # 2. Calendar and cyclical features
        with_cal = create_calendar_features(aligned, date_col="date")

        # 3. Promotional features
        base_price = meta.get("pricing", {}).get("initial_price", 14.99)
        with_promo = create_promotional_features(
            with_cal,
            date_col="date",
            calendar=self.calendar,
            base_price=base_price,
        )

        # Reconstruct promotional surge dynamics for interpolated historical dates
        raw_dates = {
            pd.to_datetime(r["date"]).date()
            for r in records
            if r.get("samples", 1) > 1
        }
        dates_s = pd.to_datetime(with_promo["date"]).dt.date
        is_interp = ~dates_s.isin(raw_dates)
        promo_mask = (with_promo["is_steam_sale"] == 1) & is_interp
        if promo_mask.any():
            # Price elasticity lift (epsilon ~ 0.8) with weekend interaction
            elasticity_lift = (1.0 - with_promo.loc[promo_mask, "discount_depth"]).pow(-0.8) - 1.0
            weekend_boost = 1.0 + 0.15 * with_promo.loc[promo_mask, "is_weekend"]
            with_promo.loc[promo_mask, "demand"] = (
                with_promo.loc[promo_mask, "demand"]
                * (1.0 + elasticity_lift * weekend_boost)
            )

        # Reconstruct organic post-sale gameplay tail (players installing and playing newly purchased copies)
        tail_mask = is_interp & (with_promo["is_steam_sale"] == 0) & (with_promo["days_since_last_sale"].between(1, 7))
        if tail_mask.any():
            typical_elasticity = 0.50
            tail_decay = 0.65 * np.exp(-with_promo.loc[tail_mask, "days_since_last_sale"] / 3.5)
            weekend_tail = 1.0 + 0.15 * with_promo.loc[tail_mask, "is_weekend"]
            with_promo.loc[tail_mask, "demand"] = (
                with_promo.loc[tail_mask, "demand"]
                * (1.0 + typical_elasticity * tail_decay * weekend_tail)
            )

        # Smooth non-promotional viral anomalies on interpolated series (e.g. Silksong release spike)
        # Prevents a single monthly snapshot peak from stretching into a multi-month 95k plateau
        if is_interp.any():
            non_promo_interp = is_interp & (with_promo["is_steam_sale"] == 0)
            robust_median = float(with_promo["demand"].median())
            viral_ceiling = max(robust_median * 3.0, 25000.0)
            excess_mask = non_promo_interp & (with_promo["demand"] > viral_ceiling)
            if excess_mask.any():
                with_promo.loc[excess_mask, "demand"] = viral_ceiling

        # 4. Autoregressive lags (Listing 5 in syllabus)
        with_lags = create_lag_features(with_promo, target_col="demand", prefix="sales")

        # 5. Rolling window statistics (Listing 5 in syllabus)
        with_rolling = create_rolling_features(
            with_lags, target_col="demand", prefix="sales"
        )

        # 6. Static game attributes
        with_rolling["appid"] = appid
        with_rolling["game_name"] = meta.get("name", f"Game {appid}")
        with_rolling["base_price"] = base_price
        with_rolling["is_multiplayer"] = 1 if meta.get("is_multiplayer") else 0
        with_rolling["is_coop"] = 1 if meta.get("is_coop") else 0
        pos_rev = meta.get("positive_reviews", 1000)
        with_rolling["log_positive_reviews"] = float(np.log1p(pos_rev))

        # Drop initial burn-in window where lag 30 is NaN
        clean_features = with_rolling.dropna(subset=["sales_lag_30"]).copy()

        # Save single-game parquet
        game_out = self.processed_dir / f"{appid}_features.parquet"
        clean_features.to_parquet(game_out, index=False)
        logger.info(
            f"[{meta.get('name')}] Saved {len(clean_features)} rows to {game_out}"
        )

        return clean_features

    def build_all(self) -> pd.DataFrame:
        """Process all games listed in catalog and create master dataset."""
        catalog = self.storage.load_catalog()
        if not catalog:
            logger.error(
                "Catalog not found in data/raw/catalog.json. Run ingestion first!"
            )
            return pd.DataFrame()

        all_dfs: list[pd.DataFrame] = []
        for entry in catalog:
            aid = entry["appid"]
            df_game = self.process_game(aid)
            if not df_game.empty:
                all_dfs.append(df_game)

        if not all_dfs:
            return pd.DataFrame()

        master_df = pd.concat(all_dfs, ignore_index=True)
        master_parquet = self.processed_dir / "daily_features.parquet"
        master_csv = self.processed_dir / "daily_features.csv"

        master_df.to_parquet(master_parquet, index=False)
        master_df.to_csv(master_csv, index=False)

        logger.info(
            f"Successfully built master feature matrix with {len(master_df)} rows and "
            f"{len(master_df.columns)} columns across {len(all_dfs)} games."
        )
        logger.info(f"Saved to {master_parquet} and {master_csv}")
        return master_df


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.build_all()
