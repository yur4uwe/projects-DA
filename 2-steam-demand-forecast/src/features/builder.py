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

        # Filter strictly to the unbroken contiguous daily observation window (2026-07-21 onwards)
        # to eliminate artificial multi-year straight-line interpolation between monthly records.
        daily_records_mask = df["date"] >= "2026-07-21"
        if daily_records_mask.any():
            df = df[daily_records_mask].copy()

        # Harmonize single-sample peak measurements with 24-sample daily averages
        multi_sample = df[df["samples"] > 1]
        if not multi_sample.empty:
            avg_to_peak_ratio = float((multi_sample["avg_players"] / multi_sample["peak_players"]).mean())
        else:
            avg_to_peak_ratio = 0.67
        peak_ratio = 1.0 / max(avg_to_peak_ratio, 0.1)

        # Map target variable (concurrent player demand, consistent 24h daily average)
        df["demand"] = np.where(
            df["samples"] <= 1,
            df["avg_players"] * avg_to_peak_ratio,
            df["avg_players"],
        )

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
            appid=appid,
        )

        # 4. Autoregressive lags (Listing 5 in syllabus)
        with_lags = create_lag_features(with_promo, target_col="demand", prefix="sales")
        for lag_col in ["sales_lag_1", "sales_lag_2", "sales_lag_7", "sales_lag_14", "sales_lag_30"]:
            if lag_col in with_lags.columns:
                with_lags[lag_col] = with_lags[lag_col].bfill()

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
        with_rolling["peak_ratio"] = float(round(peak_ratio, 3))

        clean_features = with_rolling.copy()

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
