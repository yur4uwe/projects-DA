"""Dataset builder: orchestrates cleaning, baseline generation, labeling, and storage."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.features.baseline import assign_deal_tier, compute_out_of_fold_baseline
from src.features.cleaner import clean_and_parse_listings
from src.scraper.storage import validate_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_RAW_COLUMNS = [
    "Company",
    "Product",
    "TypeName",
    "Inches",
    "ScreenResolution",
    "Cpu",
    "Ram",
    "Memory",
    "Gpu",
    "OpSys",
    "Weight",
    "Price_euros",
]


def build_processed_dataset(
    raw_path: Path,
    output_dir: Path,
    baseline_model_path: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Build, label, and export the processed marketplace feature dataset."""
    logger.info("Loading raw dataset from %s...", raw_path)
    raw_df = validate_raw_dataset(raw_path, REQUIRED_RAW_COLUMNS)

    logger.info("Cleaning listings and extracting hardware specifications...")
    cleaned_df = clean_and_parse_listings(raw_df)

    logger.info("Computing out-of-fold specification baseline prices...")
    oof_baseline, baseline_model = compute_out_of_fold_baseline(cleaned_df)

    cleaned_df["market_baseline_price"] = oof_baseline
    cleaned_df["price_ratio"] = (
        cleaned_df["asking_price"] / cleaned_df["market_baseline_price"]
    ).round(4)
    cleaned_df["price_delta_euros"] = (
        cleaned_df["asking_price"] - cleaned_df["market_baseline_price"]
    ).round(2)
    cleaned_df["deal_tier"] = cleaned_df["price_ratio"].apply(assign_deal_tier)

    # Save baseline model if path specified
    if baseline_model_path is not None:
        baseline_model.save(baseline_model_path)

    # Save processed outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "pc_listings_processed.parquet"
    csv_path = output_dir / "pc_listings_processed.csv"

    cleaned_df.to_parquet(parquet_path, index=False)
    cleaned_df.to_csv(csv_path, index=False)

    tier_counts = cleaned_df["deal_tier"].value_counts().to_dict()
    logger.info("Processed dataset generated successfully: %d records.", len(cleaned_df))
    logger.info("Deal Tier distribution: %s", tier_counts)
    logger.info("Saved parquet to: %s", parquet_path)
    logger.info("Saved CSV to: %s", csv_path)

    return cleaned_df, parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature builder for secondary PC deal classifier."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/laptop_price.csv"),
        help="Path to raw dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed artifacts",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=Path,
        default=Path("models/baseline_regressor.joblib"),
        help="Path to save baseline regression model",
    )
    args = parser.parse_args()
    build_processed_dataset(args.raw_path, args.output_dir, args.baseline_model_path)


if __name__ == "__main__":
    main()
