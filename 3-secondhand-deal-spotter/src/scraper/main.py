"""Dataset ingestion and verification runner for secondary PC marketplace listings."""

import argparse
import logging
from pathlib import Path

from src.scraper.storage import validate_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_PC_COLUMNS = [
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


def run_ingestion(data_path: Path) -> Path:
    """Validate and report on ingested secondary PC listings."""
    logger.info("Ingesting and verifying raw dataset at %s...", data_path)
    df = validate_raw_dataset(data_path, REQUIRED_PC_COLUMNS)
    logger.info("Verified %d valid listings ready for feature extraction.", len(df))
    return data_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PC listings dataset verification runner.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/laptop_price.csv"),
        help="Path to raw dataset CSV",
    )
    args = parser.parse_args()
    run_ingestion(args.input)


if __name__ == "__main__":
    main()
