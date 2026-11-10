"""Storage and validation utilities for raw dataset files."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def validate_raw_dataset(filepath: Path, required_columns: list[str]) -> pd.DataFrame:
    """Validate that the raw dataset file exists and contains all required columns."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset {filepath} is missing required columns: {missing}")

    logger.info("Successfully loaded and validated %d rows from %s", len(df), filepath)
    return df
