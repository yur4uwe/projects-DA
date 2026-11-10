"""Unit tests for dataset ingestion and storage verification."""

from pathlib import Path

import pytest

from src.scraper.main import REQUIRED_PC_COLUMNS, run_ingestion
from src.scraper.storage import validate_raw_dataset


def test_validate_raw_dataset_success() -> None:
    data_path = Path("data/raw/laptop_price.csv")
    assert data_path.exists(), "Raw dataset file must exist"
    df = validate_raw_dataset(data_path, REQUIRED_PC_COLUMNS)
    assert len(df) > 1000
    for col in REQUIRED_PC_COLUMNS:
        assert col in df.columns


def test_validate_raw_dataset_missing_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        validate_raw_dataset(non_existent, REQUIRED_PC_COLUMNS)


def test_validate_raw_dataset_missing_column(tmp_path: Path) -> None:
    dummy_file = tmp_path / "dummy.csv"
    dummy_file.write_text("Company,Price_euros\nDell,1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_dataset(dummy_file, REQUIRED_PC_COLUMNS)


def test_run_ingestion_runner() -> None:
    data_path = Path("data/raw/laptop_price.csv")
    result_path = run_ingestion(data_path)
    assert result_path == data_path
