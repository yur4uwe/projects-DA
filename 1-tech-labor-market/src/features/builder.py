import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaner import clean_record
from .composites import add_skill_composites_to_df

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def build_feature_dataset(
    raw_data_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw JSONL files, clean records, engineer composite features, and export datasets."""
    raw_files = sorted(raw_data_dir.glob("*.jsonl"))
    if not raw_files:
        logger.warning(f"No .jsonl files found in {raw_data_dir}")
        return pd.DataFrame(), pd.DataFrame()

    cleaned_records = []
    seen_ids = set()

    for file_path in raw_files:
        logger.info(f"Processing raw file: {file_path.name}")
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    raw_dict = json.loads(line)
                    rec = clean_record(raw_dict)
                    unique_key = (rec.source, rec.id)
                    if unique_key in seen_ids:
                        continue
                    seen_ids.add(unique_key)
                    cleaned_records.append(asdict(rec))
                except Exception as exc:
                    logger.debug(f"Failed processing record: {exc}")

    if not cleaned_records:
        logger.warning("No records successfully parsed.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(cleaned_records)
    logger.info(f"Loaded {len(df)} unique records.")

    # 1. Text statistics
    df["title_char_length"] = df["title"].str.len().fillna(0).astype(int)
    df["desc_word_count"] = (
        df["description_cleaned"]
        .str.split()
        .apply(lambda w: len(w) if isinstance(w, list) else 0)
    )

    # 2. Add composite domain aggregates and individual binary skill flags
    df = add_skill_composites_to_df(df, skills_col="skills")

    # 3. Target transformations (log-salary for right-skew mitigation)
    df["log_salary_mid_usd"] = np.where(
        df["salary_mid_usd"].notnull(),
        np.log1p(df["salary_mid_usd"]),
        np.nan,
    )

    # Flatten skills list into comma-delimited string for clean parquet serialization
    df["skills_str"] = df["skills"].apply(
        lambda s: ", ".join(s) if isinstance(s, list) else ""
    )

    # 4. Partition into all records vs. salary-labeled subset for training
    df_labeled = df[df["salary_mid_usd"].notnull()].copy()

    # 5. Export datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    all_parquet = output_dir / "all_jobs_features.parquet"
    all_csv = output_dir / "all_jobs_features.csv"
    labeled_parquet = output_dir / "salary_labeled_features.parquet"
    labeled_csv = output_dir / "salary_labeled_features.csv"

    df.to_parquet(all_parquet, index=False)
    df.to_csv(all_csv, index=False)

    df_labeled.to_parquet(labeled_parquet, index=False)
    df_labeled.to_csv(labeled_csv, index=False)

    logger.info(
        f"Saved complete dataset ({len(df)} rows, {df.shape[1]} cols) to {all_parquet.name}"
    )
    logger.info(
        f"Saved salary-labeled dataset ({len(df_labeled)} rows) to {labeled_parquet.name}"
    )

    return df, df_labeled


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    proc_dir = project_root / "data" / "processed"

    build_feature_dataset(raw_dir, proc_dir)
