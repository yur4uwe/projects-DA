from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .cleaner import clean_record
from .skills import TECH_SKILL_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_feature_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    salary_only: bool = False,
) -> pd.DataFrame:
    """Read raw JSONL files, clean records, engineer features, and export dataset."""
    raw_files = sorted(raw_data_dir.glob("*.jsonl"))
    if not raw_files:
        logger.warning(f"No .jsonl files found in {raw_data_dir}")
        return pd.DataFrame()

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
                    # Deduplicate across runs by source and id
                    unique_key = (rec.source, rec.id)
                    if unique_key in seen_ids:
                        continue
                    seen_ids.add(unique_key)
                    cleaned_records.append(asdict(rec))
                except Exception as exc:
                    logger.debug(f"Failed processing record: {exc}")

    if not cleaned_records:
        logger.warning("No records successfully parsed.")
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_records)
    logger.info(f"Loaded {len(df)} unique records.")

    # 1. Feature Engineering
    df["num_skills"] = df["skills"].apply(lambda s: len(s) if isinstance(s, list) else 0)
    df["title_char_length"] = df["title"].str.len().fillna(0).astype(int)
    df["desc_word_count"] = df["description_cleaned"].str.split().apply(lambda w: len(w) if isinstance(w, list) else 0)

    # 2. Binary Skill Features (one column per canonical skill)
    for skill_name in TECH_SKILL_PATTERNS.keys():
        col_name = f"skill_{skill_name.replace('-', '_').replace('/', '_')}"
        df[col_name] = df["skills"].apply(lambda s_list: int(skill_name in s_list) if isinstance(s_list, list) else 0)

    # 3. Filter for salary regression target if requested
    if salary_only:
        initial_len = len(df)
        df = df[df["salary_mid_usd"].notnull()].copy()
        logger.info(f"Filtered for records with salary: {len(df)} of {initial_len} retained.")

    # 4. Save to processed directory
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / ("salary_features.parquet" if salary_only else "all_jobs_features.parquet")
    csv_path = output_dir / ("salary_features.csv" if salary_only else "all_jobs_features.csv")

    # Parquet cannot serialize mixed objects easily if list is empty or inconsistent, so store skills as comma-separated in CSV/parquet
    df_export = df.copy()
    df_export["skills_str"] = df_export["skills"].apply(lambda s: ", ".join(s) if isinstance(s, list) else "")

    df_export.to_parquet(parquet_path, index=False)
    df_export.to_csv(csv_path, index=False)

    logger.info(f"Saved feature dataset ({len(df_export)} rows) to:")
    logger.info(f"  - {parquet_path}")
    logger.info(f"  - {csv_path}")

    return df_export


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    proc_dir = project_root / "data" / "processed"

    build_feature_dataset(raw_dir, proc_dir, salary_only=False)
