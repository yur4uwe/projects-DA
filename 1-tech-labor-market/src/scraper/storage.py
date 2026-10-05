import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_raw_jsonl(
    records: list[dict[str, Any]], source_name: str, base_dir: Path
) -> Path:
    """Save raw extracted records into a timestamped .jsonl file."""
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    file_path = base_dir / f"{source_name}_raw_{timestamp}.jsonl"

    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {file_path}")
    return file_path
