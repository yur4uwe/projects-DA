"""Data storage utilities for raw ingestion payloads."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class IngestionStorage:
    """Manages reading and writing raw payloads in data/raw/."""

    def __init__(self, raw_dir: Path | str = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, appid: int, metadata: BaseModel | dict[str, Any]) -> Path:
        """Save game metadata as JSON."""
        target = self.raw_dir / f"{appid}_metadata.json"
        data = metadata.model_dump() if isinstance(metadata, BaseModel) else metadata
        with target.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return target

    def load_metadata(self, appid: int) -> dict[str, Any]:
        """Load game metadata JSON if it exists."""
        target = self.raw_dir / f"{appid}_metadata.json"
        if not target.exists():
            return {}
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_records_jsonl(
        self, filename: str, records: list[BaseModel | dict[str, Any]]
    ) -> Path:
        """Save a list of records as newline-delimited JSON (JSONL)."""
        target = self.raw_dir / filename
        with target.open("w", encoding="utf-8") as f:
            for r in records:
                row = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target

    def load_records_jsonl(self, filename: str) -> list[dict[str, Any]]:
        """Load records from a JSONL file."""
        target = self.raw_dir / filename
        if not target.exists():
            return []
        records: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def save_catalog(self, catalog: list[dict[str, Any]]) -> Path:
        """Save summary catalog index."""
        target = self.raw_dir / "catalog.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        return target

    def load_catalog(self) -> list[dict[str, Any]]:
        """Load summary catalog index."""
        target = self.raw_dir / "catalog.json"
        if not target.exists():
            return []
        with target.open("r", encoding="utf-8") as f:
            return json.load(f)
