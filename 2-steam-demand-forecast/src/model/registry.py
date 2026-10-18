"""Model experiment registry and SQLite tracker for milestone evolution."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class ModelRegistry:
    """Manages SQLite experiment tracking and metadata snapshots."""

    def __init__(self, db_path: str | Path = "models/experiments.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    milestone TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    mae REAL NOT NULL,
                    rmse REAL NOT NULL,
                    wape REAL NOT NULL,
                    r2 REAL NOT NULL,
                    params TEXT,
                    description TEXT
                )
                """
            )
            conn.commit()

    def record_run(
        self,
        milestone: str,
        model_name: str,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        description: str = "",
    ) -> int:
        """Insert or replace milestone experiment result."""
        created_at = datetime.now(UTC).isoformat()
        params_str = json.dumps(params or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO experiments 
                (milestone, model_name, created_at, mae, rmse, wape, r2, params, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    milestone,
                    model_name,
                    created_at,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["wape"],
                    metrics["r2"],
                    params_str,
                    description,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_experiments(self) -> pd.DataFrame:
        """Return all tracked milestone experiments as a DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT id, milestone, model_name, created_at, mae, rmse, wape, r2, description FROM experiments ORDER BY id ASC",
                conn,
            )
            return df

    def get_experiment(self, identifier: str) -> dict[str, Any] | None:
        """Fetch experiment by milestone identifier (e.g. 'M1', 'M4', 'latest')."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if identifier.lower() == "latest":
                cursor.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT 1")
            else:
                cursor.execute(
                    "SELECT * FROM experiments WHERE UPPER(milestone) = UPPER(?) OR id = ?",
                    (identifier, identifier if identifier.isdigit() else -1),
                )
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["params"] = json.loads(d.get("params") or "{}")
                return d
            return None

    def export_metadata_json(
        self, target_path: str | Path = "models/metadata.json"
    ) -> None:
        """Export best model metadata snapshot for app inference."""
        latest = self.get_experiment("latest")
        if not latest:
            return

        all_exp = self.list_experiments().to_dict(orient="records")
        payload = {
            "production_milestone": latest["milestone"],
            "production_model_name": latest["model_name"],
            "updated_at": latest["created_at"],
            "metrics": {
                "mae": latest["mae"],
                "rmse": latest["rmse"],
                "wape": latest["wape"],
                "r2": latest["r2"],
            },
            "all_milestones": all_exp,
        }

        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
