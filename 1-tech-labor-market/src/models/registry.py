import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "experiments.db"
)


@dataclass
class ExperimentRecord:
    run_id: str
    timestamp: str
    name: str
    description: str
    model_name: str
    target_transform: str
    sample_size: int
    cv_folds: int
    feature_count: int
    features_used: list[str]
    mae: float
    rmse: float
    r2: float
    medae: float
    mape: float
    hyperparams: dict[str, Any]
    extra_metadata: dict[str, Any] | None = None


def get_db_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                model_name TEXT NOT NULL,
                target_transform TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                cv_folds INTEGER NOT NULL,
                feature_count INTEGER NOT NULL,
                features_used TEXT NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                r2 REAL NOT NULL,
                medae REAL NOT NULL,
                mape REAL NOT NULL,
                hyperparams TEXT,
                extra_metadata TEXT
            )
            """
        )
    conn.close()


def log_experiment(record: ExperimentRecord, db_path: Path | None = None) -> str:
    init_db(db_path)
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                run_id, timestamp, name, description, model_name, target_transform,
                sample_size, cv_folds, feature_count, features_used,
                mae, rmse, r2, medae, mape, hyperparams, extra_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id,
                record.timestamp,
                record.name,
                record.description,
                record.model_name,
                record.target_transform,
                record.sample_size,
                record.cv_folds,
                record.feature_count,
                json.dumps(record.features_used),
                record.mae,
                record.rmse,
                record.r2,
                record.medae,
                record.mape,
                json.dumps(record.hyperparams),
                json.dumps(record.extra_metadata or {}),
            ),
        )
    conn.close()
    return record.run_id


def row_to_record(row: sqlite3.Row) -> ExperimentRecord:
    return ExperimentRecord(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        name=row["name"],
        description=row["description"] or "",
        model_name=row["model_name"],
        target_transform=row["target_transform"],
        sample_size=row["sample_size"],
        cv_folds=row["cv_folds"],
        feature_count=row["feature_count"],
        features_used=json.loads(row["features_used"]),
        mae=float(row["mae"]),
        rmse=float(row["rmse"]),
        r2=float(row["r2"]),
        medae=float(row["medae"]),
        mape=float(row["mape"]),
        hyperparams=json.loads(row["hyperparams"] or "{}"),
        extra_metadata=json.loads(row["extra_metadata"] or "{}"),
    )


def list_experiments(
    db_path: Path | None = None, limit: int = 100
) -> list[ExperimentRecord]:
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY timestamp ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [row_to_record(r) for r in rows]


def get_experiment(
    identifier: str, db_path: Path | None = None
) -> ExperimentRecord | None:
    init_db(db_path)
    all_runs = list_experiments(db_path)
    if not all_runs:
        return None

    clean_id = identifier.strip().lower()

    if clean_id in ("latest", "last"):
        return all_runs[-1]
    if clean_id in ("prev", "previous"):
        return all_runs[-2] if len(all_runs) >= 2 else all_runs[0]

    # Handle step tags: M1, m1, M2, 1, 2, etc.
    if clean_id.startswith("m") and clean_id[1:].isdigit():
        idx = int(clean_id[1:]) - 1
        if 0 <= idx < len(all_runs):
            return all_runs[idx]
    if clean_id.isdigit():
        idx = int(clean_id) - 1
        if 0 <= idx < len(all_runs):
            return all_runs[idx]

    # Exact match on run_id or name (case-insensitive)
    for r in all_runs:
        if r.run_id.lower() == clean_id or r.name.lower() == clean_id:
            return r

    # Prefix or substring match on run_id or name
    for r in all_runs:
        if clean_id in r.run_id.lower() or clean_id in r.name.lower():
            return r

    return None


def get_latest_runs(db_path: Path | None = None, n: int = 2) -> list[ExperimentRecord]:
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?", (n,))
    rows = cursor.fetchall()
    conn.close()
    return [row_to_record(r) for r in reversed(rows)]
