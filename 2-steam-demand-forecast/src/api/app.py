"""FastAPI HTTP Forecasting Service for Steam Indie Demand Prediction."""

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sales_calendar import SteamSalesCalendar
from src.ingestion.storage import IngestionStorage
from src.models.registry import ModelRegistry

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class GlobalAppState:
    models: dict[str, Any] = {}
    catalog: list[dict[str, Any]] = []
    calendar: SteamSalesCalendar = SteamSalesCalendar()
    registry: ModelRegistry = ModelRegistry(MODELS_DIR / "experiments.db")


state = GlobalAppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models, catalog, and calendar artifacts into memory on startup."""
    storage = IngestionStorage(raw_dir=PROJECT_ROOT / "data" / "raw")
    state.catalog = storage.load_catalog()

    hybrid_path = MODELS_DIR / "indie_demand_model.joblib"
    lgbm_path = MODELS_DIR / "lgbm_global_model.joblib"

    if hybrid_path.exists():
        state.models["hybrid"] = joblib.load(hybrid_path)
    if lgbm_path.exists():
        state.models["lgbm"] = joblib.load(lgbm_path)

    yield
    state.models.clear()


app = FastAPI(
    title="Steam Demand Forecasting API",
    description="Dedicated asynchronous HTTP service offloading heavy Prophet + LightGBM recursive forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_game_features(appid: int) -> pd.DataFrame:
    parquet_path = DATA_DIR / f"{appid}_features.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Game features not found for AppID {appid}"
        )
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
