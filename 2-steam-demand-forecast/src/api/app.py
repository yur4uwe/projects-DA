"""FastAPI HTTP Forecasting Service for Steam Indie Demand Prediction."""

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sales_calendar import SteamSalesCalendar
from src.ingestion.storage import IngestionStorage
from src.models.predict import simulate_promotional_campaign
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


class ForecastRequest(BaseModel):
    appid: int = Field(default=367520, description="Steam AppID")
    horizon_days: int = Field(
        default=14, ge=1, le=60, description="Forecast horizon in days"
    )
    start_date: date | None = Field(
        default=None, description="Start date for simulation"
    )
    discount_depth: float = Field(
        default=0.33,
        ge=0.0,
        le=0.90,
        description="Planned discount percentage (0.0 to 0.9)",
    )
    sale_duration_days: int = Field(
        default=7, ge=1, le=30, description="Sale duration in days"
    )
    is_custom_event: bool = Field(
        default=False, description="Whether this is a custom publisher deal"
    )
    custom_event_name: str = Field(
        default="Custom Publisher Sale", description="Name of custom event"
    )
    model_engine: str = Field(default="hybrid", description="'hybrid' or 'lgbm'")


class ElasticityRequest(BaseModel):
    appid: int = Field(default=367520, description="Steam AppID")
    base_price: float = Field(default=14.99, gt=0, description="Base price in USD")
    start_date: date | None = Field(
        default=None, description="Start date for discount test"
    )
    sale_duration_days: int = Field(
        default=7, ge=1, le=30, description="Sale duration in days"
    )
    is_custom_event: bool = Field(default=False)
    test_discounts: list[float] = Field(default=[0.10, 0.20, 0.33, 0.50, 0.66, 0.75])
    model_engine: str = Field(default="hybrid")


def _get_game_features(appid: int) -> pd.DataFrame:
    parquet_path = DATA_DIR / f"{appid}_features.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Game features not found for AppID {appid}"
        )
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint reporting service status and loaded model engines."""
    return {
        "status": "healthy",
        "loaded_models": list(state.models.keys()),
        "catalog_size": len(state.catalog),
        "production_milestone": "M4",
    }


@app.get("/catalog")
def get_catalog() -> list[dict[str, Any]]:
    """Return all ingested games with metadata."""
    return state.catalog


@app.get("/experiments")
def get_experiments() -> list[dict[str, Any]]:
    """Return experiment history and metrics from SQLite."""
    df = state.registry.list_experiments()
    return df.to_dict(orient="records")


@app.get("/games/{appid}/history")
def get_game_history(appid: int, limit: int = 60) -> list[dict[str, Any]]:
    """Return recent historical player counts and rolling baseline for a game."""
    df = _get_game_features(appid)
    recent = df.tail(limit).copy()
    records = []
    for _, row in recent.iterrows():
        records.append(
            {
                "date": str(row["date"].date()),
                "demand": float(row["demand"]),
                "sales_rolling_7": float(row.get("sales_rolling_7", row["demand"])),
            }
        )
    return records


@app.get("/models/importances")
def get_feature_importances() -> list[dict[str, Any]]:
    """Return top feature importances from the global GBDT model."""
    lgbm_model = state.models.get("lgbm")
    if lgbm_model and hasattr(lgbm_model, "feature_importances_"):
        fi_df = lgbm_model.feature_importances_.head(12)
        return fi_df.to_dict(orient="records")
    return []


@app.post("/forecast")
def predict_forecast(req: ForecastRequest) -> dict[str, Any]:
    """Execute multi-step recursive forecasting asynchronously outside the UI thread."""
    model_key = "hybrid" if "hybrid" in req.model_engine.lower() else "lgbm"
    if model_key not in state.models:
        raise HTTPException(
            status_code=500, detail=f"Model engine '{model_key}' not loaded."
        )

    game_df = _get_game_features(req.appid)
    model = state.models[model_key]

    sim_res = simulate_promotional_campaign(
        history_df=game_df,
        model=model,
        horizon_days=req.horizon_days,
        start_date=req.start_date,
        discount_depth=req.discount_depth,
        sale_duration_days=req.sale_duration_days,
        is_custom_event=req.is_custom_event,
        custom_event_name=req.custom_event_name,
        calendar=state.calendar,
    )

    # Convert DataFrame to serializable dict
    forecast_records = sim_res["forecast_df"].to_dict(orient="records")
    for r in forecast_records:
        r["date"] = str(r["date"])

    return {
        "appid": req.appid,
        "model_engine": model_key,
        "horizon_days": sim_res["horizon_days"],
        "recent_baseline_ccu": sim_res["recent_baseline_ccu"],
        "peak_promo_ccu": sim_res["peak_promo_ccu"],
        "peak_surge_multiplier": sim_res["peak_surge_multiplier"],
        "total_incremental_player_days": sim_res["total_incremental_player_days"],
        "cooldown_compliant": sim_res["cooldown_compliant"],
        "cooldown_note": sim_res["cooldown_note"],
        "forecast_series": forecast_records,
    }


@app.post("/elasticity")
def compute_elasticity(req: ElasticityRequest) -> list[dict[str, Any]]:
    """Calculate promotional sensitivity curve across multiple discount thresholds."""
    model_key = "hybrid" if "hybrid" in req.model_engine.lower() else "lgbm"
    if model_key not in state.models:
        raise HTTPException(
            status_code=500, detail=f"Model engine '{model_key}' not loaded."
        )

    game_df = _get_game_features(req.appid)
    model = state.models[model_key]

    results: list[dict[str, Any]] = []
    for d in req.test_discounts:
        sim = simulate_promotional_campaign(
            history_df=game_df,
            model=model,
            horizon_days=14,
            start_date=req.start_date,
            discount_depth=d,
            sale_duration_days=req.sale_duration_days,
            is_custom_event=req.is_custom_event,
            calendar=state.calendar,
        )
        effective_p = req.base_price * (1.0 - d)
        results.append(
            {
                "discount_percent": f"{int(d * 100)}%",
                "discount_decimal": d,
                "effective_price": round(effective_p, 2),
                "peak_ccu": sim["peak_promo_ccu"],
                "surge_multiplier": sim["peak_surge_multiplier"],
                "incremental_player_days": sim["total_incremental_player_days"],
            }
        )

    return results
