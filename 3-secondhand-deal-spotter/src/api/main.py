"""FastAPI server exposing deal tier prediction and static data endpoints."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.schema import DEAL_TIERS
from src.models.predict import predict_deal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pc_listings_processed.parquet"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
IMPORTANCE_PATH = PROJECT_ROOT / "models" / "global_feature_importance.json"

app = FastAPI(title="Deal Spotter API", version="0.1.0")


class ListingInput(BaseModel):
    Company: str
    TypeName: str
    Inches: float = Field(ge=10.0, le=20.0)
    Ram_GB: int
    Weight_kg: float
    Cpu_Brand: str
    Cpu_Tier: str
    Cpu_Freq_GHz: float
    Primary_Storage_GB: int
    Primary_Storage_Type: str
    Total_Storage_GB: int
    Gpu_Brand: str
    Gpu_Tier: str
    OpSys: str
    Screen_Width: int
    Screen_Height: int
    Is_Touchscreen: bool
    Is_IPS: bool
    asking_price: float


class PredictionOutput(BaseModel):
    deal_tier: str
    confidence: float
    probabilities: dict[str, float]
    asking_price: float
    market_baseline_price: float
    price_delta_euros: float
    price_ratio: float
    relative_savings_pct: float


# Cache loaded data in module scope.
_cached_data: pd.DataFrame | None = None
_cached_metadata: dict[str, Any] | None = None
_cached_importance: list[dict[str, Any]] | None = None


def _load_data() -> pd.DataFrame:
    global _cached_data
    if _cached_data is None:
        if DATA_PATH.exists():
            _cached_data = pd.read_parquet(DATA_PATH)
        else:
            _cached_data = pd.DataFrame()
    return _cached_data


def _load_metadata() -> dict[str, Any]:
    global _cached_metadata
    if _cached_metadata is None:
        if METADATA_PATH.exists():
            with METADATA_PATH.open("r", encoding="utf-8") as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {}
    return _cached_metadata


def _load_importance() -> list[dict[str, Any]]:
    global _cached_importance
    if _cached_importance is None:
        if IMPORTANCE_PATH.exists():
            with IMPORTANCE_PATH.open("r", encoding="utf-8") as f:
                _cached_importance = json.load(f)
        else:
            _cached_importance = []
    return _cached_importance


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(listing: ListingInput) -> PredictionOutput:
    try:
        result = predict_deal(listing.model_dump())
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    return PredictionOutput(**result)


@app.get("/data/listings")
def get_listings() -> dict[str, Any]:
    """Return processed listings dataset as JSON records."""
    df = _load_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Processed dataset not found.")
    # Replace NaN/inf with None for JSON serialization
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    records = json.loads(df_clean.to_json(orient="records"))
    return {"count": len(records), "records": records}


@app.get("/data/metadata")
def get_metadata() -> dict[str, Any]:
    meta = _load_metadata()
    if not meta:
        raise HTTPException(status_code=404, detail="Metadata not found.")
    return meta


@app.get("/data/feature-importance")
def get_feature_importance() -> list[dict[str, Any]]:
    importance = _load_importance()
    if not importance:
        raise HTTPException(status_code=404, detail="Feature importance not found.")
    return importance


@app.get("/data/unique-values")
def get_unique_values() -> dict[str, list[str]]:
    """Return unique values for categorical columns used by the frontend form."""
    df = _load_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Processed dataset not found.")
    return {
        "companies": sorted(df["Company"].unique().tolist()),
        "type_names": sorted(df["TypeName"].unique().tolist()),
        "deal_tiers": DEAL_TIERS,
    }
