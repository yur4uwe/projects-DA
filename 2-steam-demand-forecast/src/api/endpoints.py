from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.app import (
    _get_game_features,
    state,
)
from src.api.models import ElasticityRequest, ForecastRequest
from src.model.predict import simulate_promotional_campaign

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint reporting service status and loaded model engines."""
    return {
        "status": "healthy",
        "loaded_models": list(state.models.keys()),
        "catalog_size": len(state.catalog),
        "production_milestone": "M4",
    }


@router.get("/catalog")
def get_catalog() -> list[dict[str, Any]]:
    """Return all ingested games with metadata."""
    return state.catalog


@router.get("/experiments")
def get_experiments() -> list[dict[str, Any]]:
    """Return experiment history and metrics from SQLite."""
    df = state.registry.list_experiments()
    return df.to_dict(orient="records")


@router.get("/games/{appid}/history")
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


@router.get("/models/importances")
def get_feature_importances() -> list[dict[str, Any]]:
    """Return top feature importances from the global GBDT model."""
    lgbm_model = state.models.get("lgbm")
    if lgbm_model and hasattr(lgbm_model, "feature_importances_"):
        fi_df = lgbm_model.feature_importances_.head(12)
        return fi_df.to_dict(orient="records")
    return []


@router.post("/forecast")
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


@router.post("/elasticity")
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
