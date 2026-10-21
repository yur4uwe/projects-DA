import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


# -----------------------------------------------------------------------------
# Strict API Service Health Check
# -----------------------------------------------------------------------------
def query_api_health() -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# API Client Functions
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_catalog() -> list[dict[str, Any]]:
    resp = httpx.get(f"{API_URL}/catalog", timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def fetch_game_history(appid: int, limit: int = 60) -> pd.DataFrame:
    resp = httpx.get(
        f"{API_URL}/games/{appid}/history", params={"limit": limit}, timeout=10.0
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def execute_forecast_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(f"{API_URL}/forecast", json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def execute_elasticity_simulation(payload: dict[str, Any]) -> list[dict[str, Any]]:
    resp = httpx.post(f"{API_URL}/elasticity", json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_experiments() -> list[dict[str, Any]]:
    resp = httpx.get(f"{API_URL}/experiments", timeout=5.0)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_feature_importances() -> list[dict[str, Any]]:
    resp = httpx.get(f"{API_URL}/models/importances", timeout=5.0)
    resp.raise_for_status()
    return resp.json()
