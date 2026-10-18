#!/usr/bin/env bash
set -e

echo "Starting FastAPI Demand Forecasting Service on port 8000..."
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit Decision Studio on port 8501..."
exec uv run streamlit run src/app/main.py --server.port=8501 --server.address=0.0.0.0
