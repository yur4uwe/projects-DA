"""Entrypoint script to run the standalone forecasting HTTP server."""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Steam Demand Forecasting Service on http://{host}:{port} ...")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False, workers=1)
