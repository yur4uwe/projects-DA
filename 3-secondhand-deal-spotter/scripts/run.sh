#!/usr/bin/env bash
# Start both backend and frontend services locally (without Docker).
# Usage: ./scripts/run.sh [--backend-only | --frontend-only]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
export API_URL="${API_URL:-http://${API_HOST}:${API_PORT}}"

cleanup() {
    echo "Shutting down..."
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

MODE="${1:-all}"

if [ "$MODE" != "--frontend-only" ]; then
    echo "Starting backend on ${API_HOST}:${API_PORT}..."
    uv run uvicorn src.api.main:app --host "$API_HOST" --port "$API_PORT" &
    BACKEND_PID=$!

    # Wait for backend to be ready
    for i in $(seq 1 30); do
        if curl -sf "http://${API_HOST}:${API_PORT}/health" > /dev/null 2>&1; then
            echo "Backend is ready."
            break
        fi
        sleep 1
    done
fi

if [ "$MODE" != "--backend-only" ]; then
    echo "Starting frontend on port ${STREAMLIT_PORT}..."
    uv run streamlit run src/app/main.py \
        --server.port="$STREAMLIT_PORT" \
        --server.address="0.0.0.0" &
    FRONTEND_PID=$!
fi

echo "Services running. Press Ctrl+C to stop."
wait
