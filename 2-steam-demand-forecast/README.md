# Steam Player Demand & Trend Forecasting

## Objective
Ingest historical player activity from the Steam platform and build time-series models to forecast concurrent player demand and analyze promotional impacts.

---

## Running Instructions

### Local Execution with uv
```bash
# Sync dependencies (creates/updates environment automatically)
uv sync

# Run the Streamlit dashboard
uv run streamlit run src/app/main.py
```

To add new dependencies to this project:
```bash
uv add <package-name>
```

### Containerized Execution with Docker
```bash
# Build Docker image
docker build -t steam-demand-forecast .

# Run container
docker run -p 8501:8501 steam-demand-forecast
```

---

## Directory Structure

```text
02-steam-demand-forecast/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   └── app/
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Status & Roadmap
- [ ] Ingestion pipeline setup (API / historical data)
- [ ] Time-series cleaning & calendar alignment
- [ ] Lag and rolling feature generation
- [ ] Model training (baseline and tree-based)
- [ ] Streamlit interface
- [ ] Docker packaging
