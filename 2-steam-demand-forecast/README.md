# Steam Indie Game Demand Forecasting & Promotional Scheduler

## Objective
Ingest historical player activity from the Steam platform and build time-series models to forecast daily concurrent player demand, analyze promotional discount elasticity, and provide an interactive sales scheduling tool for indie game developers conforming to the hybrid **Prophet + LightGBM** architecture outlined in [program.pdf](file:///home/yur4uwe/Downloads/program%20%281%29-1.pdf#L16631).

---

## Technical Architecture

```mermaid
flowchart TD
    subgraph INGEST ["1. Data Ingestion (src/ingestion/)"]
        API_STEAM["Official Steam API & Store API\n(Live CCU, Base Price, Current Discount)"] --> RAW["Raw Ingestion Store\n(data/raw/*.jsonl)"]
        SC_CHART["SteamCharts Time-Series Scraper\n(Daily CCU & Peak series)"] --> RAW
        SC_HIST["SteamCharts Monthly Archive\n(Multi-year historical aggregates)"] --> RAW
        PROMO_CAL["Steam Sales Calendar Engine\n(Summer, Autumn, Winter, Spring, Weekend Deals)"] --> RAW
    end

    subgraph FEAT ["2. Feature Engineering (src/features/)"]
        RAW --> RESAMPLE["Daily Aggregation & Alignment\n(daily_avg, daily_peak, missing gap handling)"]
        RESAMPLE --> PROMO_FEAT["Promotional Features\n(discount_pct, is_steam_sale, days_in_sale, decay_factor)"]
        RESAMPLE --> CAL_FEAT["Calendar & Seasonality\n(day_of_week, is_weekend, month, cyclical sin/cos)"]
        RESAMPLE --> LAG_FEAT["Autoregressive Lags (Listing 5)\n(sales_lag_1, sales_lag_7, sales_lag_30)"]
        RESAMPLE --> ROLL_FEAT["Rolling Aggregations\n(sales_rolling_7, sales_rolling_30)"]
        PROMO_FEAT & CAL_FEAT & LAG_FEAT & ROLL_FEAT --> PARQUET["Processed Time-Series Parquet\n(data/processed/daily_features.parquet)"]
    end

    subgraph MODEL ["3. Modeling & Lineage (src/models/)"]
        PARQUET --> TIME_SPLIT["Temporal Holdout Split\n(Sequential validation; zero lookahead)"]
        TIME_SPLIT --> M1["M1: 7-Day Seasonal Persistence Baseline"]
        TIME_SPLIT --> M2["M2: Regularized Autoregressive Ridge Regression"]
        TIME_SPLIT --> M3["M3: LightGBM Regressor (Lags & Promotional Dynamics)"]
        TIME_SPLIT --> M4["M4: Hybrid Prophet + LightGBM Ensemble (Listing 5)"]
        M1 & M2 & M3 & M4 --> REGISTRY["Model Registry & SQLite Lineage\n(models/experiments.db, metadata.json)"]
        M4 --> EXPORT["Saved Model Artifact\n(models/indie_demand_model.joblib)"]
    end

    subgraph APP ["4. Streamlit Decision Studio (src/app/)"]
        EXPORT & REGISTRY --> DASH["Streamlit Interactive Studio\n- Multi-parameter game & campaign sidebar\n- KPI telemetry (Baseline CCU, Peak Surge, Lift, Cooldown)\n- Tab 1: Forecast & Schedule Visualizer\n- Tab 2: 'What-If' Discount Elasticity Simulator\n- Tab 3: Model Lineage & Diagnostics"]
        DASH --> DOCKER["Containerized Deployment (Dockerfile)"]
    end
```

---

## Running Instructions

### Local Execution with uv
```bash
# Sync dependencies (creates/updates environment automatically)
uv sync

# Ingest historical & real-time Steam data
uv run python -m src.ingestion.runner

# Build feature matrices
uv run python -m src.features.builder

# Train models and record lineage (M1 - M4)
uv run python -m src.models.train

# Option A: Decoupled Production Mode (Recommended - Zero UI Stutter)
# Terminal 1: Launch FastAPI forecasting service (Port 8000)
uv run python -m src.api.server

# Terminal 2: Launch Streamlit decision studio (Port 8501)
uv run streamlit run src/app/main.py

# Option B: Standalone Mode (In-Process Fallback)
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
2-steam-demand-forecast/
├── data/
│   ├── raw/                 # Raw API & scraper payloads
│   └── processed/           # Aligned daily time-series Parquet files
├── models/
│   ├── experiments.db       # SQLite model lineage tracker
│   ├── metadata.json        # Production model metadata & metrics
│   └── indie_demand_model.joblib
├── src/
│   ├── ingestion/           # Steam API, SteamCharts scraper, sales calendar
│   ├── features/            # Autoregressive lags, rolling windows, promo indicators
│   ├── models/              # Prophet baseline, LightGBM, Hybrid, Registry & Lineage
│   └── app/                 # Streamlit decision studio
├── tests/                   # Ingestion, feature integrity, and model unit tests
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Status & Roadmap

- [x] **Step 1: Environment & Project Setup** (`pyproject.toml`, `uv sync`)
- [x] **Step 2: Ingestion Pipeline Setup** (`steam_client.py`, `steamcharts.py`, `sales_calendar.py`)
- [x] **Step 3: Time-Series Feature Engineering** (`lags.py`, `rolling.py`, `promotions.py`, `builder.py`)
- [x] **Step 4: Modeling & Experiment Lineage** (`prophet_model.py`, `lgbm_model.py`, `hybrid.py`, `lineage.py`, `train.py`)
- [x] **Step 5: Streamlit Decision-Support Studio** (`main.py` with multi-parameter sidebar, 3 analytical tabs)
- [x] **Step 6: Unit Testing & Docker Packaging** (`tests/`, `Dockerfile`)
