# Second-Hand Deal Spotter & Market Classification

## Overview
A machine learning classification system designed to detect and categorize pricing attractiveness in secondary marketplaces. By analyzing product hardware specifications, form factor, and listed asking prices against prevailing market distributions, the model classifies listings into distinct deal tiers (*Great Deal*, *Fair Price*, and *Overpriced*).

Unlike direct regression models, this project frames deal spotting as an ensemble classification problem using Random Forest, evaluating multi-class decision boundaries, class probability calibration, and feature attribution across diverse item segments.

---

## Running Instructions

The dashboard uses a FastAPI backend. Training requires the Python source in
`src/models/` and a local copy of the input dataset. If your checkout does not
contain `src/models/`, obtain that source before training or starting the API.

### Prepare the data and models

Put the laptop listings CSV in `data/raw/laptop_price.csv`, then run these
commands from `3-secondhand-deal-spotter/`:
```bash
# Sync dependencies (creates/updates environment automatically)
uv sync

# Ingest and verify raw marketplace dataset
uv run python -m src.scraper.main

# Build cleaned features and specification baselines
uv run python -m src.features.builder

# Train and benchmark Random Forest classifiers against baselines
uv run python -m src.models.train

# Run automated tests
uv run pytest tests/ -v
```

### Local execution

Start the API in one terminal and the dashboard in another:

```bash
uv run uvicorn src.api.main:app --port 8000
uv run streamlit run src/app/main.py
```

To add new dependencies to this project:
```bash
uv add <package-name>
```

### Containerized Execution with Docker
```bash
docker compose up --build
```

Open the dashboard at <http://localhost:8501>. The API health endpoint is
<http://localhost:8000/health>. Compose mounts the prepared `data/` and
`models/` directories into the backend.

---

## Directory Structure

```text
3-secondhand-deal-spotter/
├── data/
│   ├── raw/
│   │   └── laptop_price.csv
│   └── processed/
│       ├── pc_listings_processed.csv
│       └── pc_listings_processed.parquet
├── models/
│   ├── baseline_regressor.joblib
│   ├── deal_classifier.joblib
│   ├── global_feature_importance.json
│   └── metadata.json
├── src/
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── storage.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── builder.py
│   │   ├── cleaner.py
│   │   └── schema.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── pipeline.py
│   │   ├── predict.py
│   │   └── train.py
│   └── app/
│       └── main.py
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_scraper.py
├── compose.yml
├── pyproject.toml
└── README.md
```

---

## Model Evaluation Summary (5-Fold Stratified Cross-Validation)

| Model Architecture | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Log Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy (Most Frequent)** | 0.5741 | 0.3333 | 0.2431 | 0.4187 | 15.3524 |
| **Multinomial Logistic Regression** | 0.9417 | 0.9388 | 0.9394 | 0.9420 | 0.1914 |
| **Random Forest (Uncalibrated)** | 0.9194 | 0.9113 | 0.9135 | 0.9194 | 0.3909 |
| **Calibrated Random Forest (Production)** | 0.9233 | 0.9159 | 0.9180 | 0.9233 | 0.2132 |

---

## Status & Roadmap
- [x] Marketplace listing ingestion & data source setup
- [x] Spec extraction, cleaning & market baseline generation
- [x] Deal tier classification labeling & feature engineering
- [x] Random Forest classification pipeline & multi-class evaluation
- [x] Interactive Streamlit deal explorer & listing evaluator
- [x] Docker containerization & test coverage
