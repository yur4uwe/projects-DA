# Second-Hand Deal Spotter & Market Classification

## Overview
A machine learning classification system designed to detect and categorize pricing attractiveness in secondary marketplaces. By analyzing product specifications, age, condition, and listed asking prices against prevailing market distributions, the model classifies listings into distinct deal tiers (such as *Great Deal*, *Fair Price*, and *Overpriced*).

Unlike direct regression models, this project frames deal spotting as an ensemble classification problem using Random Forest, evaluating multi-class decision boundaries, class probability calibration, and feature attribution across diverse item segments.

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
docker build -t secondhand-deal-spotter .

# Run container
docker run -p 8501:8501 secondhand-deal-spotter
```

---

## Directory Structure

```text
3-secondhand-deal-spotter/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── scraper/
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
- [ ] Marketplace listing ingestion & data source setup
- [ ] Spec extraction, cleaning & market baseline generation
- [ ] Deal tier classification labeling & feature engineering
- [ ] Random Forest classification pipeline & multi-class evaluation
- [ ] Interactive Streamlit deal explorer & listing evaluator
- [ ] Docker containerization & test coverage
