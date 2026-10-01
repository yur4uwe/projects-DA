# Second-Hand Market Valuation & Deal Spotter

## Objective
Scrape active marketplace listings, estimate fair market values using regression models, and identify statistical pricing anomalies to detect underpriced deals.

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
03-secondhand-deal-spotter/
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
- [ ] Marketplace scraper setup
- [ ] Data cleaning & categorical parsing
- [ ] Depreciation & attribute feature engineering
- [ ] Valuation model & residual scoring implementation
- [ ] Streamlit interface
- [ ] Docker packaging
