# Tech Labor Market & Compensation Intelligence

## Objective
Collect tech job postings via web scraping, model salary ranges using tree-based regressors, and evaluate skill attribution.

---

## Running Instructions

### Local Execution with uv
```bash
# Sync dependencies (creates/updates environment automatically)
uv sync

# Run the Streamlit dashboard
uv run streamlit run src/app/main.py
```

### Containerized Execution with Docker
```bash
# Build Docker image
docker build -t tech-labor-market .

# Run container
docker run -p 8501:8501 tech-labor-market
```

---

## Directory Structure

```text
01-tech-labor-market/
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
├── requirements.txt
└── README.md
```

---

## Status & Roadmap
- [ ] Scraping pipeline setup
- [ ] Data cleaning & text normalization
- [ ] Feature extraction and encoding
- [ ] Model training & baseline evaluation
- [ ] Streamlit interface
- [ ] Docker packaging
