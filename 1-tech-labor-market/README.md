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
- [x] Scraping pipeline setup (`requests`, JSON-LD / REST APIs, Djinni + RemoteOK)
- [x] Data cleaning & text normalization (salary conversion, CEFR English, experience parsing)
- [x] Feature extraction and encoding (taxonomy mapping, composite indicators, binary flags)
- [x] Model training & baseline evaluation (Dummy median vs. Ridge vs. LightGBM L1, TreeSHAP)
- [x] Streamlit interface (salary prediction, local SHAP waterfall, upskilling simulator)
- [x] Docker packaging (`Dockerfile` with `uv`)

---

## Future Plans

- **Geography Arbitrage:** Encode US vs. EU vs. domestic contract flags to capture regional compensation differentials.
- **Outlier Trimming:** Filter extreme top 2.5% contracts (>$10k/mo) to eliminate residual skew on standard market roles.
- **Company Domain Signals:** Encode sector tags (Fintech, Crypto, Gaming, Outstaff) to capture domain wage premiums.
- **Skill Synergy Interactions:** Add cross-product composite flags (`Cloud + DevOps`, `Go + Kafka`, `React + Node`) for tech stack bundle premiums.
- **TF-IDF / Text Embeddings:** Extract TruncatedSVD components from job descriptions to capture architectural responsibility beyond regex tokens.
- **Temporal Scraping Automation:** Schedule recurring ingestion runs to accumulate 1,000+ records and track salary inflation over time.

