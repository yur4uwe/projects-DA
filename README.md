# Applied Data Analysis Portfolio

This repository contains three end-to-end data analysis and machine learning projects developed for the course referenced in [program.pdf](file:///home/yur4uwe/Uni/data-analisys/program.pdf).

---

## Projects Overview

| ID | Directory | Domain | Description | Status |
| :- | :--- | :--- | :--- | :--- |
| 01 | [01-tech-labor-market](file:///home/yur4uwe/Uni/data-analisys/01-tech-labor-market/README.md) | HR Tech / Labor Economics | Tech job scraping, salary modeling, and skill attribution. | In progress |
| 02 | [02-steam-demand-forecast](file:///home/yur4uwe/Uni/data-analisys/02-steam-demand-forecast/README.md) | Digital Media / Gaming Analytics | Steam player count time-series ingestion and demand forecasting. | In progress |
| 03 | [03-secondhand-deal-spotter](file:///home/yur4uwe/Uni/data-analisys/03-secondhand-deal-spotter/README.md) | E-commerce / Second-Hand Market | Marketplace listing valuation and pricing anomaly detection. | In progress |

---

## Repository Structure

```text
data-analisys/
├── 01-tech-labor-market/          # Project 1
│   ├── data/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
├── 02-steam-demand-forecast/      # Project 2
│   ├── data/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
├── 03-secondhand-deal-spotter/    # Project 3
│   ├── data/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── README.md
├── .gitignore
├── program.pdf
├── pyproject.toml
└── README.md
```

---

## Running Instructions

Each project uses `uv` with its own `pyproject.toml`.

### Project 1: Tech Labor Market

#### Local execution with uv
```bash
cd 01-tech-labor-market
uv sync
uv run streamlit run src/app/main.py
```

#### Containerized execution with Docker
```bash
cd 01-tech-labor-market
docker build -t tech-labor-market .
docker run -p 8501:8501 tech-labor-market
```

---

### Project 2: Steam Player Demand Forecast

#### Local execution with uv
```bash
cd 02-steam-demand-forecast
uv sync
uv run streamlit run src/app/main.py
```

#### Containerized execution with Docker
```bash
cd 02-steam-demand-forecast
docker build -t steam-demand-forecast .
docker run -p 8501:8501 steam-demand-forecast
```

---

### Project 3: Second-Hand Market Valuation

#### Local execution with uv
```bash
cd 03-secondhand-deal-spotter
uv sync
uv run streamlit run src/app/main.py
```

#### Containerized execution with Docker
```bash
cd 03-secondhand-deal-spotter
docker build -t secondhand-deal-spotter .
docker run -p 8501:8501 secondhand-deal-spotter
```
