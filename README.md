# Applied Data Analysis & Machine Learning Portfolio

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated collection of three end-to-end (E2E) data analysis and machine learning systems spanning distinct domains. Built with realistic data acquisition pipelines (custom web scraping and public API ingestion), tree-based predictive models (`LightGBM`, `XGBoost`), time-series forecasting (`Prophet`), model explainability (`SHAP`), and containerized interactive dashboards (`Streamlit` + `Docker`).

---

## 📁 Projects Overview

| # | Project Name | Domain | Data Sourcing | Core ML & Analytics | Serving & UI | Spec Link |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| **01** | **Tech Labor Market & Compensation Intelligence** | HR Tech / Labor Economics | Web scraping tech boards (Djinni, RemoteOK) | `LightGBM`, `XGBoost`, `SHAP` values, Skill Extraction NLP | Streamlit Dashboard | [View Spec](file:///home/yur4uwe/Uni/data-analisys/01-tech-labor-market/README.md) |
| **02** | **Steam Player Demand & Trend Forecasting** | Digital Media / Gaming Analytics | Steam Web API & SteamCharts / SteamSpy historical data | Hybrid `Prophet` + `LightGBM` residual modeling, Lag features | Streamlit Scenario Simulator | [View Spec](file:///home/yur4uwe/Uni/data-analisys/02-steam-demand-forecast/README.md) |
| **03** | **Second-Hand Market Valuation & Deal Spotter** | E-Commerce / Automotive & Hardware | Web scraping marketplace listings (AutoRia / OLX) | `LightGBM` Regressor, Residual & Statistical Anomaly Detection | Streamlit Bargain Radar | [View Spec](file:///home/yur4uwe/Uni/data-analisys/03-secondhand-deal-spotter/README.md) |

---

## 🏛️ Repository Architecture

This repository is organized as a **modular monorepo**: each project is completely autonomous with its own data pipelines, source modules, tests, dependencies, and `Dockerfile`.

```text
data-analisys/
├── 01-tech-labor-market/          # Project 1: Salary prediction & skill ROI
│   ├── data/                      # Raw & processed datasets
│   ├── src/                       # Scraper, features, model training & Streamlit UI
│   ├── Dockerfile                 # Standalone Docker container
│   ├── requirements.txt           # Project-specific dependencies
│   └── README.md                  # Project 1 documentation
│
├── 02-steam-demand-forecast/      # Project 2: Player count time-series forecast
│   ├── data/                      # Time-series snapshots
│   ├── src/                       # Ingestion, Prophet baseline, LightGBM residuals & UI
│   ├── Dockerfile                 # Standalone Docker container
│   ├── requirements.txt           # Project-specific dependencies
│   └── README.md                  # Project 2 documentation
│
├── 03-secondhand-deal-spotter/    # Project 3: Fair market value & deal anomaly detector
│   ├── data/                      # Marketplace listings
│   ├── src/                       # Scraper, depreciation modeling, residual deal scoring & UI
│   ├── Dockerfile                 # Standalone Docker container
│   ├── requirements.txt           # Project-specific dependencies
│   └── README.md                  # Project 3 documentation
│
├── .gitignore                     # Monorepo data & artifact exclusion rules
├── program.pdf                    # Academic course syllabus
└── README.md                      # Master portfolio index (this file)
```

---

## 🚀 Quickstart & Execution

Each project can be operated and run independently.

### Running with Docker (Recommended)
Navigate into any project directory and run:

```bash
# Build the project image
docker build -t <project-tag> .

# Launch the interactive Streamlit service
docker run -p 8501:8501 <project-tag>
```

### Running Locally with Python
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies for the specific project
pip install -r <project-folder>/requirements.txt

# Run the project dashboard
streamlit run <project-folder>/src/app/main.py
```

---

## 🎓 Academic Course Reference
This work is developed as part of the curriculum: **"Проєкт з розробки системи аналізу даних"** (Specialty 122 "Computer Science", lecturer: B.M. Pavlyshenko). See [program.pdf](file:///home/yur4uwe/Uni/data-analisys/program.pdf) for the course syllabus.
