# Issue 02: Analytical Architecture & Flat Promotional Demand Response Analysis

**Status**: Resolved  
**Components**: `src/model/hybrid.py`, `src/model/predict.py`, `src/features/builder.py`, `src/features/constants.py`, `src/app/main.py`, `src/app/server_api.py`  
**Date**: 2026-10-22  

---

## 1. Project Architecture Overview

The repository is structured as an end-to-end analytical and machine-learning pipeline for Steam indie game demand modeling:

```
2-steam-demand-forecast/
├── data/
│   ├── raw/                  # Catalog metadata (6 games) and raw JSONL records
│   └── processed/            # Feature matrices (.parquet, .csv)
├── src/
│   ├── ingestion/            # Steam API client, SteamCharts scraper, and SteamSalesCalendar
│   ├── features/             # Feature engineering pipeline
│   │   ├── cleaner.py        # Grid alignment & interpolation
│   │   ├── calendar.py       # Cyclical day-of-week, day-of-year, weekend indicators
│   │   ├── promotions.py     # Discount features, SALE_RAMP, storefront_exposure
│   │   ├── lags.py           # Autoregressive lags (1, 2, 7, 14, 30)
│   │   ├── rolling.py        # Rolling means/stds (3, 7, 14, 30), momentum, trend_3d
│   │   └── builder.py        # Master FeaturePipeline compiling daily_features.parquet
│   ├── model/                # Forecasting engines & milestone models
│   │   ├── baseline.py       # M1: Naive 7-day seasonal persistence
│   │   ├── ridge_model.py    # M2: Standardized L2-regularized autoregressive Ridge
│   │   ├── lgbm_model.py     # M3: Gradient-boosted decision trees (LightGBM)
│   │   ├── hybrid.py         # M4: Prophet macro baseline + LightGBM residual model
│   │   ├── predict.py        # Multi-step recursive simulation engine
│   │   └── train.py          # Temporal holdout training & SQLite model registry
│   ├── api/                  # FastAPI backend (endpoints for forecast, elasticity, health)
│   └── app/                  # Streamlit frontend studio (main.py, server_api.py)
├── docs/issues/              # Issue and architectural investigation documents
└── tests/                    # 14 unit and integration tests (pytest)
```

---

## 2. Issue A: Timeouts Across Frontend Tabs (Resolved)

### Symptoms
When scheduling promotions in future months (e.g. November 20 or December 17), the Streamlit dashboard experienced request timeouts (`httpx.ReadTimeout`), causing connection errors and disabling tabs.

### Root Causes
1. **Unbatched Prophet Inferences**:
   In `src/model/predict.py`, `simulate_promotional_campaign` called Prophet row-by-row inside the simulation loop for every day in the horizon. A 46-day simulation invoked Prophet 46–92 times on single-row DataFrames, taking ~3.7 seconds per simulation.
2. **Elasticity Matrix Multiplier**:
   Tab 2 (`/elasticity`) runs 6 separate simulations across `[0.10, 0.20, 0.33, 0.50, 0.66, 0.75]`. Total execution time reached $6 \times 4.5\text{s} = 27\text{s}$ to $40\text{s}$, exceeding the 30.0s HTTP client timeout.
3. **Cascading Health Check Failures**:
   The backend worker was tied up in CPU-bound Prophet inference, causing `query_api_health` (timeout 1.5s) to fail and trigger the "Prediction Service Offline" screen.
4. **Uncached Streamlit Execution**:
   Because Streamlit runs all tabs on every rerun, `/elasticity` was executed on every interaction without caching.

### Implemented Fixes
1. **Prophet Batch Precomputation** (`src/model/predict.py`):
   Precomputed Prophet's static calendar baseline across all `future_dates` in a single vectorized batch call prior to the recursive simulation loop.
   - Batch evaluation time: **0.054 seconds** (down from 3.71s, a ~68x speedup).
2. **Client-Side Streamlit Caching & Timeout Cushion** (`src/app/server_api.py`):
   - Added `@st.cache_data(ttl=120)` to `execute_elasticity_simulation` and `execute_forecast_simulation`.
   - Raised health check timeout to 5.0s and simulation timeout to 60.0s.

---

## 3. Issue B: Promotional Demand "Almost Always Flat"

### Observed Behavior
Across the catalog, promotional discount events often produce minimal or visually flat demand responses in forward simulations:
- **Dead Cells** (Base: 2,548 CCU): +142% surge (+3,766 CCU)
- **Hollow Knight** (Base: 6,251 CCU): +51% surge (+3,335 CCU)
- **Slay the Spire** (Base: 6,566 CCU): +39% surge (+2,611 CCU)
- **Terraria** (Base: 33,168 CCU): **+11% surge (+3,747 CCU) — Visually Flat**
- **Stardew Valley** (Base: 53,030 CCU): **+19% surge (+10,135 CCU) — Subdued**

### Technical Root Causes

#### 1. Unscaled Native Residuals in a Cross-Game GBDT
In `HybridProphetLGBMModel` (`src/model/hybrid.py`):
- Prophet trains on normalized scale $y / \text{rolling\_30}$.
- However, LightGBM is trained on **unscaled native CCU residuals**:
  $$\text{residual} = y - \hat{y}_{\text{Prophet}}$$
- The catalog spans games from 2,500 CCU to 53,000 CCU ($21\times$ difference).
- A tree split on `is_steam_sale == 1` and `discount_depth` outputs an average additive CCU number (around $+3,000$ to $+4,000$ CCU).
- For a 2,500 CCU game, $+3,500$ CCU is a $+140\%$ surge.
- For a 33,000 CCU game (Terraria), $+3,500$ CCU is an 11% bump. On a 0–45,000 CCU chart axis, an 11% bump looks completely flat.

#### 2. Autoregressive Lag Dominance
Feature importance analysis shows:
- `sales_lag_1`: Gain = **185,736**
- `is_steam_sale`: Gain = **96**
- `discount_depth`: Gain = **20**
When a promotion starts, `sales_lag_1` is still at the pre-sale baseline. Because LightGBM relies predominantly on `sales_lag_1` for split decisions, it outputs predictions close to baseline until elevated demand feeds back into lags.

#### 3. Complete Transfer of Promo Lift to LightGBM
Because `promotion = 0.0` was set in Prophet inference to eliminate double-counting, Prophet contributes 0 CCU of promotional lift. If LightGBM outputs a conservative residual, there is no underlying macro multiplier to scale the demand proportionally to the game's baseline.

---

## 4. Recommended Architectural Solutions for Issue B

1. **Multiplicative / Percentage Residual Modeling**:
   Train LightGBM on percentage residual or log residual rather than raw CCU difference:
   $$\text{residual\_ratio} = \frac{y_{\text{true}}}{\max(\hat{y}_{\text{Prophet}}, 10.0)}$$
   $$\hat{y} = \hat{y}_{\text{Prophet}} \times \hat{r}_{\text{LGBM}}$$
   This guarantees that a 50% discount predicts a proportional $+50\%$ surge on Terraria (+16,500 CCU) just as it does on Hollow Knight (+3,150 CCU).

2. **Game-Scaled Interaction Terms**:
   Add `base_scale_x_discount = sales_rolling_30 * discount_depth * storefront_exposure` directly into LightGBM feature columns. This gives the tree an explicit signal of expected volume proportional to the game's CCU tier.
