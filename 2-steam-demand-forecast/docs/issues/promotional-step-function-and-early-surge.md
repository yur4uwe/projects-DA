# Issue 01: Promotional Step-Function Discontinuity and Day 1 Surge Overestimation

**Status**: Resolved  
**Components**: `src/features/promotions.py`, `src/features/builder.py`, `src/model/lgbm_model.py`, `src/model/hybrid.py`, `src/model/predict.py`  
**Date**: 2026-10-22  

---

## 1. Problem Description

When simulating promotional campaigns in `simulate_promotional_campaign`, simulated concurrent player demand (CCU) exhibited an abrupt, instantaneous step increase on Day 0:

For Hollow Knight (AppID 367520, baseline CCU ~6,300):
- **Pre-Sale Baseline (Day -1)**: 6,397 CCU
- **Sale Day 0 (Day 1 of discount)**: 14,058 CCU (+120% / +7,661 CCU overnight jump)
- **Sale Day 1**: 12,272 CCU

In reality, Steam concurrent players (24-hour daily average CCU) do not double instantaneously on Day 0. Sales launch at 10:00 AM PST (19:00 UTC), meaning the first 10–14 hours of Day 0 are at normal unpromoted volume. Additionally, players experience latency: wishlist notifications are queued across hours, games require download and installation, and player sessions concentrate into evening hours.

---

## 2. Root Cause Analysis

Decomposition of Day 0 prediction components (`model.predict_components`) revealed three compounding root causes:

### Root Cause 1: Synthetic Training Data Imposed an Instantaneous Step Function
Over 99% of promotional training samples originate from historical monthly archives linearly interpolated into daily series. To inject promotional elasticity into these flat spans, `src/features/builder.py` used a uniform boolean mask multiplying Day 0 demand by full elasticity with no rollout ramp. Both Prophet and LightGBM were trained on targets with vertical step-function edges.

### Root Cause 2: Double-Counting Promotional Uplift Across the Hybrid Ensemble
In `src/model/hybrid.py`, promotional lift was estimated simultaneously by two distinct models:
1. **Prophet Macro Baseline**:
   - Used linear regressors `promotion` and `holiday`.
   - On Day 0, Prophet baseline jumped from **6,560 CCU** to **9,612 CCU** (+3,052 CCU / +46.5%).
2. **LightGBM Residual Regressor**:
   - Used features `discount_depth`, `is_steam_sale`, and `promo_x_discount`.
   - On Day 0, LightGBM predicted a positive residual of **+4,446 CCU** (compared to -162 CCU organically).
3. **Combined Output**:
   $$\hat{y}_{\text{promo}} = 9,612 + 4,446 = 14,058 \text{ CCU}$$
Both models independently attributed demand expansion to the promotional indicator, producing a combined +120% jump.

---

## 3. Resolution Implemented

1. **Storefront Exposure Feature & Realistic Target Ramp** (`src/features/promotions.py`, `src/features/builder.py`):
   - Defined `SALE_RAMP = {0: 0.35, 1: 0.85}` (default 1.0).
   - In `builder.py`, scaled historical interpolated promotional targets:
     $$\text{demand} \times (1.0 + \text{elasticity\_lift} \times \text{weekend\_boost} \times \text{storefront\_exposure})$$
   - Added `storefront_exposure` to `DEFAULT_FEATURES` in `src/model/lgbm_model.py`.
2. **Decoupled Counterfactual Baseline in Prophet** (`src/model/hybrid.py`):
   - Retained `promotion` and `holiday` in `self.prophet_model.fit()` to prevent annual Fourier seasonality from absorbing sale spikes.
   - Evaluated Prophet with `promotion = 0.0` and `holiday = 0.0` in both residual calculation and `predict_components()`.
   - `prophet_base` is now guaranteed strictly organic, with 100% of promotional uplift modeled by LightGBM.
3. **Simulation Symmetry** (`src/model/predict.py`):
   - `day_org["storefront_exposure"] = 0.0`
   - `day_promo["storefront_exposure"] = SALE_RAMP.get(days_from_start, 1.0)`

---

## 4. Verification & Results

- **Model Benchmarks (30-day Temporal Holdout)**:
  - M4 Hybrid MAE: **2526.22 CCU** (down from 3980.16 CCU, -36.5% error reduction)
  - M4 Hybrid RMSE: **4341.18 CCU** (down from 6913.41 CCU)
  - M4 Hybrid WAPE: **14.65%** (down from 23.07%)
  - M4 Hybrid $R^2$: **0.9451** (up from 0.8609)
- **Hollow Knight Campaign Trajectory (Oct 25, 50% discount)**:
  - Day 0 (First 24h): Smooth launch ramp to 9,824 CCU (no 14,000+ vertical spike).
  - Intra-sale: Stable plateau ~8,000–9,500 CCU with natural weekend elevation.
  - Final day: Urgency bump to 7,932 CCU.
  - Post-sale: Monotonic decay to 6,296 CCU organic baseline.
- **Tests**: 14 of 14 unit and integration tests passing.
