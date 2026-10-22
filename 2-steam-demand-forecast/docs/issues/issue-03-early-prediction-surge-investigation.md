# Issue 03: Early Prediction CCU Surge Investigation and Resolution

**Status**: Resolved  
**Components**: `src/model/predict.py`, `tests/test_models.py`  
**Related**: `docs/issues/promotional-step-function-and-early-surge.md`, `docs/issues/issue-02-flat-sales-and-project-structure.md`  

---

## 1. Problem Statement

At the start of forward simulation trajectories, predicted demand consistently surged by +1,000 to +3,000 CCU (and up to +20,000 CCU on higher-scale titles such as *Stardew Valley*) during the first 2 to 5 days (e.g. October 19–24), before decaying back to baseline.

This transient dome occurred even when:
- `discount_depth = 0.0` (no sale scheduled), or
- The promotional campaign was scheduled far in the future (e.g., November 1), where the first two weeks were supposed to represent pure organic demand.

Because this artificial surge occurred early in the simulation, it often matched or exceeded the actual promotional lift of scheduled sales, creating confusing and distorted forecast trajectories.

---

## 2. Root Cause Analysis

Investigation revealed two compounding mechanisms:

### 2.1 Recursive Autoregressive Feedback Loop in Forward Simulation (`predict.py`)
In multi-step simulation, the organic counterfactual baseline was evaluated recursively:
```python
p_base = precomputed_p_bases[idx]
X_org = model._prepare_organic_features(day_org, base_rolling_30)
r_org = float(model.lgb_organic_model.predict(X_org)[0])
damped_r = r_org * np.exp(-idx / 14.0)
org_val = max(0.0, p_base * np.exp(damped_r))

new_org_row["demand"] = org_val
org_buffer = pd.concat([org_buffer, new_org_row], ignore_index=True)
```

1. **Boundary State**: On Day 0 (Monday, Oct 19), `sales_lag_1` and `sales_lag_2` inherited the trailing weekend CCU (Oct 17–18). Because Saturday/Sunday demand is naturally elevated, `norm_lag_1` was > 1.0.
2. **Teacher-Forcing Mismatch**: `lgb_organic_model` was trained on 1-step historical targets. Seeing `norm_lag_1 > 1.0`, the trees predicted a positive log ratio $r_{\text{org}} > 0$ (+14% to +20%).
3. **Autoregressive Compounding**: `org_val` was fed back into `org_buffer` as `sales_lag_1` for Day 1. Meanwhile, `sales_rolling_30` remained frozen at historical baseline (`base_rolling_30`). LightGBM saw an even larger normalized lag on Day 1 and predicted an even larger residual ($r_{\text{org}} = +0.35$).
4. **Insufficient Damping**: Because $r_{\text{org}}$ grew faster than $\exp(-idx / 14.0)$ decayed over the first 4 days, the product $r_{\text{org}} \cdot \exp(-idx / 14.0)$ inflated from 0.14 on Day 0 to 0.48 on Day 3, creating a massive artificial dome before eventually collapsing.

### 2.2 Ungated Promotional Uplift at Zero Discount
In `predict.py`, the condition for entering promotional calculation was:
```python
in_promo_window = promo_start <= f_date <= promo_end
```
When `discount_depth = 0.0`, `in_promo_window` still evaluated to `True`. The Ridge uplift model received features such as `storefront_exposure = 1.0` and `is_major_seasonal_sale = 1`. Because the model had positive coefficients on exposure and sale type, it predicted an uplift of +30.9% on zero-discount rows, followed by a post-sale decay tail.

---

## 3. Implemented Fix

### 3.1 Asymptotic Boundary Residual Decay
Forward simulation now uses an asymptotic impulse response formulation:
- At the simulation cutoff (the boundary between observed history and forecast), the initial residual $r_0$ between the last known actual demand $y_T$ and Prophet's baseline $P_T$ is measured directly:
  $$r_0 = \text{clip}\left(\log\frac{y_T}{P_T}, -0.5, 0.5\right)$$
- Over future simulation days $h \ge 0$, this initial discrepancy decays toward zero with an autocorrelation half-life $\tau = 3.5$ days:
  $$r(h) = r_0 \cdot \exp\left(-\frac{h}{\tau}\right)$$
- The organic forecast is evaluated as:
  $$\hat{y}_{\text{org}}(h) = P(h) \cdot \exp(r(h))$$

**Mathematical Properties**:
1. **$C^0$ Continuity**: At $h = 0$, $\hat{y}_{\text{org}}(0) = P_0 \cdot \frac{y_0}{P_0} = y_0$, eliminating artificial jumps between historical data and the forecast origin.
2. **No Autoregressive Compounding**: The residual path is strictly monotonic and bounded by $|r_0|$, completely eliminating recursive positive feedback loops.
3. **Convergence to Macro Seasonality**: By day 7–10, $\exp(-h/\tau) \to 0$, smoothly handing full control to Prophet's macro trend, day-of-week, and day-of-year seasonalities.

### 3.2 Strict Promotional Gating
Promotional uplift and post-sale decay tails are now gated on non-zero discounts:
```python
is_discounted = discount_depth > 0.001
if in_promo_window and is_discounted:
    ...
elif f_date > promo_end and is_discounted:
    ...
else:
    pred_val = org_val
```
Zero-discount simulations are guaranteed to produce pure organic forecasts.

---

## 4. Verification and Empirical Comparison

### 4.1 First 5 Forecast Days Comparison (Oct 19–23, 2026, Unpromoted)

| Title | Historical Endpoint (Oct 18) | Baseline Mean (30d) | Old Forecast Day 0–3 | New Forecast Day 0–3 | Result |
|---|---|---|---|---|---|
| **Hollow Knight** | 6,780 CCU | 6,391 CCU | 7,464 $\to$ 7,810 $\to$ 7,294 | 6,761 $\to$ 6,691 $\to$ 6,630 | Smooth decay to baseline |
| **Terraria** | 37,396 CCU | 32,755 CCU | 37,173 $\to$ 37,482 $\to$ 34,247 | 37,293 $\to$ 36,238 $\to$ 35,421 | Continuous downward transition |
| **Stardew Valley** | 59,257 CCU | 50,944 CCU | 64,675 $\to$ 71,325 $\to$ 71,199 | 59,093 $\to$ 57,156 $\to$ 55,673 | Eliminated +20k CCU transient |
| **Slay the Spire** | 6,892 CCU | 6,612 CCU | 7,822 $\to$ 9,385 $\to$ 10,818 | 6,873 $\to$ 6,831 $\to$ 6,792 | Eliminated +4k CCU transient |

### 4.2 Test Suite
- Added regression test `test_forecast_no_early_organic_surge` in `tests/test_models.py`.
- Full pytest suite passing (17 of 17 tests passed).
