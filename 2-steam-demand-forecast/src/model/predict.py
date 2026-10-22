"""Multi-step recursive forward forecasting and what-if promotional simulation."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.features.calendar import create_calendar_features
from src.features.constants import (
    SALE_RAMP,
    TAIL_DECAY_RATE,
    TAIL_INITIAL_FACTOR,
)
from src.features.promotions import create_promotional_features
from src.ingestion.sales_calendar import SteamSalesCalendar


def simulate_promotional_campaign(
    history_df: pd.DataFrame,
    model: Any,
    horizon_days: int = 14,
    start_date: date | None = None,
    discount_depth: float = 0.33,
    sale_duration_days: int = 7,
    is_custom_event: bool = False,
    custom_event_name: str = "Custom Publisher Sale",
    calendar: SteamSalesCalendar | None = None,
) -> dict[str, Any]:
    """Simulate future daily demand and evaluate promotional uplift under scheduled sales."""
    cal = calendar or SteamSalesCalendar()
    hist = history_df.sort_values("date").copy()
    last_date = pd.to_datetime(hist["date"].iloc[-1]).date()

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()

    # The simulation timeline always starts immediately after historical data (unbroken continuity)
    sim_start = last_date + timedelta(days=1)
    promo_start = start_date if start_date and start_date >= sim_start else sim_start
    promo_end = promo_start + timedelta(days=sale_duration_days - 1)

    # When the sale is scheduled in the future, span through the lead-up gap plus horizon_days
    if promo_start > sim_start:
        lead_up_days = (promo_start - sim_start).days
        total_horizon = lead_up_days + horizon_days
    else:
        total_horizon = horizon_days

    future_dates = [sim_start + timedelta(days=i) for i in range(total_horizon)]

    # Establish recent baseline demand and historical rolling statistics
    recent_baseline = float(hist["demand"].iloc[-14:].mean())
    base_rolling_30 = (
        float(hist["demand"].iloc[-30:].mean())
        if len(hist) >= 30
        else recent_baseline
    )
    base_rolling_std_30 = (
        float(hist["demand"].iloc[-30:].std())
        if len(hist) >= 30
        else float(hist["demand"].std())
    )
    base_p = float(hist.get("base_price", pd.Series([14.99])).iloc[-1])

    forecast_rows: list[dict[str, Any]] = []

    # Maintain organic simulation buffer (counterfactual baseline without promotion)
    # This prevents promotional shocks from exponentially compounding across autoregressive lags
    org_buffer = hist.copy()
    peak_promotional_lift = 0.0

    # Precompute Prophet macro organic baseline across all future dates in a single batch call (~0.05s)
    has_hybrid = hasattr(model, "prophet_model") and (
        hasattr(model, "lgb_residual_model")
        or hasattr(model, "lgb_organic_model")
        or hasattr(model, "uplift_model")
    )
    if has_hybrid:
        batch_pdf = pd.DataFrame()
        batch_pdf["ds"] = [pd.to_datetime(d).tz_localize(None) for d in future_dates]
        batch_pdf["promotion"] = 0.0
        batch_pdf["holiday"] = 0.0
        batch_prophet = model.prophet_model.predict(batch_pdf)
        norm_bases = np.clip(np.exp(batch_prophet["yhat"].values), 0.0, None)
        precomputed_p_bases = norm_bases * base_rolling_30
        precomputed_p_lowers = (
            np.clip(np.exp(batch_prophet["yhat_lower"].values), 0.0, None) * base_rolling_30
        )
        precomputed_p_uppers = (
            np.clip(np.exp(batch_prophet["yhat_upper"].values), 0.0, None) * base_rolling_30
        )

        # Compute initial boundary residual at simulation cutoff (last historical day)
        last_demand = float(hist["demand"].iloc[-1])
        last_pdf = pd.DataFrame(
            {"ds": [pd.to_datetime(last_date).tz_localize(None)], "promotion": 0.0, "holiday": 0.0}
        )
        last_prophet = model.prophet_model.predict(last_pdf)
        last_p_base = float(
            np.clip(np.exp(last_prophet["yhat"].iloc[0]), 0.0, None) * base_rolling_30
        )
        boundary_log_res = float(
            np.clip(np.log(max(last_demand, 1.0) / max(last_p_base, 1.0)), -0.5, 0.5)
        )
        boundary_add_res = float(
            np.clip(last_demand - last_p_base, -last_p_base * 0.5, last_p_base * 0.5)
        )
    else:
        precomputed_p_bases = None
        precomputed_p_lowers = None
        precomputed_p_uppers = None
        boundary_log_res = 0.0
        boundary_add_res = 0.0

    for idx, f_date in enumerate(future_dates):
        # Is the promo active on this day?
        in_promo_window = (promo_start <= f_date <= promo_end) and (discount_depth > 0.001)
        days_from_start = (f_date - promo_start).days

        # Base calendar features
        day_row = pd.DataFrame([{"date": pd.to_datetime(f_date)}])
        day_with_cal = create_calendar_features(day_row, date_col="date")

        # ---------------------------------------------------------------------
        # 1. Counterfactual Organic Baseline Evaluation (promo = 0)
        # ---------------------------------------------------------------------
        day_org = create_promotional_features(
            day_with_cal,
            date_col="date",
            calendar=cal,
            base_price=base_p,
            default_discount_depth=0.0,
        )
        day_org["is_steam_sale"] = 0
        day_org["discount_depth"] = 0.0
        day_org["promo_x_discount"] = 0.0
        day_org["promo_x_weekend"] = 0
        day_org["sale_day_index"] = 0

        # Anchor macro scale and long rolling statistics to historical unpromoted baseline
        day_org["sales_rolling_30"] = base_rolling_30
        day_org["sales_rolling_std_30"] = base_rolling_std_30

        # Organic autoregressive lags derived from unpromoted organic buffer
        org_demands = org_buffer["demand"].values
        day_org["sales_lag_1"] = org_demands[-1]
        day_org["sales_lag_2"] = (
            org_demands[-2] if len(org_demands) >= 2 else org_demands[-1]
        )
        day_org["sales_lag_7"] = (
            org_demands[-7] if len(org_demands) >= 7 else org_demands[-1]
        )
        day_org["sales_lag_14"] = (
            org_demands[-14] if len(org_demands) >= 14 else org_demands[-1]
        )
        day_org["sales_lag_30"] = (
            org_demands[-30] if len(org_demands) >= 30 else org_demands[-1]
        )

        day_org["sales_rolling_3"] = float(np.mean(org_demands[-3:]))
        day_org["sales_rolling_std_3"] = float(np.std(org_demands[-3:]))
        day_org["sales_rolling_7"] = float(np.mean(org_demands[-7:]))
        day_org["sales_rolling_std_7"] = float(np.std(org_demands[-7:]))
        day_org["sales_rolling_14"] = float(np.mean(org_demands[-14:]))
        day_org["sales_rolling_std_14"] = float(np.std(org_demands[-14:]))
        day_org["sales_momentum_7_30"] = float(
            day_org["sales_rolling_7"].iloc[0] / max(base_rolling_30, 1.0)
        )
        day_org["sales_momentum_3_7"] = float(
            np.clip(
                day_org["sales_rolling_3"].iloc[0] / max(day_org["sales_rolling_7"].iloc[0], 1.0),
                0.5,
                2.0,
            )
        )
        prev_3d = org_demands[-4] if len(org_demands) >= 4 else org_demands[0]
        denom_trend = max(3.0 * day_org["sales_rolling_7"].iloc[0], 1.0)
        day_org["sales_trend_3d"] = float(
            np.clip((org_demands[-1] - prev_3d) / denom_trend, -1.0, 1.0)
        )
        day_org["sale_progress"] = 0.0
        day_org["days_until_sale_end"] = -1
        day_org["storefront_exposure"] = 0.0

        for static_col in [
            "appid",
            "game_name",
            "base_price",
            "is_multiplayer",
            "is_coop",
            "log_positive_reviews",
        ]:
            if static_col in hist.columns:
                day_org[static_col] = hist[static_col].iloc[-1]

        # Predict organic counterfactual baseline with asymptotic boundary residual decay
        # Transient shocks at the historical cutoff decay smoothly with half-life tau = 3.5 days,
        # preventing recursive autoregressive positive feedback loops from inflating early days.
        if hasattr(model, "uplift_model") or hasattr(model, "lgb_organic_model"):
            p_base = float(precomputed_p_bases[idx])
            damped_r = boundary_log_res * np.exp(-idx / 3.5)
            org_val = max(0.0, p_base * np.exp(damped_r))
            org_lower = max(0.0, float(precomputed_p_lowers[idx]) * np.exp(damped_r)) if precomputed_p_lowers is not None else org_val * 0.85
            org_upper = max(0.0, float(precomputed_p_uppers[idx]) * np.exp(damped_r)) if precomputed_p_uppers is not None else org_val * 1.15
        elif has_hybrid:
            p_base = float(precomputed_p_bases[idx])
            damped_res = boundary_add_res * np.exp(-idx / 3.5)
            org_val = max(0.0, p_base + damped_res)
            org_lower = max(0.0, float(precomputed_p_lowers[idx]) + damped_res * 0.8)
            org_upper = max(0.0, float(precomputed_p_uppers[idx]) + damped_res * 1.2)
        elif hasattr(model, "predict_components"):
            org_comps = model.predict_components(day_org)
            p_base = float(org_comps["prophet_base"][0])
            damped_res = boundary_add_res * np.exp(-idx / 3.5)
            org_val = max(0.0, p_base + damped_res)
            org_lower = float(org_comps["yhat_lower"][0])
            org_upper = float(org_comps["yhat_upper"][0])
        else:
            org_preds = model.predict(day_org)
            org_val = float(org_preds[0])
            org_lower = float(max(0.0, org_val * 0.85))
            org_upper = float(org_val * 1.15)

        # Advance organic counterfactual buffer
        new_org_row = day_org.copy()
        new_org_row["demand"] = org_val
        org_buffer = pd.concat([org_buffer, new_org_row], ignore_index=True)

        # ---------------------------------------------------------------------
        # 2. Promotional Uplift or Post-Sale Decay Evaluation
        # ---------------------------------------------------------------------
        is_discounted = discount_depth > 0.001

        if in_promo_window and is_discounted:
            day_promo = day_org.copy()
            active_discount = discount_depth
            day_promo["is_steam_sale"] = 1
            day_promo["discount_depth"] = active_discount
            day_promo["effective_price"] = base_p * (1.0 - active_discount)
            day_promo["promo_x_discount"] = active_discount
            day_promo["promo_x_weekend"] = day_promo["is_weekend"]
            day_promo["sale_day_index"] = days_from_start
            day_promo["sale_progress"] = float((days_from_start + 1) / max(sale_duration_days, 1))
            day_promo["days_until_sale_end"] = max(sale_duration_days - 1 - days_from_start, 0)
            day_promo["storefront_exposure"] = float(SALE_RAMP.get(days_from_start, 1.0))
            day_promo["days_since_last_sale"] = 0
            if is_custom_event:
                day_promo["sale_name"] = custom_event_name
            else:
                day_promo["is_major_seasonal_sale"] = 1

            if hasattr(model, "uplift_model"):
                X_up = model._prepare_uplift_features(day_promo)
                rel_lift = max(0.0, float(model.uplift_model.predict(X_up)[0]))
                pred_val = org_val * (1.0 + rel_lift)
                peak_promotional_lift = max(peak_promotional_lift, rel_lift)
                pred_lower = max(0.0, pred_val * 0.85)
                pred_upper = pred_val * 1.15
            elif has_hybrid:
                p_base = float(precomputed_p_bases[idx])
                raw_lgb_promo = float(model.lgb_residual_model.predict(day_promo)[0])
                raw_promo_val = max(0.0, p_base + raw_lgb_promo)
                lift = max(0.0, raw_promo_val - org_val)
                peak_promotional_lift = max(peak_promotional_lift, lift)
                pred_val = org_val + lift
                pred_lower = org_lower + lift * 0.80
                pred_upper = org_upper + lift * 1.20
            elif hasattr(model, "predict_components"):
                promo_comps = model.predict_components(day_promo)
                raw_promo_val = float(promo_comps["combined"][0])
                lift = max(0.0, raw_promo_val - org_val)
                peak_promotional_lift = max(peak_promotional_lift, lift)
                pred_val = org_val + lift
                pred_lower = org_lower + lift * 0.80
                pred_upper = org_upper + lift * 1.20
            else:
                raw_promo_preds = model.predict(day_promo)
                raw_promo_val = float(raw_promo_preds[0])
                lift = max(0.0, raw_promo_val - org_val)
                pred_val = org_val + lift
                pred_lower = org_lower + lift * 0.80
                pred_upper = org_upper + lift * 1.20
            sale_name_str = custom_event_name if is_custom_event else "Steam Seasonal Sale"

        elif f_date > promo_end and is_discounted:
            days_since = (f_date - promo_end).days
            # Smooth post-sale gameplay tail: installations and playtime decay smoothly back to organic
            tail_factor = TAIL_INITIAL_FACTOR * np.exp(-days_since / TAIL_DECAY_RATE)
            if hasattr(model, "uplift_model"):
                tail_lift = peak_promotional_lift * tail_factor if tail_factor > 0.02 else 0.0
                pred_val = org_val * (1.0 + tail_lift)
            else:
                tail_lift = peak_promotional_lift * tail_factor if tail_factor > 0.02 else 0.0
                pred_val = org_val + tail_lift
            pred_lower = max(0.0, pred_val * 0.85)
            pred_upper = pred_val * 1.15
            sale_name_str = "None"
        else:
            pred_val = org_val
            pred_lower = org_lower
            pred_upper = org_upper
            sale_name_str = "None"

        forecast_rows.append(
            {
                "date": f_date,
                "predicted_ccu": round(pred_val, 1),
                "lower_bound": round(pred_lower, 1),
                "upper_bound": round(pred_upper, 1),
                "is_promo_active": in_promo_window,
                "discount_percent": int(discount_depth * 100) if in_promo_window else 0,
                "sale_name": sale_name_str,
                "day_of_week": day_with_cal["day_of_week"].iloc[0],
                "is_weekend": day_with_cal["is_weekend"].iloc[0],
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)

    # 2. Analytical KPI calculation
    promo_rows_df = forecast_df[forecast_df["is_promo_active"]]
    peak_promo_ccu = (
        float(promo_rows_df["predicted_ccu"].max())
        if not promo_rows_df.empty
        else float(forecast_df["predicted_ccu"].max())
    )
    peak_surge_multiplier = round(peak_promo_ccu / max(recent_baseline, 1.0), 2)
    total_forecasted_player_days = float(forecast_df["predicted_ccu"].sum())
    total_baseline_player_days = float(recent_baseline * len(forecast_df))
    incremental_lift = float(
        max(0.0, total_forecasted_player_days - total_baseline_player_days)
    )

    # Steam 30-day discount cooldown validation:
    # Steam policy mandates a 30-day cooldown between custom promotional discounts (excluding official major seasonal sales)
    cooldown_compliant = True
    cooldown_note = "Compliant with Steam discount policy."
    if is_custom_event:
        # Check if promo_start is within 30 days of the last historical sale
        days_since = cal.evaluate_date(promo_start)["days_since_last_sale"]
        if days_since < 30:
            cooldown_compliant = False
            cooldown_note = f"Warning: Steam requires 30 days cooldown between custom discounts (only {days_since} days elapsed at start of sale)."

    return {
        "forecast_df": forecast_df,
        "recent_baseline_ccu": round(recent_baseline, 1),
        "peak_promo_ccu": round(peak_promo_ccu, 1),
        "peak_surge_multiplier": peak_surge_multiplier,
        "total_incremental_player_days": round(incremental_lift, 1),
        "cooldown_compliant": cooldown_compliant,
        "cooldown_note": cooldown_note,
        "horizon_days": horizon_days,
        "total_simulated_days": len(forecast_df),
    }
