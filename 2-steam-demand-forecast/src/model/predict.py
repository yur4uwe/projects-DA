"""Multi-step recursive forward forecasting and what-if promotional simulation."""

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.features.calendar import create_calendar_features
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

    sim_start = start_date or (last_date + timedelta(days=1))
    future_dates = [sim_start + timedelta(days=i) for i in range(horizon_days)]

    # 1. Determine promo schedule across future dates
    sale_end = sim_start + timedelta(days=sale_duration_days - 1)

    # Establish recent baseline demand (mean of last 14 days)
    recent_baseline = float(hist["demand"].iloc[-14:].mean())

    forecast_rows: list[dict[str, Any]] = []

    # Maintain sliding buffer of past and newly generated values for recursive lag updates
    sim_buffer = hist.copy()

    for idx, f_date in enumerate(future_dates):
        # Is the promo active on this day?
        in_promo_window = sim_start <= f_date <= sale_end

        # Base calendar features
        day_row = pd.DataFrame([{"date": pd.to_datetime(f_date)}])
        day_with_cal = create_calendar_features(day_row, date_col="date")

        # Map promo event
        active_discount = discount_depth if in_promo_window else 0.0

        day_with_promo = create_promotional_features(
            day_with_cal,
            date_col="date",
            calendar=cal,
            base_price=float(hist.get("base_price", pd.Series([14.99])).iloc[-1]),
            default_discount_depth=active_discount,
        )

        if in_promo_window:
            day_with_promo["is_steam_sale"] = 1
            day_with_promo["discount_depth"] = discount_depth
            day_with_promo["sale_name"] = (
                custom_event_name if is_custom_event else day_with_promo["sale_name"]
            )
            day_with_promo["sale_day_index"] = idx

        # Compute dynamic autoregressive lags from sim_buffer
        past_demands = sim_buffer["demand"].values
        day_with_promo["sales_lag_1"] = past_demands[-1]
        day_with_promo["sales_lag_2"] = (
            past_demands[-2] if len(past_demands) >= 2 else past_demands[-1]
        )
        day_with_promo["sales_lag_7"] = (
            past_demands[-7] if len(past_demands) >= 7 else past_demands[-1]
        )
        day_with_promo["sales_lag_14"] = (
            past_demands[-14] if len(past_demands) >= 14 else past_demands[-1]
        )
        day_with_promo["sales_lag_30"] = (
            past_demands[-30] if len(past_demands) >= 30 else past_demands[-1]
        )

        # Dynamic rolling window features
        day_with_promo["sales_rolling_7"] = float(np.mean(past_demands[-7:]))
        day_with_promo["sales_rolling_std_7"] = float(np.std(past_demands[-7:]))
        day_with_promo["sales_rolling_14"] = float(np.mean(past_demands[-14:]))
        day_with_promo["sales_rolling_std_14"] = float(np.std(past_demands[-14:]))
        day_with_promo["sales_rolling_30"] = float(np.mean(past_demands[-30:]))
        day_with_promo["sales_rolling_std_30"] = float(np.std(past_demands[-30:]))
        day_with_promo["sales_momentum_7_30"] = float(
            day_with_promo["sales_rolling_7"].iloc[0]
            / max(day_with_promo["sales_rolling_30"].iloc[0], 1.0)
        )

        # Static game features
        for static_col in [
            "appid",
            "game_name",
            "base_price",
            "is_multiplayer",
            "is_coop",
            "log_positive_reviews",
        ]:
            if static_col in hist.columns:
                day_with_promo[static_col] = hist[static_col].iloc[-1]

        # Predict with model
        if hasattr(model, "predict_components"):
            comps = model.predict_components(day_with_promo)
            pred_val = float(comps["combined"][0])
            pred_lower = float(comps["yhat_lower"][0])
            pred_upper = float(comps["yhat_upper"][0])
        else:
            preds = model.predict(day_with_promo)
            pred_val = float(preds[0])
            pred_lower = float(max(0.0, pred_val * 0.85))
            pred_upper = float(pred_val * 1.15)

        # Append prediction to simulation buffer for subsequent lag steps
        new_row = day_with_promo.copy()
        new_row["demand"] = pred_val
        sim_buffer = pd.concat([sim_buffer, new_row], ignore_index=True)

        forecast_rows.append(
            {
                "date": f_date,
                "predicted_ccu": round(pred_val, 1),
                "lower_bound": round(pred_lower, 1),
                "upper_bound": round(pred_upper, 1),
                "is_promo_active": in_promo_window,
                "discount_percent": int(discount_depth * 100) if in_promo_window else 0,
                "sale_name": day_with_promo["sale_name"].iloc[0],
                "day_of_week": day_with_promo["day_of_week"].iloc[0],
                "is_weekend": day_with_promo["is_weekend"].iloc[0],
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)

    # 2. Analytical KPI calculation
    peak_promo_ccu = float(forecast_df["predicted_ccu"].max())
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
        # Check if last sale was within 30 days
        days_since = cal.evaluate_date(sim_start)["days_since_last_sale"]
        if days_since < 30:
            cooldown_compliant = False
            cooldown_note = f"Warning: Steam requires 30 days cooldown between custom discounts (only {days_since} days elapsed)."

    return {
        "forecast_df": forecast_df,
        "recent_baseline_ccu": round(recent_baseline, 1),
        "peak_promo_ccu": round(peak_promo_ccu, 1),
        "peak_surge_multiplier": peak_surge_multiplier,
        "total_incremental_player_days": round(incremental_lift, 1),
        "cooldown_compliant": cooldown_compliant,
        "cooldown_note": cooldown_note,
        "horizon_days": horizon_days,
    }
