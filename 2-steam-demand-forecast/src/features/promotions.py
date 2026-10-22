"""Promotional event mapping and discount elasticity features."""

import numpy as np
import pandas as pd

from src.features.constants import SALE_RAMP
from src.ingestion.sales_calendar import SteamSalesCalendar


def create_promotional_features(
    df: pd.DataFrame,
    date_col: str = "date",
    calendar: SteamSalesCalendar | None = None,
    base_price: float = 14.99,
    default_discount_depth: float = 0.0,
) -> pd.DataFrame:
    """Map Steam Sales Calendar and discount indicators onto time series."""
    out = df.copy()
    cal = calendar or SteamSalesCalendar()

    dates = pd.to_datetime(out[date_col]).dt.date

    evaluations = [cal.evaluate_date(d) for d in dates]
    eval_df = pd.DataFrame(evaluations)

    for col in eval_df.columns:
        out[col] = eval_df[col].values

    # Storefront exposure ramp: Day 0 (35% due to 19:00 UTC start + download queue), Day 1 (85%), Day 2+ (100%)
    exp_series = out["sale_day_index"].map(SALE_RAMP).fillna(1.0)
    out["storefront_exposure"] = np.where(out["is_steam_sale"] == 1, exp_series, 0.0)

    # Discount depth: if an active Steam sale is happening and no explicit discount column is given,
    # assume typical indie discount depth (e.g. 20-50%), otherwise default_discount_depth
    if "discount_depth" not in out.columns:
        # Typical indie discount during major seasonal sale is 33% to 50%
        out["discount_depth"] = np.where(
            out["is_major_seasonal_sale"] == 1,
            0.40,
            np.where(out["is_steam_sale"] == 1, 0.25, default_discount_depth),
        )

    out["effective_price"] = base_price * (1.0 - out["discount_depth"])

    # Post-sale demand decay marker (exponential decay half-life ~ 5 days)
    out["post_sale_decay_7d"] = np.exp(-out["days_since_last_sale"].clip(0, 30) / 5.0)

    # Interaction terms
    if "is_weekend" in out.columns:
        out["promo_x_weekend"] = out["is_steam_sale"] * out["is_weekend"]
    else:
        out["promo_x_weekend"] = 0

    out["promo_x_discount"] = out["is_steam_sale"] * out["discount_depth"]

    return out
