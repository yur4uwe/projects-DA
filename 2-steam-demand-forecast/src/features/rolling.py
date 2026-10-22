"""Rolling window aggregation features conforming to course Listing 5."""

import pandas as pd


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    prefix: str = "sales",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create rolling statistics strictly avoiding future data leakage.

    Features are computed on the shifted series (t-1) so value at t
    reflects strictly past observations.
    """
    out = df.copy()
    window_sizes = windows or [3, 7, 14, 30]

    # Shift target by 1 day so today's observation isn't included in today's feature
    shifted_target = out[target_col].shift(1)

    for w in window_sizes:
        col_mean = f"{prefix}_rolling_{w}"
        out[col_mean] = shifted_target.rolling(window=w, min_periods=1).mean()

        col_std = f"{prefix}_rolling_std_{w}"
        out[col_std] = shifted_target.rolling(window=w, min_periods=2).std().fillna(0.0)

    # Momentum ratio: short-term 7d rolling to long-term 30d rolling
    col_3 = f"{prefix}_rolling_3"
    col_7 = f"{prefix}_rolling_7"
    col_30 = f"{prefix}_rolling_30"
    if col_7 in out.columns and col_30 in out.columns:
        out[f"{prefix}_momentum_7_30"] = (
            out[col_7] / out[col_30].replace(0, 1.0)
        ).fillna(1.0)

    # Bounded momentum ratio: short-term 3d rolling to medium 7d rolling
    if col_3 in out.columns and col_7 in out.columns:
        out[f"{prefix}_momentum_3_7"] = (
            (out[col_3] / out[col_7].replace(0, 1.0)).fillna(1.0).clip(0.5, 2.0)
        )
        # Bounded 3-day directional trend normalized by 7d mean:
        # (y_{t-1} - y_{t-4}) / (3 * rolling_7), clipped to [-1.0, 1.0]
        denom = (3.0 * out[col_7]).replace(0, 1.0)
        out[f"{prefix}_trend_3d"] = (
            ((shifted_target - shifted_target.shift(3)) / denom).fillna(0.0).clip(-1.0, 1.0)
        )

    return out
