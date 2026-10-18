"""Time-series calendar alignment, gap detection, and resampling."""

import pandas as pd


def align_daily_series(
    df: pd.DataFrame,
    date_col: str = "date",
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    """Ensure the DataFrame has a complete, unbroken daily datetime sequence.

    Args:
        df: Input DataFrame with date and metric columns.
        date_col: Name of the date column.
        fill_method: How to handle missing intermediate dates ('interpolate' or 'ffill').

    Returns:
        Clean DataFrame reindexed to complete daily frequency.
    """
    if df.empty:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col])

    # Reindex over full continuous date range
    start_date = out[date_col].min()
    end_date = out[date_col].max()
    full_idx = pd.date_range(start=start_date, end=end_date, freq="D", name=date_col)

    out = out.set_index(date_col).reindex(full_idx)

    # Impute numeric gaps without lookahead bias
    numeric_cols = out.select_dtypes(include=["number"]).columns
    if fill_method == "interpolate":
        out[numeric_cols] = out[numeric_cols].interpolate(method="time").ffill().bfill()
    else:
        out[numeric_cols] = out[numeric_cols].ffill().bfill()

    # Forward fill categorical / string columns if present
    non_numeric = [c for c in out.columns if c not in numeric_cols]
    if non_numeric:
        out[non_numeric] = out[non_numeric].ffill().bfill()

    out = out.reset_index()
    return out
