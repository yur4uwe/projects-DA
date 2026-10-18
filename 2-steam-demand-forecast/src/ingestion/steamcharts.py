"""SteamCharts client for high-resolution time-series and monthly historical tables."""

import logging
import re
from collections import defaultdict
from datetime import UTC, date, datetime
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DailyPlayerRecord(BaseModel):
    date: date
    avg_players: float
    peak_players: int
    min_players: int
    samples: int


class MonthlyPlayerRecord(BaseModel):
    month_label: str
    avg_players: float
    gain: float
    gain_percent: float
    peak_players: int


class SteamChartsClient:
    """Client for fetching and parsing SteamCharts time-series data."""

    BASE_URL = "https://steamcharts.com/app"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/html, */*",
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "SteamChartsClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_raw_chart_data(self, appid: int) -> list[list[int | float]]:
        """Fetch raw [[timestamp_ms, player_count], ...] from SteamCharts."""
        url = f"{self.BASE_URL}/{appid}/chart-data.json"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.error(f"Error fetching chart-data for appid {appid}: {e}")
        return []

    def get_daily_series(self, appid: int) -> list[DailyPlayerRecord]:
        """Aggregate SteamCharts chart points into uniform daily records."""
        raw_points = self.get_raw_chart_data(appid)
        if not raw_points:
            return []

        # Bucket by UTC date
        daily_buckets: dict[date, list[int]] = defaultdict(list)
        for point in raw_points:
            if (
                len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                ts_ms = point[0]
                count = int(point[1])
                dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC).date()
                daily_buckets[dt].append(count)

        records: list[DailyPlayerRecord] = []
        for d in sorted(daily_buckets.keys()):
            vals = daily_buckets[d]
            records.append(
                DailyPlayerRecord(
                    date=d,
                    avg_players=float(sum(vals) / len(vals)),
                    peak_players=max(vals),
                    min_players=min(vals),
                    samples=len(vals),
                )
            )
        return records

    def get_monthly_table(self, appid: int) -> list[MonthlyPlayerRecord]:
        """Scrape historical monthly summary table from SteamCharts."""
        url = f"{self.BASE_URL}/{appid}"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logger.error(f"Error fetching monthly HTML for appid {appid}: {e}")
            return []

        table_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
        records: list[MonthlyPlayerRecord] = []

        for row in table_rows:
            cols = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
            if len(cols) >= 5:
                # Clean HTML tags and entities
                clean_cols = [
                    re.sub(r"&#43;", "+", re.sub(r"<[^>]+>", "", c)).strip()
                    for c in cols
                ]
                month_label = clean_cols[0]
                if not month_label or "Month" in month_label:
                    continue

                def _parse_float(val: str) -> float:
                    clean = re.sub(r"[^\d.-]", "", val)
                    return float(clean) if clean and clean != "-" else 0.0

                avg_p = _parse_float(clean_cols[1])
                gain = _parse_float(clean_cols[2])
                gain_pct = _parse_float(clean_cols[3])
                peak_p = int(_parse_float(clean_cols[4]))

                records.append(
                    MonthlyPlayerRecord(
                        month_label=month_label,
                        avg_players=avg_p,
                        gain=gain,
                        gain_percent=gain_pct,
                        peak_players=peak_p,
                    )
                )

        return records
