"""Steam sales calendar catalog and date evaluation engine."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class SteamSaleEvent:
    name: str
    sale_type: str  # "major_seasonal", "fest", "custom"
    start_date: date
    end_date: date
    typical_discount_multiplier: float = 1.0
    appid: int | None = None


# Curated catalog of major Steam sales from 2018 through 2026
HISTORICAL_STEAM_SALES: list[SteamSaleEvent] = [
    # 2018
    SteamSaleEvent(
        "Steam Spring Sale 2018", "major_seasonal", date(2018, 3, 22), date(2018, 3, 29)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2018", "major_seasonal", date(2018, 6, 21), date(2018, 7, 5)
    ),
    SteamSaleEvent(
        "Steam Halloween Sale 2018", "fest", date(2018, 10, 29), date(2018, 11, 1)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2018",
        "major_seasonal",
        date(2018, 11, 21),
        date(2018, 11, 27),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2018", "major_seasonal", date(2018, 12, 20), date(2019, 1, 3)
    ),
    # 2019
    SteamSaleEvent(
        "Steam Lunar New Year 2019", "fest", date(2019, 2, 4), date(2019, 2, 11)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2019", "major_seasonal", date(2019, 6, 25), date(2019, 7, 9)
    ),
    SteamSaleEvent(
        "Steam Halloween Sale 2019", "fest", date(2019, 10, 28), date(2019, 11, 1)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2019",
        "major_seasonal",
        date(2019, 11, 26),
        date(2019, 12, 3),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2019", "major_seasonal", date(2019, 12, 19), date(2020, 1, 2)
    ),
    # 2020
    SteamSaleEvent(
        "Steam Summer Sale 2020", "major_seasonal", date(2020, 6, 25), date(2020, 7, 9)
    ),
    SteamSaleEvent(
        "Steam Halloween Sale 2020", "fest", date(2020, 10, 29), date(2020, 11, 2)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2020",
        "major_seasonal",
        date(2020, 11, 25),
        date(2020, 12, 1),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2020", "major_seasonal", date(2020, 12, 22), date(2021, 1, 5)
    ),
    # 2021
    SteamSaleEvent(
        "Steam Spring Cleaning 2021", "fest", date(2021, 5, 27), date(2021, 5, 31)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2021", "major_seasonal", date(2021, 6, 24), date(2021, 7, 8)
    ),
    SteamSaleEvent(
        "Steam Halloween Sale 2021", "fest", date(2021, 10, 28), date(2021, 11, 1)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2021",
        "major_seasonal",
        date(2021, 11, 24),
        date(2021, 11, 30),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2021", "major_seasonal", date(2021, 12, 22), date(2022, 1, 5)
    ),
    # 2022
    SteamSaleEvent(
        "Steam Summer Sale 2022", "major_seasonal", date(2022, 6, 23), date(2022, 7, 7)
    ),
    SteamSaleEvent(
        "Steam Scream Fest 2022", "fest", date(2022, 10, 25), date(2022, 11, 1)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2022",
        "major_seasonal",
        date(2022, 11, 22),
        date(2022, 11, 29),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2022", "major_seasonal", date(2022, 12, 22), date(2023, 1, 5)
    ),
    # 2023
    SteamSaleEvent(
        "Steam Spring Sale 2023", "major_seasonal", date(2023, 3, 16), date(2023, 3, 23)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2023", "major_seasonal", date(2023, 6, 29), date(2023, 7, 13)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2023",
        "major_seasonal",
        date(2023, 11, 21),
        date(2023, 11, 28),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2023", "major_seasonal", date(2023, 12, 21), date(2024, 1, 4)
    ),
    # 2024
    SteamSaleEvent(
        "Steam Spring Sale 2024", "major_seasonal", date(2024, 3, 14), date(2024, 3, 21)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2024", "major_seasonal", date(2024, 6, 27), date(2024, 7, 11)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2024",
        "major_seasonal",
        date(2024, 11, 27),
        date(2024, 12, 4),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2024", "major_seasonal", date(2024, 12, 19), date(2025, 1, 2)
    ),
    # 2025
    SteamSaleEvent(
        "Steam Spring Sale 2025", "major_seasonal", date(2025, 3, 13), date(2025, 3, 20)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2025", "major_seasonal", date(2025, 6, 26), date(2025, 7, 10)
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2025",
        "major_seasonal",
        date(2025, 11, 25),
        date(2025, 12, 2),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2025", "major_seasonal", date(2025, 12, 18), date(2026, 1, 1)
    ),
    # 2026
    SteamSaleEvent(
        "Steam Spring Sale 2026", "major_seasonal", date(2026, 3, 12), date(2026, 3, 19)
    ),
    SteamSaleEvent(
        "Steam Summer Sale 2026", "major_seasonal", date(2026, 6, 25), date(2026, 7, 9)
    ),
    SteamSaleEvent(
        "Terraria Late Summer Sale",
        "custom",
        date(2026, 8, 20),
        date(2026, 8, 31),
        typical_discount_multiplier=0.50,
        appid=105600,
    ),
    SteamSaleEvent(
        "Hades Supergiant Publisher Sale",
        "custom",
        date(2026, 9, 26),
        date(2026, 10, 12),
        typical_discount_multiplier=0.66,
        appid=1145360,
    ),
    SteamSaleEvent(
        "Steam Autumn Sale 2026",
        "major_seasonal",
        date(2026, 11, 24),
        date(2026, 12, 1),
    ),
    SteamSaleEvent(
        "Steam Winter Sale 2026", "major_seasonal", date(2026, 12, 17), date(2027, 1, 1)
    ),
]


class SteamSalesCalendar:
    """Provides sale querying and calendar feature indicators."""

    def __init__(self, events: list[SteamSaleEvent] | None = None) -> None:
        self.events = sorted(
            events or HISTORICAL_STEAM_SALES, key=lambda e: e.start_date
        )

    def get_event_on(
        self, target_date: date | datetime, appid: int | None = None
    ) -> SteamSaleEvent | None:
        """Return the active sale event on a specific date, if any."""
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        matching = [
            ev for ev in self.events
            if ev.start_date <= target_date <= ev.end_date
            and (ev.appid is None or appid is None or ev.appid == appid)
        ]
        if not matching:
            return None

        # Prefer game-specific event over global event
        for ev in matching:
            if appid is not None and ev.appid == appid:
                return ev
        return matching[0]

    def evaluate_date(
        self, target_date: date | datetime, appid: int | None = None
    ) -> dict[str, object]:
        """Compute calendar feature flags for a given date."""
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        ev = self.get_event_on(target_date, appid=appid)
        if ev:
            day_index = (target_date - ev.start_date).days
            total_days = (ev.end_date - ev.start_date).days + 1
            days_until_end = (ev.end_date - target_date).days
            discount = float(ev.typical_discount_multiplier if ev.typical_discount_multiplier is not None else 0.33)
            return {
                "is_steam_sale": 1,
                "is_major_seasonal_sale": 1 if ev.sale_type == "major_seasonal" else 0,
                "sale_name": ev.name,
                "sale_day_index": day_index,
                "sale_progress": (day_index + 1) / max(total_days, 1),
                "days_until_sale_end": max(days_until_end, 0),
                "days_since_last_sale": 0,
                "discount_depth": discount,
            }

        # Calculate days since previous sale and days until next sale
        past_events = [
            e for e in self.events
            if e.end_date < target_date and (e.appid is None or appid is None or e.appid == appid)
        ]
        days_since_last = (
            (target_date - past_events[-1].end_date).days if past_events else 999
        )

        return {
            "is_steam_sale": 0,
            "is_major_seasonal_sale": 0,
            "sale_name": "None",
            "sale_day_index": -1,
            "sale_progress": 0.0,
            "days_until_sale_end": -1,
            "days_since_last_sale": min(days_since_last, 999),
            "discount_depth": 0.0,
        }

    def get_upcoming_sales(
        self, from_date: date | datetime, limit: int = 5
    ) -> list[SteamSaleEvent]:
        """Get upcoming sales starting after from_date."""
        if isinstance(from_date, datetime):
            from_date = from_date.date()

        upcoming = [e for e in self.events if e.end_date >= from_date]
        return upcoming[:limit]
