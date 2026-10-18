"""Unit tests for ingestion components and calendar logic."""

from datetime import date

from src.ingestion.sales_calendar import SteamSalesCalendar
from src.ingestion.steam_client import GameMetadata, GamePricing


def test_sales_calendar_detection() -> None:
    cal = SteamSalesCalendar()

    # Summer Sale 2024 ran from June 27 to July 11
    eval_summer = cal.evaluate_date(date(2024, 7, 1))
    assert eval_summer["is_steam_sale"] == 1
    assert eval_summer["is_major_seasonal_sale"] == 1
    assert eval_summer["sale_day_index"] == 4
    assert eval_summer["days_since_last_sale"] == 0

    # Non-sale date (e.g. August 15, 2024)
    eval_off = cal.evaluate_date(date(2024, 8, 15))
    assert eval_off["is_steam_sale"] == 0
    assert eval_off["is_major_seasonal_sale"] == 0
    assert eval_off["sale_day_index"] == -1
    assert eval_off["days_since_last_sale"] > 0


def test_game_metadata_schema() -> None:
    pricing = GamePricing(
        is_free=False,
        currency="USD",
        initial_price=14.99,
        final_price=9.99,
        discount_percent=33,
    )
    meta = GameMetadata(
        appid=367520,
        name="Hollow Knight",
        is_free=False,
        pricing=pricing,
        genres=["Action", "Indie"],
        categories=["Single-player"],
        is_multiplayer=False,
        is_coop=False,
        positive_reviews=400000,
        negative_reviews=12000,
        owners_bracket="5,000,000 .. 10,000,000",
        top_tags=["Metroidvania", "Souls-like"],
    )
    assert meta.appid == 367520
    assert meta.pricing.discount_percent == 33
    assert not meta.is_multiplayer
