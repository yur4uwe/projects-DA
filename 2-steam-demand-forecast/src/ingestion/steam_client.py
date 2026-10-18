"""Official Steam API and Store client for real-time telemetry and metadata."""

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GamePricing(BaseModel):
    is_free: bool = False
    currency: str = "USD"
    initial_price: float = 0.0
    final_price: float = 0.0
    discount_percent: int = 0


class GameMetadata(BaseModel):
    appid: int
    name: str
    is_free: bool = False
    release_date: str | None = None
    pricing: GamePricing = Field(default_factory=GamePricing)
    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    is_multiplayer: bool = False
    is_coop: bool = False
    positive_reviews: int = 0
    negative_reviews: int = 0
    owners_bracket: str = "Unknown"
    top_tags: list[str] = Field(default_factory=list)


class SteamClient:
    """Client for Steam Web API, Steam Store API, and SteamSpy fallback."""

    STEAM_API_BASE = "https://api.steampowered.com"
    STEAM_STORE_BASE = "https://store.steampowered.com/api"
    STEAM_SPY_BASE = "https://steamspy.com/api.php"

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
                )
            },
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "SteamClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_current_players(self, appid: int) -> int:
        """Fetch current concurrent player count from official Steam API."""
        url = f"{self.STEAM_API_BASE}/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"
        try:
            resp = self.client.get(url, params={"appid": appid})
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("response", {}).get("player_count", 0))
        except Exception as e:
            logger.warning(f"Failed to fetch live CCU for appid {appid}: {e}")
            return 0

    def get_store_details(self, appid: int) -> dict[str, Any]:
        """Fetch detailed store record (pricing, release, categories) from Steam Store API."""
        url = f"{self.STEAM_STORE_BASE}/appdetails"
        try:
            resp = self.client.get(url, params={"appids": appid, "cc": "us", "l": "en"})
            resp.raise_for_status()
            data = resp.json()
            app_data = data.get(str(appid), {})
            if app_data.get("success"):
                return app_data.get("data", {})
        except Exception as e:
            logger.warning(f"Failed to fetch store details for appid {appid}: {e}")
        return {}

    def get_steamspy_details(self, appid: int) -> dict[str, Any]:
        """Fetch owner bracket, tags, and review stats from SteamSpy."""
        try:
            resp = self.client.get(
                self.STEAM_SPY_BASE, params={"request": "appdetails", "appid": appid}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch SteamSpy details for appid {appid}: {e}")
            return {}

    def get_game_metadata(self, appid: int) -> GameMetadata:
        """Fetch unified game metadata combining Store API and SteamSpy."""
        store_data = self.get_store_details(appid)
        spy_data = self.get_steamspy_details(appid)

        name = store_data.get("name") or spy_data.get("name") or f"AppID {appid}"
        is_free = store_data.get("is_free", False)

        price_overview = store_data.get("price_overview", {})
        initial_price = (
            (price_overview.get("initial", 0) / 100.0) if price_overview else 0.0
        )
        final_price = (
            (price_overview.get("final", 0) / 100.0) if price_overview else 0.0
        )
        discount_percent = (
            int(price_overview.get("discount_percent", 0)) if price_overview else 0
        )

        # Categories & Mechanics
        categories = [
            c.get("description", "") for c in store_data.get("categories", [])
        ]
        is_multi = any(
            "multi-player" in c.lower() or "pvp" in c.lower() for c in categories
        )
        is_coop = any("co-op" in c.lower() for c in categories)

        # Genres
        genres = [g.get("description", "") for g in store_data.get("genres", [])]
        if not genres and "genre" in spy_data:
            genres = [g.strip() for g in spy_data["genre"].split(",") if g.strip()]

        # Tags
        tags_dict = spy_data.get("tags") or {}
        top_tags = list(tags_dict.keys())[:10] if isinstance(tags_dict, dict) else []

        release_date = store_data.get("release_date", {}).get("date")

        return GameMetadata(
            appid=appid,
            name=name,
            is_free=is_free,
            release_date=release_date,
            pricing=GamePricing(
                is_free=is_free,
                currency=price_overview.get("currency", "USD"),
                initial_price=initial_price,
                final_price=final_price,
                discount_percent=discount_percent,
            ),
            genres=genres,
            categories=categories,
            is_multiplayer=is_multi,
            is_coop=is_coop,
            positive_reviews=int(spy_data.get("positive", 0)),
            negative_reviews=int(spy_data.get("negative", 0)),
            owners_bracket=str(spy_data.get("owners", "Unknown")),
            top_tags=top_tags,
        )
