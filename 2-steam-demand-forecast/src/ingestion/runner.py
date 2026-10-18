"""CLI orchestrator to ingest target indie games data into data/raw/."""

import argparse
import logging
from typing import Any

from src.ingestion.sales_calendar import SteamSalesCalendar
from src.ingestion.steam_client import SteamClient
from src.ingestion.steamcharts import SteamChartsClient
from src.ingestion.storage import IngestionStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion.runner")

DEFAULT_TARGET_GAMES = [
    {"appid": 367520, "name": "Hollow Knight", "archetype": "Metroidvania / Soulslike"},
    {"appid": 105600, "name": "Terraria", "archetype": "Sandbox / Survival Co-op"},
    {"appid": 413150, "name": "Stardew Valley", "archetype": "Farming / Cozy Sim"},
    {"appid": 588650, "name": "Dead Cells", "archetype": "Action Roguelite"},
    {"appid": 646570, "name": "Slay the Spire", "archetype": "Roguelike Deckbuilder"},
    {"appid": 1145360, "name": "Hades", "archetype": "Action Roguelike / Story-Rich"},
]


def ingest_game(
    appid: int,
    steam_client: SteamClient,
    charts_client: SteamChartsClient,
    storage: IngestionStorage,
) -> dict[str, Any]:
    """Ingest metadata, real-time metrics, daily series, and monthly archive for an appid."""
    logger.info(f"Ingesting data for AppID {appid}...")

    # 1. Fetch metadata & live CCU
    metadata = steam_client.get_game_metadata(appid)
    live_ccu = steam_client.get_current_players(appid)
    logger.info(
        f"[{metadata.name}] Live CCU: {live_ccu:,} | Base Price: ${metadata.pricing.initial_price:.2f} | Discount: {metadata.pricing.discount_percent}%"
    )

    # 2. Fetch daily series
    daily_series = charts_client.get_daily_series(appid)
    logger.info(
        f"[{metadata.name}] Ingested {len(daily_series)} daily time-series points."
    )

    # 3. Fetch monthly table archive
    monthly_archive = charts_client.get_monthly_table(appid)
    logger.info(
        f"[{metadata.name}] Ingested {len(monthly_archive)} monthly historical records."
    )

    # 4. Save to storage
    storage.save_metadata(appid, metadata)
    storage.save_records_jsonl(f"{appid}_daily_series.jsonl", daily_series)
    storage.save_records_jsonl(f"{appid}_monthly_archive.jsonl", monthly_archive)

    return {
        "appid": appid,
        "name": metadata.name,
        "live_ccu": live_ccu,
        "base_price": metadata.pricing.initial_price,
        "current_discount": metadata.pricing.discount_percent,
        "genres": metadata.genres,
        "top_tags": metadata.top_tags,
        "is_multiplayer": metadata.is_multiplayer,
        "is_coop": metadata.is_coop,
        "daily_records_count": len(daily_series),
        "monthly_records_count": len(monthly_archive),
    }


def run_pipeline(target_appids: list[int] | None = None) -> None:
    """Run full ingestion pipeline across all target games."""
    storage = IngestionStorage()
    calendar = SteamSalesCalendar()
    logger.info(f"Initialized Sales Calendar with {len(calendar.events)} major events.")

    appids_to_process = target_appids or [g["appid"] for g in DEFAULT_TARGET_GAMES]
    catalog_entries: list[dict[str, Any]] = []

    with SteamClient() as steam_client, SteamChartsClient() as charts_client:
        for aid in appids_to_process:
            try:
                entry = ingest_game(aid, steam_client, charts_client, storage)
                catalog_entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to ingest appid {aid}: {e}", exc_info=True)

    storage.save_catalog(catalog_entries)
    logger.info(
        f"Successfully updated raw catalog with {len(catalog_entries)} titles in data/raw/catalog.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Steam Game Ingestion Pipeline")
    parser.add_argument("--appid", type=int, help="Ingest a specific Steam AppID")
    args = parser.parse_args()

    target = [args.appid] if args.appid else None
    run_pipeline(target)


if __name__ == "__main__":
    main()
