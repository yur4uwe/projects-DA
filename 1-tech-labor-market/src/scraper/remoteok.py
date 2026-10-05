import logging
from datetime import UTC, datetime
from typing import Any

from .client import HttpClient

logger = logging.getLogger(__name__)

REMOTEOK_API_URL = "https://remoteok.com/api"


def scrape_remoteok_jobs(
    limit: int | None = None, client: HttpClient | None = None
) -> list[dict[str, Any]]:
    if client is None:
        client = HttpClient(
            base_delay=1.0,
            custom_user_agent="TechLaborMarketResearch/1.0 (academic study on tech compensation)",
        )
    logger.info(f"Fetching RemoteOK jobs from {REMOTEOK_API_URL}...")
    response = client.get(REMOTEOK_API_URL)

    try:
        raw_data = response.json()
    except ValueError as exc:
        logger.error(f"Failed to parse RemoteOK JSON: {exc}")
        return []

    if not isinstance(raw_data, list):
        logger.warning(f"Unexpected RemoteOK response type: {type(raw_data)}")
        return []

    now_iso = datetime.now(UTC).isoformat()
    standardized_jobs = []

    for item in raw_data:
        if not isinstance(item, dict):
            continue
        # Skip metadata/legal notice item
        if "legal" in item or "id" not in item:
            continue

        # Standardize salary structure to match schema
        salary_min = item.get("salary_min")
        salary_max = item.get("salary_max")
        base_salary = {}
        if salary_min or salary_max:
            base_salary = {
                "currency": "USD",
                "minValue": salary_min,
                "maxValue": salary_max,
                "unitText": "YEAR",
            }

        record = {
            "source": "remoteok",
            "scraped_at": now_iso,
            "id": str(item.get("id", "")),
            "title": item.get("position", ""),
            "category": item.get("tags", [None])[0] if item.get("tags") else "",
            "industry": "",
            "date_posted": item.get("date", ""),
            "valid_through": "",
            "url": item.get("url", "") or item.get("apply_url", ""),
            "employment_type": "FULL_TIME",
            "job_location_type": "TELECOMMUTE",
            "hiring_organization": item.get("company", ""),
            "tags": item.get("tags", []),
            "experience_requirements": {},
            "base_salary": base_salary,
            "description": item.get("description", ""),
        }
        standardized_jobs.append(record)

        if limit and len(standardized_jobs) >= limit:
            break

    logger.info(
        f"RemoteOK: successfully ingested {len(standardized_jobs)} job postings."
    )
    return standardized_jobs
