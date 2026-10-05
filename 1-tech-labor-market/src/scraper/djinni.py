import json
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup

from .client import HttpClient

logger = logging.getLogger(__name__)

BASE_URL = "https://djinni.co/jobs/"


def scrape_djinni_page(
    page: int = 1,
    extra_params: dict[str, Any] | None = None,
    client: HttpClient | None = None,
) -> list[dict[str, Any]]:
    if client is None:
        client = HttpClient(base_delay=1.5)

    params: dict[str, Any] = {"page": page}
    if extra_params:
        params.update(extra_params)

    logger.info(f"Fetching Djinni page {page}...")
    response = client.get(BASE_URL, params=params)

    soup = BeautifulSoup(response.text, "html.parser")
    ld_json_tag = soup.find("script", type="application/ld+json")

    if not ld_json_tag or not ld_json_tag.string:
        logger.warning(f"No JSON-LD script tag found on page {page}.")
        return []

    try:
        data = json.loads(ld_json_tag.string.strip())
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON-LD on page {page}: {exc}")
        return []

    if isinstance(data, dict):
        raw_jobs = [data]
    elif isinstance(data, list):
        raw_jobs = data
    else:
        raw_jobs = []

    now_iso = datetime.now(UTC).isoformat()
    standardized_jobs = []

    for item in raw_jobs:
        if not isinstance(item, dict) or item.get("@type") != "JobPosting":
            continue

        standardized_record = {
            "source": "djinni",
            "scraped_at": now_iso,
            "id": str(item.get("identifier", "")),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "industry": item.get("industry", ""),
            "date_posted": item.get("datePosted", ""),
            "valid_through": item.get("validThrough", ""),
            "url": item.get("url", ""),
            "employment_type": item.get("employmentType", ""),
            "job_location_type": item.get("jobLocationType", ""),
            "hiring_organization": item.get("hiringOrganization", {}).get("name", "")
            if isinstance(item.get("hiringOrganization"), dict)
            else "",
            "experience_requirements": item.get("experienceRequirements", {}),
            "base_salary": item.get("baseSalary", {}),
            "description": item.get("description", ""),
        }
        standardized_jobs.append(standardized_record)

    logger.info(
        f"Djinni page {page}: successfully extracted {len(standardized_jobs)} job postings."
    )
    return standardized_jobs


def scrape_djinni_pages(
    max_pages: int = 5,
    start_page: int = 1,
    extra_params: dict[str, Any] | None = None,
    client: HttpClient | None = None,
) -> Generator[list[dict[str, Any]], None, None]:
    """Iterate over multiple pages and yield records page-by-page."""
    if client is None:
        client = HttpClient(base_delay=1.5)

    for page_num in range(start_page, start_page + max_pages):
        records = scrape_djinni_page(
            page=page_num, extra_params=extra_params, client=client
        )
        if not records:
            logger.info(
                f"No more jobs found at page {page_num}. Terminating pagination."
            )
            break
        yield records
