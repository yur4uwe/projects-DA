import argparse
import logging
from pathlib import Path

from .djinni import scrape_djinni_pages
from .remoteok import scrape_remoteok_jobs
from .storage import save_raw_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run():
    parser = argparse.ArgumentParser(
        description="Tech Labor Market Data Ingestion Requester"
    )
    parser.add_argument(
        "--source",
        choices=["djinni", "remoteok", "all"],
        default="all",
        help="Data source to scrape (default: all)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
        help="Max pages to scrape for Djinni (default: 2)",
    )
    parser.add_argument(
        "--has-salary",
        action="store_true",
        default=False,
        help="Filter Djinni to vacancies with explicit public salary (editorial=has_public_salary)",
    )
    parser.add_argument(
        "--remoteok-limit",
        type=int,
        default=50,
        help="Limit of vacancies to ingest from RemoteOK (default: 50, 0 for unlimited)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"

    all_records = []

    # 1. Djinni Ingestion
    if args.source in ("djinni", "all"):
        logger.info(f"--- Running Djinni Ingestion (max_pages={args.pages}, has_salary={args.has_salary}) ---")
        djinni_records = []
        extra_params = {}
        if args.has_salary:
            extra_params["editorial"] = "has_public_salary"
        for page_records in scrape_djinni_pages(max_pages=args.pages, extra_params=extra_params if extra_params else None):
            djinni_records.extend(page_records)

        if djinni_records:
            out_path = save_raw_jsonl(djinni_records, "djinni", raw_data_dir)
            salary_count = sum(1 for r in djinni_records if r.get("base_salary"))
            logger.info(
                f"[Djinni Summary] Collected: {len(djinni_records)} | "
                f"With Salary: {salary_count} ({salary_count / len(djinni_records):.1%}) | "
                f"Saved to: {out_path.name}"
            )
            all_records.extend(djinni_records)

    # 2. RemoteOK Ingestion
    if args.source in ("remoteok", "all"):
        logger.info("--- Running RemoteOK Ingestion ---")
        limit = None if args.remoteok_limit == 0 else args.remoteok_limit
        remoteok_records = scrape_remoteok_jobs(limit=limit)

        if remoteok_records:
            out_path = save_raw_jsonl(remoteok_records, "remoteok", raw_data_dir)
            salary_count = sum(1 for r in remoteok_records if r.get("base_salary"))
            logger.info(
                f"[RemoteOK Summary] Collected: {len(remoteok_records)} | "
                f"With Salary: {salary_count} ({salary_count / len(remoteok_records):.1%}) | "
                f"Saved to: {out_path.name}"
            )
            all_records.extend(remoteok_records)

    total_salaries = sum(1 for r in all_records if r.get("base_salary"))
    logger.info(
        f"--- Ingestion Completed --- Total Records: {len(all_records)} | "
        f"With Salary Info: {total_salaries}"
    )


if __name__ == "__main__":
    run()
