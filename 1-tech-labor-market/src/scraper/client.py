import logging
import random
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_USER_AGENTS = [
    (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
]


class HttpClient:
    """Robust HTTP client with rotating User-Agents, rate limiting, and retries."""

    def __init__(
        self,
        base_delay: float = 1.5,
        max_retries: int = 3,
        timeout: int = 15,
        custom_user_agent: str | None = None,
    ):
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.custom_user_agent = custom_user_agent
        self.session = requests.Session()

    def _get_headers(self) -> dict[str, str]:
        user_agent = self.custom_user_agent or random.choice(DEFAULT_USER_AGENTS)
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,uk;q=0.8",
            "Sec-Ch-Ua": '"Chromium";v="128", "Not;A=Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Upgrade-Insecure-Requests": "1",
        }

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        """Execute GET request with exponential backoff on transient errors."""
        headers = self._get_headers()

        for attempt in range(1, self.max_retries + 1):
            try:
                # Polite jitter delay between requests
                jitter = random.uniform(0.2, 0.8)
                time.sleep(self.base_delay + jitter)

                logger.debug(
                    f"Requesting GET {url} with params={params} (attempt {attempt}/{self.max_retries})"
                )
                response = self.session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )

                if response.status_code == 200:
                    return response

                if response.status_code in (429, 500, 502, 503, 504):
                    backoff = (2**attempt) + random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Received HTTP {response.status_code} for {url}. Backing off for {backoff:.2f}s..."
                    )
                    time.sleep(backoff)
                    continue

                response.raise_for_status()

            except (requests.RequestException, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    logger.error(
                        f"Failed GET {url} after {self.max_retries} attempts: {exc}"
                    )
                    raise
                backoff = (2**attempt) + random.uniform(0.5, 1.0)
                logger.warning(f"Request error: {exc}. Retrying in {backoff:.2f}s...")
                time.sleep(backoff)

        raise RuntimeError(f"Exceeded max retries for {url}")
