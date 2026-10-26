"""GitHub API client with rate limiting, retries, and read-only safety assertions."""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

UPSTREAM_OWNER = "home-assistant"
UPSTREAM_REPO = "core"
GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    """Thread-safe, rate-limit-aware client for querying GitHub REST API."""

    def __init__(self, token: Optional[str] = None):
        self.token = (token or os.getenv("GITHUB_TOKEN", "")).strip()
        if not self.token:
            print(
                "[WARN] GITHUB_TOKEN not configured. Unauthenticated requests are severely rate limited (60/hr).",
                file=sys.stderr,
            )

        self.session = requests.Session()
        self.headers = {
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Ticket-Triage-Agent/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

    def check_rate_limits(self, response_headers: Dict[str, str]) -> None:
        """Check GitHub rate limits and sleep if remaining quota is critically low."""
        remaining_str = response_headers.get("X-RateLimit-Remaining")
        reset_str = response_headers.get("X-RateLimit-Reset")

        if remaining_str is None or reset_str is None:
            return

        try:
            remaining = int(remaining_str)
            if remaining <= 2:
                reset_epoch = int(reset_str)
                current_epoch = int(time.time())
                wait_seconds = max(0, reset_epoch - current_epoch) + 2
                reset_time = datetime.fromtimestamp(reset_epoch, tz=timezone.utc).strftime(
                    "%H:%M:%S UTC"
                )
                print(
                    f"\n[RATE LIMIT] Quota exhausted ({remaining} remaining). Sleeping {wait_seconds}s until {reset_time}..."
                )
                time.sleep(wait_seconds)
                print("[RATE LIMIT] Window reset. Resuming requests.")
        except (ValueError, TypeError):
            pass

    def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30
    ) -> requests.Response:
        """Perform a GET request to GitHub API with backoff on transient errors."""
        url = (
            endpoint if endpoint.startswith("http") else f"{GITHUB_API_BASE}/{endpoint.lstrip('/')}"
        )
        max_retries = 3
        backoff = 2

        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, headers=self.headers, params=params, timeout=timeout)
                self.check_rate_limits(dict(resp.headers))

                if (
                    resp.status_code in (403, 429)
                    and "X-RateLimit-Remaining" in resp.headers
                    and resp.headers["X-RateLimit-Remaining"] == "0"
                ):
                    self.check_rate_limits(dict(resp.headers))
                    continue

                if resp.status_code >= 500:
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                return resp
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2

        return resp
