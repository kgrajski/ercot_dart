"""API interaction module for ERCOT API access."""

import time
from typing import Dict
from typing import Optional

import requests

from src.data.ercot.auth import ERCOTAuth


class ERCOTApi:
    """Handles API interactions with rate limiting and retries."""

    # Base URL for all ERCOT API endpoints
    BASE_URL = "https://api.ercot.com/api/public-reports"

    # API rate limit (requests per minute)
    RATE_LIMIT = 28  # It is actually 30, but we'll be conservative
    MIN_REQUEST_INTERVAL = 60.0 / RATE_LIMIT  # seconds between requests
    MAX_RETRIES = 3  # Maximum number of retries for rate limit errors
    RATE_LIMIT_WAIT = 5  # Seconds to wait after hitting rate limit

    def __init__(self, auth: ERCOTAuth):
        """Initialize with authentication handler.

        Args:
            auth (ERCOTAuth): Authentication handler instance
        """
        self.auth = auth
        self._last_request_time = 0

    def make_request(
        self, endpoint_path: str, params: Optional[Dict] = None, retry_count: int = 0
    ) -> requests.Response:
        """Make an API request with rate limiting and retry logic.

        Args:
            endpoint_path (str): API endpoint path to request
            params (dict, optional): Query parameters
            retry_count (int): Current retry attempt number

        Returns:
            requests.Response: The API response

        Raises:
            Exception: If max retries exceeded or non-rate-limit error occurs
        """
        # Ensure we respect rate limiting
        self._wait_for_rate_limit()

        # Build full URL
        url = f"{self.BASE_URL}/{endpoint_path}"

        try:
            # Make the request
            response = requests.get(url, params=params, headers=self.auth.get_headers())

            # Update last request time
            self._last_request_time = time.time()

            if response.status_code == 429:  # Rate limit exceeded
                if retry_count >= self.MAX_RETRIES:
                    raise Exception(
                        f"Max retries ({self.MAX_RETRIES}) exceeded for rate limit"
                    )

                # Wait longer than the suggested time
                print(
                    f"\nRate limit hit. Waiting {self.RATE_LIMIT_WAIT} seconds before retry {retry_count + 1}..."
                )
                time.sleep(self.RATE_LIMIT_WAIT)
                return self.make_request(endpoint_path, params, retry_count + 1)

            return response

        except requests.exceptions.RequestException as e:
            if retry_count >= self.MAX_RETRIES:
                raise Exception(f"Max retries ({self.MAX_RETRIES}) exceeded: {str(e)}")

            print(
                f"\nRequest failed. Waiting {self.RATE_LIMIT_WAIT} seconds before retry {retry_count + 1}..."
            )
            time.sleep(self.RATE_LIMIT_WAIT)
            return self.make_request(endpoint_path, params, retry_count + 1)

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect the rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
