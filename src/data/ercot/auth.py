"""Authentication module for ERCOT API access."""

import os
import time
from typing import Tuple
from urllib.parse import quote

import requests
from dotenv import load_dotenv


class ERCOTAuth:
    """Handles authentication with the ERCOT API."""

    # Token lifetime in seconds (1 hour)
    TOKEN_LIFETIME = 3600
    # Refresh token when less than this many seconds remain (5 minutes)
    TOKEN_REFRESH_THRESHOLD = 300

    def __init__(self):
        """Initialize with credentials loaded from environment."""
        self.username, self.password, self.subscription_key = self.load_credentials()
        self.access_token = None
        self.token_created_at = None

    @staticmethod
    def load_credentials() -> Tuple[str, str, str]:
        """Load ERCOT API credentials from environment variables.

        Returns:
            tuple: (username, password, subscription_key)

        Raises:
            ValueError: If any required credentials are missing
        """
        # Load environment variables
        load_dotenv()

        # Load ERCOT API credentials
        ercot_username = os.getenv("ERCOT_USERNAME")
        ercot_password = os.getenv("ERCOT_PASSWORD")
        ercot_subscription_key = os.getenv("ERCOT_SUBSCRIPTION_KEY")

        if not all([ercot_username, ercot_password, ercot_subscription_key]):
            raise ValueError(
                "Please set ERCOT_USERNAME, ERCOT_PASSWORD, and ERCOT_SUBSCRIPTION_KEY in .env file"
            )

        return ercot_username, ercot_password, ercot_subscription_key

    def should_refresh_token(self) -> bool:
        """Check if token should be refreshed.

        Returns:
            bool: True if token should be refreshed, False otherwise
        """
        if not self.access_token or not self.token_created_at:
            return True

        elapsed = time.time() - self.token_created_at
        remaining = self.TOKEN_LIFETIME - elapsed

        return remaining <= self.TOKEN_REFRESH_THRESHOLD

    def get_token(self) -> str:
        """Get ERCOT API authentication token using instance credentials.

        Returns:
            str: Authentication token for ERCOT API

        Raises:
            ValueError: If authentication fails or no token is received
        """
        if self.should_refresh_token():
            self.access_token = self._get_ercot_token(self.username, self.password)
            self.token_created_at = time.time()
            print("\nToken refreshed")

        return self.access_token

    @staticmethod
    def _get_ercot_token(username: str, password: str) -> str:
        """Get ERCOT API authentication token.

        Args:
            username (str): ERCOT API username
            password (str): ERCOT API password

        Returns:
            str: Authentication token for ERCOT API

        Raises:
            ValueError: If authentication fails or no token is received

        Note:
            ERCOT token lifetime is 1 hour.
        """
        # URL encode the credentials
        username_encoded = quote(username)
        password_encoded = quote(password)

        # Define authentication parameters
        client_id = "fec253ea-0d06-4272-a5e6-b478baeecd70"
        AUTH_PARAMS = {
            "base_url": "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com",
            "flow": "B2C_1_PUBAPI-ROPC-FLOW",
            "client_id": client_id,
            "scope": f"openid+{client_id}+offline_access",
        }

        # Build authentication URL
        AUTH_URL = (
            f"{AUTH_PARAMS['base_url']}/"
            f"{AUTH_PARAMS['flow']}/oauth2/v2.0/token"
            f"?username={{username}}"
            f"&password={{password}}"
            f"&grant_type=password"
            f"&scope={AUTH_PARAMS['scope']}"
            f"&client_id={AUTH_PARAMS['client_id']}"
            f"&response_type=id_token"
        )

        # Sign In/Authenticate
        auth_response = requests.post(
            AUTH_URL.format(username=username_encoded, password=password_encoded)
        )
        response_json = auth_response.json()

        if "error" in response_json:
            print("\nAuthentication Error Details:")
            print(f"Error Type: {response_json.get('error')}")
            print(f"Error Description: {response_json.get('error_description')}")
            print("\nPlease verify:")
            print("1. Your username and password are correct")
            print(
                "2. Your username is in the correct format (might need to be full email)"
            )
            print("3. You have completed ERCOT's registration process")
            print("4. Your account has been activated")
            raise ValueError("Authentication failed - see details above")

        # Retrieve access token
        access_token = response_json.get("access_token")
        if not access_token:
            print("\nWarning: Received successful response but no access token found")
            print(f"Full response: {response_json}")
            raise ValueError("No access token in response")

        return access_token

    def get_headers(self) -> dict:
        """Get headers for API requests including authentication.

        Returns:
            dict: Headers including subscription key and bearer token
        """
        return {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Authorization": f"Bearer {self.get_token()}",
        }
