"""
ERCOT API Client Package

This package provides a Python interface to the ERCOT (Electric Reliability Council of Texas) API.
It handles authentication, rate limiting, and data retrieval for various ERCOT endpoints.
"""

# Core API and authentication
from src.data.ercot.api import ERCOTApi
from src.data.ercot.auth import ERCOTAuth

# Data clients
from src.data.ercot.clients import DAMSettlementPointPricesClient
from src.data.ercot.clients import DAMSystemLambdaClient
from src.data.ercot.clients import LoadForecastClient
from src.data.ercot.clients import SolarGenerationClient
from src.data.ercot.clients import WindGenerationClient
from src.data.ercot.ercot_data import ERCOTBaseClient
from src.data.ercot.processors import ERCOTProcessor

__all__ = [
    "ERCOTAuth",
    "ERCOTApi",
    "ERCOTProcessor",
    "ERCOTBaseClient",
    "LoadForecastClient",
    "SolarGenerationClient",
    "WindGenerationClient",
    "DAMSettlementPointPricesClient",
    "DAMSystemLambdaClient",
]
