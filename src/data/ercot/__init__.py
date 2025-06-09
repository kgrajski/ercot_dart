"""
ERCOT API Client Package

This package provides a Python interface to the ERCOT (Electric Reliability Council of Texas) API.
It handles authentication, rate limiting, and data retrieval for various ERCOT endpoints.
"""

from src.data.ercot.api import ERCOTApi
from src.data.ercot.auth import ERCOTAuth
from src.data.ercot.clients.dam_lambda import DAMSystemLambdaClient
from src.data.ercot.clients.dam_spp import DAMSettlementPointPricesClient
from src.data.ercot.clients.load import LoadForecastClient
from src.data.ercot.clients.solar import SolarGenerationClient
from src.data.ercot.clients.wind import WindGenerationClient
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
