"""
ERCOT API Client Package

This package provides a Python interface to the ERCOT (Electric Reliability Council of Texas) API.
It handles authentication, rate limiting, and data retrieval for various ERCOT endpoints.
"""

from .auth import ERCOTAuth
from .api import ERCOTApi
from .processors import ERCOTProcessor
from .base import ERCOTBaseClient
from .clients.load import LoadForecastClient
from .clients.solar import SolarGenerationClient

__all__ = [
    'ERCOTAuth',
    'ERCOTApi',
    'ERCOTProcessor',
    'ERCOTBaseClient',
    'LoadForecastClient',
    'SolarGenerationClient',
] 