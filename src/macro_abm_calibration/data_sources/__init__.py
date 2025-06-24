"""
Data source connectors for macroeconomic data.

This module provides interfaces and implementations for connecting to
various economic data sources including OECD, Eurostat, and ICIO databases.
"""

from .base import DataSource, QueryResult, DataSourceError
from .oecd import OECDDataSource
from .eurostat import EurostatDataSource
from .icio import ICIODataSource
from .factory import create_data_source, DataSourceFactory

__all__ = [
    "DataSource",
    "QueryResult", 
    "DataSourceError",
    "OECDDataSource",
    "EurostatDataSource",
    "ICIODataSource",
    "create_data_source",
    "DataSourceFactory",
]