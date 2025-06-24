"""
Macroeconomic Agent-Based Model Calibration Package

This package provides tools for calibrating agent-based macroeconomic models
using OECD data sources including national accounts, input-output tables,
and financial statistics.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .config import CalibrationConfig
from .models import Country, Industry, TimeFrame
from .data_sources import (
    DataSource, OECDDataSource, EurostatDataSource, ICIODataSource,
    DataSourceFactory, DataSourceManager
)
from .processors import (
    DataProcessor, CurrencyConverter, IndustryAggregator, 
    DataHarmonizer, CalibrationPipeline
)
from .calibrators import (
    Calibrator, CalibrationResult, CalibrationStatus,
    ABMParameterEstimator, InitialConditionsSetter, ModelValidator
)

__all__ = [
    "CalibrationConfig",
    "Country", 
    "Industry",
    "TimeFrame",
    "DataSource",
    "OECDDataSource",
    "EurostatDataSource", 
    "ICIODataSource",
    "DataSourceFactory",
    "DataSourceManager",
    "DataProcessor",
    "CurrencyConverter",
    "IndustryAggregator",
    "DataHarmonizer",
    "CalibrationPipeline",
    "Calibrator",
    "CalibrationResult",
    "CalibrationStatus",
    "ABMParameterEstimator",
    "InitialConditionsSetter",
    "ModelValidator",
]