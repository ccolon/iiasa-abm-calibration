"""
Data processing pipeline for macroeconomic calibration.

This module provides processors that transform raw data from various sources
into calibration-ready datasets, following the logic from the original MATLAB code.
"""

from .base import DataProcessor, ProcessingResult, ProcessingError
from .currency import CurrencyConverter
from .industry import IndustryAggregator
from .harmonizer import DataHarmonizer
from .pipeline import CalibrationPipeline
from .utils import calculate_deflator, interpolate_missing_data

__all__ = [
    "DataProcessor",
    "ProcessingResult", 
    "ProcessingError",
    "CurrencyConverter",
    "IndustryAggregator",
    "DataHarmonizer",
    "CalibrationPipeline",
    "calculate_deflator",
    "interpolate_missing_data",
]