"""
Base classes for ABM calibration.

This module provides the foundational classes and interfaces for all
calibration operations in the macroeconomic ABM system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

import pandas as pd
import numpy as np


class CalibrationStatus(Enum):
    """Status of calibration process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class CalibrationMetadata:
    """Metadata for calibration operations."""
    calibrator_name: str
    operation_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    computation_time: Optional[float] = None
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class CalibrationResult:
    """Result of a calibration operation."""
    status: CalibrationStatus
    data: Union[Dict[str, Any], pd.DataFrame, np.ndarray]
    metadata: CalibrationMetadata
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        """Check if calibration was successful."""
        return self.status == CalibrationStatus.COMPLETED
    
    @property
    def has_errors(self) -> bool:
        """Check if calibration has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if calibration has warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        if self.status == CalibrationStatus.COMPLETED:
            self.status = CalibrationStatus.FAILED
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)


class Calibrator(ABC):
    """
    Abstract base class for all calibration operations.
    
    This class provides the common interface and functionality for all
    calibration components including parameter estimation, initial conditions
    setting, and model validation.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            name: Name of the calibrator
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"macro_abm_calibration.calibrators.{self.name}")
        self._operation_counter = 0
    
    @abstractmethod
    def calibrate(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Perform calibration operation.
        
        Args:
            data: Input data for calibration
            parameters: Calibration parameters
            
        Returns:
            CalibrationResult with calibrated parameters or initial conditions
        """
        pass
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID."""
        self._operation_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{timestamp}_{self._operation_counter:03d}"
    
    def validate_inputs(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Validate input data and parameters.
        
        Args:
            data: Input data
            parameters: Parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic data validation
        if data is None:
            errors.append("Input data cannot be None")
        elif isinstance(data, pd.DataFrame) and data.empty:
            errors.append("Input DataFrame cannot be empty")
        elif isinstance(data, dict) and not data:
            errors.append("Input data dictionary cannot be empty")
        
        return errors
    
    def get_info(self) -> Dict[str, Any]:
        """Get calibrator information."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "operations_performed": self._operation_counter
        }


class EconomicCalibrator(Calibrator):
    """
    Base class for economic model calibrators.
    
    This class extends the base Calibrator with economic-specific functionality
    and validation methods.
    """
    
    # Standard economic variables expected in calibration data
    REQUIRED_VARIABLES = [
        "gdp", "consumption", "investment", "exports", "imports",
        "unemployment_rate", "interest_rate", "inflation_rate"
    ]
    
    # Standard country codes for validation
    EXPECTED_COUNTRIES = [
        "USA", "GBR", "DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "PRT",
        "GRC", "IRL", "FIN", "SVK", "SVN", "EST", "LVA", "LTU", "LUX", "MLT",
        "CYP", "CZE", "HUN", "POL", "BGR", "ROU", "HRV", "DNK", "SWE", "JPN", "MEX"
    ]
    
    def validate_economic_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[str]:
        """
        Validate economic data structure and content.
        
        Args:
            data: Economic data to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if isinstance(data, pd.DataFrame):
            errors.extend(self._validate_economic_dataframe(data))
        elif isinstance(data, dict):
            for name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df_errors = self._validate_economic_dataframe(df, f"{name}.")
                    errors.extend(df_errors)
        
        return errors
    
    def _validate_economic_dataframe(
        self,
        data: pd.DataFrame,
        prefix: str = ""
    ) -> List[str]:
        """Validate individual economic DataFrame."""
        errors = []
        
        # Check for required columns
        if "REF_AREA" not in data.columns:
            errors.append(f"{prefix}Missing REF_AREA column")
        
        if "TIME_PERIOD" not in data.columns:
            errors.append(f"{prefix}Missing TIME_PERIOD column")
        
        # Check for economic variables
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            errors.append(f"{prefix}No numeric variables found")
        
        # Check for reasonable value ranges
        for col in numeric_cols:
            if col.endswith("_rate") or "rate" in col.lower():
                # Rates should typically be between -50% and 50%
                extreme_rates = data[col].abs() > 0.5
                if extreme_rates.any():
                    errors.append(f"{prefix}Extreme values in {col} (>50%)")
            
            # Check for all-zero or all-NaN columns
            if data[col].isna().all():
                errors.append(f"{prefix}All NaN values in {col}")
            elif (data[col] == 0).all():
                errors.append(f"{prefix}All zero values in {col}")
        
        return errors
    
    def calculate_economic_indicators(
        self,
        data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate standard economic indicators from raw data.
        
        Args:
            data: Raw economic data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Growth rates
        if "gdp" in data.columns:
            indicators["gdp_growth"] = data["gdp"].pct_change()
        
        if "consumption" in data.columns:
            indicators["consumption_growth"] = data["consumption"].pct_change()
        
        # Ratios
        if "consumption" in data.columns and "gdp" in data.columns:
            indicators["consumption_gdp_ratio"] = data["consumption"] / data["gdp"]
        
        if "investment" in data.columns and "gdp" in data.columns:
            indicators["investment_gdp_ratio"] = data["investment"] / data["gdp"]
        
        # Trade balance
        if "exports" in data.columns and "imports" in data.columns:
            indicators["trade_balance"] = data["exports"] - data["imports"]
            
            if "gdp" in data.columns:
                indicators["trade_balance_gdp_ratio"] = indicators["trade_balance"] / data["gdp"]
        
        return indicators


class CalibrationManager:
    """
    Manager class for coordinating multiple calibrators.
    
    This class orchestrates the calibration process across different
    calibrators and manages dependencies between them.
    """
    
    def __init__(self):
        """Initialize calibration manager."""
        self.calibrators: Dict[str, Calibrator] = {}
        self.results: Dict[str, CalibrationResult] = {}
        self.logger = logging.getLogger("macro_abm_calibration.calibrators.manager")
    
    def register_calibrator(self, name: str, calibrator: Calibrator) -> None:
        """
        Register a calibrator.
        
        Args:
            name: Name to register the calibrator under
            calibrator: Calibrator instance
        """
        self.calibrators[name] = calibrator
        self.logger.info(f"Registered calibrator: {name}")
    
    def run_calibration_sequence(
        self,
        data: Dict[str, pd.DataFrame],
        sequence: List[str],
        parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, CalibrationResult]:
        """
        Run calibrators in specified sequence.
        
        Args:
            data: Input data for calibration
            sequence: Ordered list of calibrator names to run
            parameters: Parameters for each calibrator
            
        Returns:
            Dictionary of calibration results
        """
        results = {}
        working_data = data.copy()
        
        for calibrator_name in sequence:
            if calibrator_name not in self.calibrators:
                self.logger.error(f"Calibrator {calibrator_name} not found")
                continue
            
            calibrator = self.calibrators[calibrator_name]
            calibrator_params = parameters.get(calibrator_name, {}) if parameters else {}
            
            self.logger.info(f"Running calibrator: {calibrator_name}")
            
            try:
                result = calibrator.calibrate(working_data, calibrator_params)
                results[calibrator_name] = result
                
                # Update working data with results for next calibrator
                if result.is_success and isinstance(result.data, dict):
                    working_data.update(result.data)
                
                self.logger.info(f"Calibrator {calibrator_name} completed: {result.status.value}")
                
            except Exception as e:
                self.logger.error(f"Calibrator {calibrator_name} failed: {e}")
                
                # Create failed result
                metadata = CalibrationMetadata(
                    calibrator_name=calibrator_name,
                    operation_id=calibrator._generate_operation_id()
                )
                
                failed_result = CalibrationResult(
                    status=CalibrationStatus.FAILED,
                    data={},
                    metadata=metadata,
                    errors=[str(e)]
                )
                
                results[calibrator_name] = failed_result
        
        self.results.update(results)
        return results
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of all calibration results."""
        summary = {
            "total_calibrators": len(self.calibrators),
            "completed_calibrations": len(self.results),
            "successful_calibrations": sum(1 for r in self.results.values() if r.is_success),
            "failed_calibrations": sum(1 for r in self.results.values() if not r.is_success),
            "calibrator_status": {
                name: result.status.value 
                for name, result in self.results.items()
            }
        }
        
        return summary