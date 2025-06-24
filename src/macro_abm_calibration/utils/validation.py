"""
Data validation utilities for ensuring data quality and consistency.

This module provides comprehensive validation functions that replicate
and extend the validation logic from the original MATLAB code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ..models import Country, Industry


@dataclass
class ValidationResult:
    """
    Result of a data validation check.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        details: Additional validation details
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_detail(self, key: str, value: Union[str, int, float]) -> None:
        """Add a detail."""
        self.details[key] = value


class DataValidator:
    """
    Comprehensive data validator for macroeconomic time series.
    
    This class provides validation methods that ensure data quality,
    consistency, and completeness across different datasets.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize validator.
        
        Args:
            tolerance: Tolerance for numerical comparisons
        """
        self.tolerance = tolerance
    
    def validate_time_series_dimensions(
        self,
        data: pd.DataFrame,
        expected_countries: List[str],
        expected_years: List[int],
        name: str = "dataset"
    ) -> ValidationResult:
        """
        Validate time series data dimensions.
        
        Args:
            data: DataFrame with time series data
            expected_countries: Expected country codes
            expected_years: Expected years
            name: Dataset name for error messages
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Check if DataFrame is empty
        if data.empty:
            result.add_error(f"{name}: DataFrame is empty")
            return result
        
        # Validate countries (assuming countries are in columns or index)
        if hasattr(data, 'columns'):
            actual_countries = set(data.columns)
            expected_countries_set = set(expected_countries)
            
            missing_countries = expected_countries_set - actual_countries
            extra_countries = actual_countries - expected_countries_set
            
            if missing_countries:
                result.add_error(f"{name}: Missing countries: {missing_countries}")
            
            if extra_countries:
                result.add_warning(f"{name}: Extra countries: {extra_countries}")
        
        # Validate time dimension
        if hasattr(data, 'index'):
            if isinstance(data.index, pd.DatetimeIndex):
                actual_years = set(data.index.year)
            else:
                # Assume numeric year index
                actual_years = set(data.index)
            
            expected_years_set = set(expected_years)
            missing_years = expected_years_set - actual_years
            extra_years = actual_years - expected_years_set
            
            if missing_years:
                result.add_error(f"{name}: Missing years: {missing_years}")
            
            if extra_years:
                result.add_warning(f"{name}: Extra years: {extra_years}")
        
        # Add dimension details
        result.add_detail("rows", len(data))
        result.add_detail("columns", len(data.columns) if hasattr(data, 'columns') else 0)
        
        return result
    
    def validate_missing_values(
        self,
        data: pd.DataFrame,
        name: str = "dataset",
        max_missing_pct: float = 0.1
    ) -> ValidationResult:
        """
        Validate missing values in dataset.
        
        Args:
            data: DataFrame to validate
            name: Dataset name for error messages
            max_missing_pct: Maximum allowed percentage of missing values
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        if data.empty:
            result.add_error(f"{name}: DataFrame is empty")
            return result
        
        # Calculate missing value statistics
        total_values = data.size
        missing_values = data.isnull().sum().sum()
        missing_pct = missing_values / total_values if total_values > 0 else 0
        
        result.add_detail("total_values", total_values)
        result.add_detail("missing_values", missing_values)
        result.add_detail("missing_percentage", missing_pct)
        
        # Check overall missing percentage
        if missing_pct > max_missing_pct:
            result.add_error(
                f"{name}: Missing values ({missing_pct:.2%}) exceed threshold ({max_missing_pct:.2%})"
            )
        
        # Check for columns/rows with all missing values
        if hasattr(data, 'columns'):
            all_missing_cols = data.columns[data.isnull().all()].tolist()
            if all_missing_cols:
                result.add_error(f"{name}: Columns with all missing values: {all_missing_cols}")
        
        all_missing_rows = data.index[data.isnull().all(axis=1)].tolist()
        if all_missing_rows:
            result.add_warning(f"{name}: Rows with all missing values: {len(all_missing_rows)}")
        
        return result
    
    def validate_numerical_consistency(
        self,
        data: pd.DataFrame,
        name: str = "dataset"
    ) -> ValidationResult:
        """
        Validate numerical consistency (no infinite values, reasonable ranges).
        
        Args:
            data: DataFrame to validate
            name: Dataset name for error messages
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        if data.empty:
            result.add_error(f"{name}: DataFrame is empty")
            return result
        
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            result.add_warning(f"{name}: No numeric columns found")
            return result
        
        # Check for infinite values
        inf_mask = np.isinf(numeric_data)
        inf_count = inf_mask.sum().sum()
        
        if inf_count > 0:
            result.add_error(f"{name}: Found {inf_count} infinite values")
            
            # Report columns with infinite values
            inf_columns = numeric_data.columns[inf_mask.any()].tolist()
            result.add_detail("infinite_value_columns", inf_columns)
        
        # Check for extremely large values (potential data errors)
        large_threshold = 1e12  # Adjust based on expected data scale
        large_mask = np.abs(numeric_data) > large_threshold
        large_count = large_mask.sum().sum()
        
        if large_count > 0:
            result.add_warning(f"{name}: Found {large_count} extremely large values (>{large_threshold})")
        
        # Check for negative values where they shouldn't be (e.g., GDP, prices)
        # This would need to be customized based on variable types
        negative_mask = numeric_data < 0
        negative_count = negative_mask.sum().sum()
        
        if negative_count > 0:
            result.add_detail("negative_values", negative_count)
            negative_columns = numeric_data.columns[negative_mask.any()].tolist()
            result.add_detail("negative_value_columns", negative_columns)
        
        # Statistical summary
        result.add_detail("numeric_columns", len(numeric_data.columns))
        result.add_detail("mean_value", float(numeric_data.mean().mean()))
        result.add_detail("std_value", float(numeric_data.std().mean()))
        
        return result
    
    def validate_exchange_rate_consistency(
        self,
        rates: pd.DataFrame,
        base_currency: str = "USD"
    ) -> ValidationResult:
        """
        Validate exchange rate data for consistency and reasonableness.
        
        Args:
            rates: DataFrame with exchange rates
            base_currency: Base currency for validation
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        if rates.empty:
            result.add_error("Exchange rates: DataFrame is empty")
            return result
        
        # Check for rates equal to zero or negative
        zero_negative_mask = rates <= 0
        zero_negative_count = zero_negative_mask.sum().sum()
        
        if zero_negative_count > 0:
            result.add_error(f"Exchange rates: Found {zero_negative_count} zero or negative rates")
        
        # Check for extreme rate changes (potential data errors)
        if len(rates) > 1:
            rate_changes = rates.pct_change().abs()
            extreme_change_threshold = 0.5  # 50% change threshold
            extreme_changes = rate_changes > extreme_change_threshold
            extreme_count = extreme_changes.sum().sum()
            
            if extreme_count > 0:
                result.add_warning(
                    f"Exchange rates: Found {extreme_count} extreme rate changes (>{extreme_change_threshold:.0%})"
                )
        
        # Check for base currency rate (should be 1.0 if present)
        if base_currency in rates.columns:
            base_rates = rates[base_currency]
            if not np.allclose(base_rates, 1.0, rtol=self.tolerance):
                result.add_error(f"Exchange rates: Base currency {base_currency} rates not equal to 1.0")
        
        return result
    
    def validate_input_output_consistency(
        self,
        intermediate_consumption: np.ndarray,
        final_demand: np.ndarray,
        output: np.ndarray,
        name: str = "Input-Output"
    ) -> ValidationResult:
        """
        Validate input-output table consistency.
        
        Args:
            intermediate_consumption: Intermediate consumption matrix
            final_demand: Final demand vector/matrix
            output: Output vector
            name: Dataset name for error messages
            
        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        
        # Check dimensions
        if intermediate_consumption.ndim != 2:
            result.add_error(f"{name}: Intermediate consumption must be 2D matrix")
            return result
        
        n_industries = intermediate_consumption.shape[0]
        
        if intermediate_consumption.shape[1] != n_industries:
            result.add_error(f"{name}: Intermediate consumption matrix must be square")
        
        # Check output consistency: output = intermediate_consumption.sum(axis=1) + final_demand
        if final_demand.ndim == 1:
            total_demand = intermediate_consumption.sum(axis=1) + final_demand
        else:
            total_demand = intermediate_consumption.sum(axis=1) + final_demand.sum(axis=1)
        
        output_diff = np.abs(output - total_demand)
        max_diff = np.max(output_diff)
        relative_diff = max_diff / np.max(output) if np.max(output) > 0 else 0
        
        result.add_detail("max_output_difference", float(max_diff))
        result.add_detail("relative_output_difference", float(relative_diff))
        
        if relative_diff > self.tolerance:
            result.add_error(
                f"{name}: Output inconsistency. Max relative difference: {relative_diff:.4f}"
            )
        
        # Check for negative values
        if np.any(intermediate_consumption < 0):
            result.add_warning(f"{name}: Negative values in intermediate consumption")
        
        if np.any(final_demand < 0):
            result.add_warning(f"{name}: Negative values in final demand")
        
        if np.any(output < 0):
            result.add_error(f"{name}: Negative values in output")
        
        return result


def validate_data_consistency(
    datasets: Dict[str, pd.DataFrame],
    countries: List[Country],
    industries: List[Industry],
    years: List[int],
    tolerance: float = 0.01
) -> Dict[str, ValidationResult]:
    """
    Validate consistency across multiple datasets.
    
    Args:
        datasets: Dictionary of datasets to validate
        countries: List of expected countries
        industries: List of expected industries
        years: List of expected years
        tolerance: Tolerance for numerical comparisons
        
    Returns:
        Dictionary of validation results for each dataset
    """
    validator = DataValidator(tolerance=tolerance)
    results = {}
    
    country_codes = [c.oecd_code for c in countries]
    
    for name, data in datasets.items():
        # Run all validation checks
        dim_result = validator.validate_time_series_dimensions(
            data, country_codes, years, name
        )
        
        missing_result = validator.validate_missing_values(data, name)
        
        numeric_result = validator.validate_numerical_consistency(data, name)
        
        # Combine results
        combined_result = ValidationResult(is_valid=True)
        
        for result in [dim_result, missing_result, numeric_result]:
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
            combined_result.details.update(result.details)
            
            if not result.is_valid:
                combined_result.is_valid = False
        
        results[name] = combined_result
    
    return results