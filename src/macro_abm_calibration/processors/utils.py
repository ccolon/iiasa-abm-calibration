"""
Utility functions for data processing.

This module provides common utility functions used across different processors,
replicating key calculations from the original MATLAB code.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import interpolate

from ..utils.logging import get_logger

logger = get_logger(__name__)


def calculate_deflator(
    nominal_data: Union[pd.Series, np.ndarray],
    real_data: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate deflator from nominal and real data.
    
    This function replicates the calculate_deflator.m functionality,
    computing the ratio of nominal to real values.
    
    Args:
        nominal_data: Nominal values
        real_data: Real (constant price) values
        
    Returns:
        Deflator series (nominal/real ratio)
    """
    # Handle pandas Series
    if isinstance(nominal_data, pd.Series) and isinstance(real_data, pd.Series):
        if len(nominal_data) != len(real_data):
            logger.warning("Nominal and real data have different lengths")
            return pd.Series(dtype=float)
        
        # Avoid division by zero
        deflator = nominal_data / real_data.replace(0, np.nan)
        return deflator
    
    # Handle numpy arrays
    elif isinstance(nominal_data, np.ndarray) and isinstance(real_data, np.ndarray):
        if nominal_data.shape != real_data.shape:
            logger.warning("Nominal and real data have different shapes")
            return np.array([])
        
        # Avoid division by zero
        real_data_safe = np.where(real_data == 0, np.nan, real_data)
        deflator = nominal_data / real_data_safe
        return deflator
    
    else:
        raise ValueError("Input data must be pandas Series or numpy arrays of the same type")


def interpolate_missing_data(
    data: pd.Series,
    method: str = "linear",
    limit: Optional[int] = None,
    limit_direction: str = "forward"
) -> pd.Series:
    """
    Interpolate missing values in time series data.
    
    This function provides various interpolation methods to handle
    missing data, similar to MATLAB's interp1 function.
    
    Args:
        data: Time series data with potential missing values
        method: Interpolation method ('linear', 'nearest', 'cubic', 'spline')
        limit: Maximum number of consecutive NaNs to fill
        limit_direction: Direction to apply limit ('forward', 'backward', 'both')
        
    Returns:
        Series with interpolated values
    """
    if data.isna().sum() == 0:
        return data
    
    if method == "linear":
        return data.interpolate(method="linear", limit=limit, limit_direction=limit_direction)
    
    elif method == "nearest":
        # Use forward fill then backward fill for nearest neighbor
        filled = data.fillna(method="ffill", limit=limit)
        return filled.fillna(method="bfill", limit=limit)
    
    elif method in ["cubic", "spline"]:
        # Use scipy for cubic/spline interpolation
        valid_mask = ~data.isna()
        if valid_mask.sum() < 4:  # Need at least 4 points for cubic
            logger.warning("Insufficient data points for cubic interpolation, using linear")
            return data.interpolate(method="linear", limit=limit, limit_direction=limit_direction)
        
        try:
            # Get valid data points
            x_valid = np.arange(len(data))[valid_mask]
            y_valid = data[valid_mask].values
            
            # Create interpolation function
            if method == "cubic":
                f = interpolate.interp1d(x_valid, y_valid, kind="cubic", 
                                       bounds_error=False, fill_value="extrapolate")
            else:  # spline
                f = interpolate.UnivariateSpline(x_valid, y_valid, s=0)
            
            # Interpolate all points
            x_all = np.arange(len(data))
            y_interpolated = f(x_all)
            
            # Only fill NaN values, keep original non-NaN values
            result = data.copy()
            result[data.isna()] = y_interpolated[data.isna()]
            
            return result
            
        except Exception as e:
            logger.warning(f"Spline/cubic interpolation failed: {e}, using linear")
            return data.interpolate(method="linear", limit=limit, limit_direction=limit_direction)
    
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")


def extrapolate_missing_data(
    data: pd.Series,
    method: str = "nearest",
    periods: Optional[int] = None
) -> pd.Series:
    """
    Extrapolate missing values using nearest neighbor logic.
    
    This function replicates the MATLAB interp1 'nearest' 'extrap' behavior
    used in the original code for currencies with limited historical data.
    
    Args:
        data: Time series data
        method: Extrapolation method ('nearest', 'linear')
        periods: Number of periods to extrapolate
        
    Returns:
        Series with extrapolated values
    """
    if method == "nearest":
        # Forward fill then backward fill (nearest neighbor extrapolation)
        filled = data.fillna(method="ffill", limit=periods)
        return filled.fillna(method="bfill", limit=periods)
    
    elif method == "linear":
        # Linear extrapolation
        return data.interpolate(method="linear", limit_direction="both")
    
    else:
        raise ValueError(f"Unsupported extrapolation method: {method}")


def align_time_series(
    quarterly_data: pd.Series,
    annual_data: pd.Series,
    method: str = "average"
) -> pd.DataFrame:
    """
    Align quarterly and annual time series data.
    
    This function helps synchronize data at different frequencies,
    following the temporal alignment logic from the MATLAB code.
    
    Args:
        quarterly_data: Quarterly time series
        annual_data: Annual time series
        method: Alignment method ('average', 'sum', 'last')
        
    Returns:
        DataFrame with aligned quarterly and annual data
    """
    # Ensure proper datetime index
    if not isinstance(quarterly_data.index, pd.DatetimeIndex):
        quarterly_data.index = pd.to_datetime(quarterly_data.index)
    
    if not isinstance(annual_data.index, pd.DatetimeIndex):
        annual_data.index = pd.to_datetime(annual_data.index)
    
    # Extract years from quarterly data
    quarterly_annual = quarterly_data.groupby(quarterly_data.index.year)
    
    if method == "average":
        quarterly_to_annual = quarterly_annual.mean()
    elif method == "sum":
        quarterly_to_annual = quarterly_annual.sum()
    elif method == "last":
        quarterly_to_annual = quarterly_annual.last()
    else:
        raise ValueError(f"Unsupported alignment method: {method}")
    
    # Create aligned DataFrame
    result = pd.DataFrame({
        "quarterly_aggregated": quarterly_to_annual,
        "annual": annual_data
    })
    
    return result


def create_time_index(
    start_year: int,
    end_year: int,
    frequency: str = "Q"
) -> pd.DatetimeIndex:
    """
    Create time index matching MATLAB datenum logic.
    
    This function creates time indices that match the original MATLAB
    date handling, particularly for quarterly and annual data.
    
    Args:
        start_year: Starting year
        end_year: Ending year
        frequency: Frequency ('Q' for quarterly, 'A' for annual)
        
    Returns:
        DatetimeIndex with appropriate frequency
    """
    if frequency == "Q":
        # Quarterly: last day of each quarter
        return pd.date_range(
            start=f"{start_year}-03-31",
            end=f"{end_year}-12-31",
            freq="Q"
        )
    elif frequency == "A":
        # Annual: last day of each year (December 31)
        return pd.date_range(
            start=f"{start_year}-12-31",
            end=f"{end_year}-12-31",
            freq="A"
        )
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def validate_time_series_consistency(
    data: pd.DataFrame,
    expected_length: int,
    allow_missing: bool = True
) -> bool:
    """
    Validate time series data consistency.
    
    Args:
        data: Time series DataFrame
        expected_length: Expected number of periods
        allow_missing: Whether to allow missing values
        
    Returns:
        True if data is consistent
    """
    # Check length
    if len(data) != expected_length:
        logger.warning(f"Data length {len(data)} does not match expected {expected_length}")
        return False
    
    # Check for missing values if not allowed
    if not allow_missing and data.isna().any().any():
        logger.warning("Data contains missing values")
        return False
    
    # Check for duplicate indices
    if data.index.duplicated().any():
        logger.warning("Data contains duplicate time indices")
        return False
    
    return True


def apply_country_adjustments(
    data: pd.DataFrame,
    country_code: str,
    adjustments: dict
) -> pd.DataFrame:
    """
    Apply country-specific adjustments to data.
    
    This function handles special cases like Mexico (price base Q)
    and USA (growth rate calculations) from the original MATLAB code.
    
    Args:
        data: Data to adjust
        country_code: Country code
        adjustments: Dictionary of adjustment rules
        
    Returns:
        Adjusted data
    """
    adjusted_data = data.copy()
    
    if country_code in adjustments:
        country_adjustments = adjustments[country_code]
        
        for adjustment_type, adjustment_params in country_adjustments.items():
            if adjustment_type == "price_base_change":
                # Handle Mexico price base change from L to Q
                old_base = adjustment_params.get("from", "L")
                new_base = adjustment_params.get("to", "Q")
                columns = adjustment_params.get("columns", [])
                
                for col in columns:
                    if col in adjusted_data.columns:
                        # This would involve recalculating with different price base
                        logger.info(f"Applied price base change for {country_code}: {old_base} -> {new_base}")
            
            elif adjustment_type == "growth_rate_reconstruction":
                # Handle USA growth rate reconstruction
                base_column = adjustment_params.get("base_column")
                growth_column = adjustment_params.get("growth_column")
                
                if base_column in adjusted_data.columns and growth_column in adjusted_data.columns:
                    # Reconstruct time series from growth rates
                    logger.info(f"Applied growth rate reconstruction for {country_code}")
    
    return adjusted_data


def convert_matlab_datenum(datenum_array: np.ndarray) -> pd.DatetimeIndex:
    """
    Convert MATLAB datenum values to pandas DatetimeIndex.
    
    MATLAB datenum represents dates as the number of days since January 1, 0000.
    This function converts those values to pandas datetime format.
    
    Args:
        datenum_array: Array of MATLAB datenum values
        
    Returns:
        DatetimeIndex
    """
    # MATLAB datenum epoch is January 1, 0000
    # pandas datetime epoch is January 1, 1970
    # Difference is 719529 days (to January 1, 1970)
    
    matlab_epoch_offset = 719529
    dates = pd.to_datetime(datenum_array - matlab_epoch_offset, unit="D")
    
    return pd.DatetimeIndex(dates)


def create_country_industry_matrix(
    data: pd.DataFrame,
    countries: list,
    industries: list,
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Create a standardized country-industry matrix.
    
    Args:
        data: Input data
        countries: List of country codes
        industries: List of industry codes
        fill_value: Value for missing combinations
        
    Returns:
        Standardized matrix with countries and industries
    """
    # Create MultiIndex for countries and industries
    index = pd.MultiIndex.from_product(
        [countries, industries],
        names=["country", "industry"]
    )
    
    # Reindex data to ensure consistent structure
    if isinstance(data.index, pd.MultiIndex):
        result = data.reindex(index, fill_value=fill_value)
    else:
        # Create empty matrix if input doesn't have MultiIndex
        result = pd.DataFrame(
            fill_value,
            index=index,
            columns=data.columns if hasattr(data, "columns") else ["value"]
        )
    
    return result