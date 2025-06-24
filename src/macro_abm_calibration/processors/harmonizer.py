"""
Data harmonization processor.

This module implements data harmonization logic that aligns quarterly and annual data,
calculates deflators, and applies country-specific adjustments following the original
MATLAB code patterns.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import DataProcessor, ProcessingResult, ProcessingStatus, ProcessingMetadata
from .utils import (
    calculate_deflator, interpolate_missing_data, align_time_series,
    apply_country_adjustments, validate_time_series_consistency
)
from ..models import Country, FrequencyType
from ..utils.validation import ValidationResult


class DataHarmonizer(DataProcessor):
    """
    Data harmonization processor.
    
    This class handles data harmonization tasks including:
    - Aligning quarterly and annual data
    - Calculating deflators from nominal and real series
    - Applying country-specific adjustments
    - Handling missing data interpolation
    """
    
    # Country-specific adjustments from MATLAB analysis
    COUNTRY_ADJUSTMENTS = {
        "MEX": {
            "price_base_change": {
                "from": "L",
                "to": "Q",
                "columns": ["real_gdp", "real_consumption", "real_investment", "real_exports", "real_imports"]
            }
        },
        "USA": {
            "growth_rate_reconstruction": {
                "variables": ["government_consumption"],
                "method": "backward_calculation"
            }
        }
    }
    
    def __init__(
        self,
        default_interpolation_method: str = "linear",
        handle_missing_data: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize data harmonizer.
        
        Args:
            default_interpolation_method: Default method for interpolating missing data
            handle_missing_data: Whether to automatically handle missing data
            name: Processor name
        """
        super().__init__(name or "DataHarmonizer")
        self.default_interpolation_method = default_interpolation_method
        self.handle_missing_data = handle_missing_data
    
    def process(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Harmonize economic data.
        
        Args:
            data: Input data to harmonize
            parameters: Processing parameters including:
                - countries: List of country codes
                - variables: List of variable names to process
                - calculate_deflators: Whether to calculate deflators
                - apply_adjustments: Whether to apply country adjustments
                - interpolation_method: Method for missing data
                
        Returns:
            ProcessingResult with harmonized data
        """
        operation_id = self._generate_operation_id()
        
        # Extract parameters
        countries = parameters.get("countries", []) if parameters else []
        variables = parameters.get("variables", []) if parameters else []
        calculate_deflators = parameters.get("calculate_deflators", True) if parameters else True
        apply_adjustments = parameters.get("apply_adjustments", True) if parameters else True
        interpolation_method = parameters.get("interpolation_method", self.default_interpolation_method) if parameters else self.default_interpolation_method
        
        # Process data
        if isinstance(data, pd.DataFrame):
            harmonized_data = self._harmonize_dataframe(
                data, countries, variables, calculate_deflators, 
                apply_adjustments, interpolation_method
            )
        else:
            harmonized_data = {}
            for name, df in data.items():
                harmonized_data[name] = self._harmonize_dataframe(
                    df, countries, variables, calculate_deflators,
                    apply_adjustments, interpolation_method
                )
        
        # Create result
        metadata = ProcessingMetadata(
            processor_name=self.name,
            operation_id=operation_id,
            parameters=parameters or {}
        )
        
        return ProcessingResult(
            data=harmonized_data,
            metadata=metadata,
            status=ProcessingStatus.COMPLETED
        )
    
    def _harmonize_dataframe(
        self,
        data: pd.DataFrame,
        countries: List[str],
        variables: List[str],
        calculate_deflators: bool,
        apply_adjustments: bool,
        interpolation_method: str
    ) -> pd.DataFrame:
        """Harmonize a single DataFrame."""
        harmonized = data.copy()
        
        # Handle missing data
        if self.handle_missing_data:
            harmonized = self._handle_missing_data(harmonized, interpolation_method)
        
        # Calculate deflators
        if calculate_deflators:
            harmonized = self._calculate_deflators(harmonized, variables)
        
        # Apply country-specific adjustments
        if apply_adjustments and countries:
            harmonized = self._apply_country_adjustments(harmonized, countries)
        
        # Align time series data
        harmonized = self._align_time_series_data(harmonized)
        
        return harmonized
    
    def _handle_missing_data(
        self,
        data: pd.DataFrame,
        interpolation_method: str
    ) -> pd.DataFrame:
        """Handle missing data using interpolation."""
        result = data.copy()
        
        # Find numeric columns with missing data
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        missing_cols = [col for col in numeric_cols if result[col].isna().any()]
        
        if missing_cols:
            self.logger.info(f"Interpolating missing data in {len(missing_cols)} columns")
            
            for col in missing_cols:
                try:
                    result[col] = interpolate_missing_data(
                        result[col], 
                        method=interpolation_method
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to interpolate {col}: {e}")
        
        return result
    
    def _calculate_deflators(
        self,
        data: pd.DataFrame,
        variables: List[str]
    ) -> pd.DataFrame:
        """
        Calculate deflators from nominal and real variables.
        
        This function replicates the deflator calculation logic from the
        original MATLAB code.
        """
        result = data.copy()
        
        # Define variable pairs for deflator calculation
        deflator_pairs = [
            ("nominal_gdp", "real_gdp", "gdp_deflator"),
            ("nominal_household_consumption", "real_household_consumption", "household_consumption_deflator"),
            ("nominal_government_consumption", "real_government_consumption", "government_consumption_deflator"),
            ("nominal_final_consumption", "real_final_consumption", "final_consumption_deflator"),
            ("nominal_fixed_capitalformation", "real_fixed_capitalformation", "fixed_capitalformation_deflator"),
            ("nominal_exports", "real_exports", "exports_deflator"),
            ("nominal_imports", "real_imports", "imports_deflator"),
        ]
        
        # Calculate deflators for each pair
        for nominal_col, real_col, deflator_col in deflator_pairs:
            if nominal_col in result.columns and real_col in result.columns:
                try:
                    # Calculate deflator using utility function
                    result[deflator_col] = calculate_deflator(
                        result[nominal_col], 
                        result[real_col]
                    )
                    
                    self.logger.debug(f"Calculated deflator: {deflator_col}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to calculate deflator {deflator_col}: {e}")
        
        return result
    
    def _apply_country_adjustments(
        self,
        data: pd.DataFrame,
        countries: List[str]
    ) -> pd.DataFrame:
        """Apply country-specific adjustments."""
        result = data.copy()
        
        # Check if data has country information
        if "REF_AREA" not in result.columns:
            return result
        
        # Apply adjustments for each country
        for country in countries:
            if country in self.COUNTRY_ADJUSTMENTS:
                country_mask = result["REF_AREA"] == country
                country_data = result[country_mask].copy()
                
                if not country_data.empty:
                    adjusted_data = apply_country_adjustments(
                        country_data, 
                        country, 
                        self.COUNTRY_ADJUSTMENTS
                    )
                    result.loc[country_mask] = adjusted_data
                    
                    self.logger.info(f"Applied adjustments for {country}")
        
        return result
    
    def _align_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align quarterly and annual time series data."""
        result = data.copy()
        
        # Check if data has time period information
        if "TIME_PERIOD" not in result.columns:
            return result
        
        # Separate quarterly and annual data
        quarterly_mask = result["TIME_PERIOD"].str.contains("Q", na=False)
        annual_mask = ~quarterly_mask & result["TIME_PERIOD"].str.match(r"^\d{4}$", na=False)
        
        quarterly_data = result[quarterly_mask]
        annual_data = result[annual_mask]
        
        if quarterly_data.empty or annual_data.empty:
            return result
        
        # Find common variables
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in quarterly_data.columns and col in annual_data.columns:
                try:
                    # Create time series for alignment
                    quarterly_series = pd.Series(
                        quarterly_data[col].values,
                        index=pd.to_datetime(quarterly_data["TIME_PERIOD"], errors="coerce")
                    )
                    
                    annual_series = pd.Series(
                        annual_data[col].values,
                        index=pd.to_datetime(annual_data["TIME_PERIOD"], errors="coerce")
                    )
                    
                    # Align and validate consistency
                    aligned = align_time_series(quarterly_series, annual_series)
                    
                    # Check for significant discrepancies
                    if "quarterly_aggregated" in aligned.columns and "annual" in aligned.columns:
                        diff_pct = abs(
                            (aligned["quarterly_aggregated"] - aligned["annual"]) / 
                            aligned["annual"].replace(0, np.nan)
                        )
                        
                        if diff_pct.max() > 0.1:  # 10% threshold
                            self.logger.warning(
                                f"Large discrepancy in {col}: max {diff_pct.max():.2%}"
                            )
                
                except Exception as e:
                    self.logger.warning(f"Failed to align time series for {col}: {e}")
        
        return result
    
    def harmonize_quarterly_annual(
        self,
        quarterly_data: pd.DataFrame,
        annual_data: pd.DataFrame,
        method: str = "average"
    ) -> pd.DataFrame:
        """
        Harmonize quarterly and annual datasets.
        
        Args:
            quarterly_data: Quarterly time series data
            annual_data: Annual time series data
            method: Aggregation method ('average', 'sum', 'last')
            
        Returns:
            Harmonized DataFrame with both frequencies
        """
        # Ensure proper time indexing
        if "TIME_PERIOD" in quarterly_data.columns:
            quarterly_data = quarterly_data.set_index("TIME_PERIOD")
        
        if "TIME_PERIOD" in annual_data.columns:
            annual_data = annual_data.set_index("TIME_PERIOD")
        
        # Find common columns
        common_cols = set(quarterly_data.columns) & set(annual_data.columns)
        common_cols = common_cols - {"REF_AREA"}  # Keep country info separate
        
        harmonized_data = {}
        
        for col in common_cols:
            try:
                # Extract time series
                quarterly_series = quarterly_data[col].dropna()
                annual_series = annual_data[col].dropna()
                
                # Align time series
                aligned = align_time_series(quarterly_series, annual_series, method)
                harmonized_data[col] = aligned
                
            except Exception as e:
                self.logger.warning(f"Failed to harmonize {col}: {e}")
        
        return pd.DataFrame(harmonized_data)
    
    def validate_temporal_consistency(
        self,
        data: pd.DataFrame,
        tolerance: float = 0.05
    ) -> ValidationResult:
        """
        Validate temporal consistency of time series data.
        
        Args:
            data: Time series data to validate
            tolerance: Tolerance for consistency checks
            
        Returns:
            ValidationResult with consistency information
        """
        result = ValidationResult(is_valid=True)
        
        if "TIME_PERIOD" not in data.columns:
            result.add_warning("No TIME_PERIOD column found")
            return result
        
        # Check for gaps in time series
        time_periods = pd.to_datetime(data["TIME_PERIOD"], errors="coerce")
        time_periods = time_periods.dropna().sort_values()
        
        if len(time_periods) > 1:
            # Check for regular intervals
            intervals = time_periods.diff()[1:]
            
            if not intervals.nunique() == 1:
                result.add_warning("Irregular time intervals detected")
            
            # Check for missing periods
            expected_periods = pd.date_range(
                start=time_periods.min(),
                end=time_periods.max(),
                freq=intervals.mode().iloc[0] if len(intervals) > 0 else "Q"
            )
            
            missing_periods = len(expected_periods) - len(time_periods)
            if missing_periods > 0:
                result.add_warning(f"{missing_periods} missing time periods")
        
        # Validate numeric consistency
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isna().any():
                na_count = data[col].isna().sum()
                result.add_detail(f"{col}_missing_values", na_count)
            
            # Check for extreme values
            if len(data[col].dropna()) > 0:
                q99 = data[col].quantile(0.99)
                q01 = data[col].quantile(0.01)
                
                if q99 / q01 > 1000:  # Very wide range
                    result.add_warning(f"Extreme value range in {col}")
        
        return result
    
    def create_balanced_panel(
        self,
        data: Dict[str, pd.DataFrame],
        countries: List[str],
        time_periods: List[str]
    ) -> pd.DataFrame:
        """
        Create a balanced panel dataset from country-specific data.
        
        Args:
            data: Dictionary of country DataFrames
            countries: List of country codes
            time_periods: List of time periods
            
        Returns:
            Balanced panel DataFrame
        """
        panel_data = []
        
        for country in countries:
            if country in data:
                country_df = data[country].copy()
                country_df["REF_AREA"] = country
                
                # Ensure all time periods are present
                country_df = country_df.set_index("TIME_PERIOD").reindex(time_periods).reset_index()
                country_df["REF_AREA"] = country
                
                panel_data.append(country_df)
        
        if panel_data:
            result = pd.concat(panel_data, ignore_index=True)
            return result.sort_values(["REF_AREA", "TIME_PERIOD"])
        else:
            return pd.DataFrame()
    
    def validate_input(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate input data for harmonization."""
        result = super().validate_input(data, parameters)
        
        # Check for time series structure
        if isinstance(data, pd.DataFrame):
            self._validate_time_series_structure(data, result)
        elif isinstance(data, dict):
            for name, df in data.items():
                self._validate_time_series_structure(df, result, name)
        
        return result
    
    def _validate_time_series_structure(
        self,
        data: pd.DataFrame,
        result: ValidationResult,
        name: str = ""
    ) -> None:
        """Validate time series structure."""
        prefix = f"{name}: " if name else ""
        
        if "TIME_PERIOD" not in data.columns:
            result.add_warning(f"{prefix}No TIME_PERIOD column found")
        
        # Check for required economic variables
        required_vars = ["nominal_gdp", "real_gdp"]
        missing_vars = [var for var in required_vars if var not in data.columns]
        
        if missing_vars:
            result.add_warning(f"{prefix}Missing required variables: {missing_vars}")
        
        # Check data completeness
        if not data.empty:
            completeness = 1 - (data.isna().sum().sum() / data.size)
            result.add_detail(f"{prefix}data_completeness", completeness)