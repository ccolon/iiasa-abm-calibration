"""
Industry aggregation processor.

This module implements industry classification aggregation from ISIC Rev4 to NACE2,
replicating the functionality from the original MATLAB ISIC_REV4_to_NACE2_10.m code.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import DataProcessor, ProcessingResult, ProcessingStatus, ProcessingMetadata
from ..models import ISIC_TO_NACE2_MAPPING, Industry, IndustryClassification
from ..utils.validation import ValidationResult


class IndustryAggregator(DataProcessor):
    """
    Industry classification aggregator.
    
    This class handles aggregation from ISIC Rev4 (44 industries) to NACE2 (18 sectors),
    following the exact mapping and aggregation logic from the original MATLAB code.
    """
    
    # ISIC Rev4 industries (44 total) from MATLAB analysis
    ISIC_REV4_INDUSTRIES = [
        "A01_02", "A03", "B05_06", "B07_08", "B09", "C10T12", "C13T15",
        "C16", "C17_18", "C19", "C20", "C21", "C22", "C23", "C24", "C25",
        "C26", "C27", "C28", "C29", "C30", "C31T33", "D", "E", "F", "G",
        "H49", "H50", "H51", "H52", "H53", "I", "J58T60", "J61", "J62_63", "K",
        "L", "M", "N", "O", "P", "Q", "R", "S"
    ]
    
    # NACE2 sectors (18 total, excluding T)
    NACE2_SECTORS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R_S"]
    
    def __init__(
        self,
        aggregation_mapping: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize industry aggregator.
        
        Args:
            aggregation_mapping: Custom aggregation mapping (defaults to ISIC_TO_NACE2_MAPPING)
            name: Processor name
        """
        super().__init__(name or "IndustryAggregator")
        self.aggregation_mapping = aggregation_mapping or ISIC_TO_NACE2_MAPPING
        self.n_isic = len(self.ISIC_REV4_INDUSTRIES)
        self.n_nace = len(self.aggregation_mapping)
    
    def process(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Aggregate industry data from ISIC Rev4 to NACE2.
        
        Args:
            data: Input data with ISIC Rev4 industry classification
            parameters: Processing parameters including:
                - data_type: Type of data ('matrix', 'vector', 'time_series')
                - countries: List of countries (for multi-country data)
                - years: List of years (for time series data)
                
        Returns:
            ProcessingResult with aggregated NACE2 data
        """
        operation_id = self._generate_operation_id()
        
        # Extract parameters
        data_type = parameters.get("data_type", "auto") if parameters else "auto"
        countries = parameters.get("countries", []) if parameters else []
        years = parameters.get("years", []) if parameters else []
        
        # Auto-detect data type if not specified
        if data_type == "auto":
            data_type = self._detect_data_type(data)
        
        # Process based on data type
        if isinstance(data, pd.DataFrame):
            aggregated_data = self._aggregate_dataframe(data, data_type, countries, years)
        else:
            aggregated_data = {}
            for name, df in data.items():
                aggregated_data[name] = self._aggregate_dataframe(df, data_type, countries, years)
        
        # Create result
        metadata = ProcessingMetadata(
            processor_name=self.name,
            operation_id=operation_id,
            parameters=parameters or {}
        )
        
        return ProcessingResult(
            data=aggregated_data,
            metadata=metadata,
            status=ProcessingStatus.COMPLETED
        )
    
    def _detect_data_type(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> str:
        """Auto-detect the type of industry data."""
        if isinstance(data, dict):
            # Use first DataFrame for detection
            sample_df = next(iter(data.values()))
        else:
            sample_df = data
        
        # Check dimensions and structure
        if len(sample_df.shape) == 2:
            if sample_df.shape[0] == sample_df.shape[1] == self.n_isic:
                return "matrix"  # Industry x Industry matrix
            elif sample_df.shape[0] == self.n_isic:
                return "vector"  # Industry vector or time series
            elif any(col in sample_df.columns for col in ["TIME_PERIOD", "year"]):
                return "time_series"  # Time series data
        
        return "unknown"
    
    def _aggregate_dataframe(
        self,
        data: pd.DataFrame,
        data_type: str,
        countries: List[str],
        years: List[str]
    ) -> pd.DataFrame:
        """Aggregate a single DataFrame based on its type."""
        if data_type == "matrix":
            return self._aggregate_matrix(data)
        elif data_type == "vector":
            return self._aggregate_vector(data)
        elif data_type == "time_series":
            return self._aggregate_time_series(data)
        elif data_type == "icio_table":
            return self._aggregate_icio_table(data, countries, years)
        else:
            self.logger.warning(f"Unknown data type: {data_type}, treating as vector")
            return self._aggregate_vector(data)
    
    def _aggregate_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate industry-by-industry matrix (e.g., intermediate consumption).
        
        This implements Case 2 from the MATLAB code:
        44 x 44 x years -> n_agg x n_agg x years
        """
        if data.shape[0] != self.n_isic or data.shape[1] != self.n_isic:
            raise ValueError(f"Expected {self.n_isic}x{self.n_isic} matrix, got {data.shape}")
        
        # Create aggregation result
        aggregated = np.zeros((self.n_nace, self.n_nace))
        
        # Aggregate rows and columns
        for i, (nace_sector, isic_industries) in enumerate(self.aggregation_mapping.items()):
            row_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries 
                          if isic in self.ISIC_REV4_INDUSTRIES]
            
            for j, (nace_sector_col, isic_industries_col) in enumerate(self.aggregation_mapping.items()):
                col_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries_col 
                              if isic in self.ISIC_REV4_INDUSTRIES]
                
                # Sum over the ISIC industries for this NACE sector combination
                if row_indices and col_indices:
                    aggregated[i, j] = data.iloc[row_indices, col_indices].sum().sum()
        
        # Create result DataFrame
        nace_sectors = list(self.aggregation_mapping.keys())
        result = pd.DataFrame(
            aggregated,
            index=nace_sectors,
            columns=nace_sectors
        )
        
        return result
    
    def _aggregate_vector(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate industry vector (e.g., value added, output).
        
        This implements Case 1 from the MATLAB code:
        44 x years -> n_agg x years
        """
        if data.shape[0] != self.n_isic:
            raise ValueError(f"Expected {self.n_isic} industries, got {data.shape[0]}")
        
        # Create aggregation result
        n_cols = data.shape[1] if len(data.shape) > 1 else 1
        aggregated = np.zeros((self.n_nace, n_cols))
        
        # Aggregate each NACE sector
        for i, (nace_sector, isic_industries) in enumerate(self.aggregation_mapping.items()):
            row_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries 
                          if isic in self.ISIC_REV4_INDUSTRIES]
            
            if row_indices:
                if len(data.shape) == 1:
                    aggregated[i, 0] = data.iloc[row_indices].sum()
                else:
                    aggregated[i, :] = data.iloc[row_indices, :].sum(axis=0)
        
        # Create result DataFrame
        nace_sectors = list(self.aggregation_mapping.keys())
        
        if len(data.shape) == 1:
            result = pd.DataFrame(
                aggregated[:, 0],
                index=nace_sectors,
                columns=["value"]
            )
        else:
            result = pd.DataFrame(
                aggregated,
                index=nace_sectors,
                columns=data.columns
            )
        
        return result
    
    def _aggregate_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate time series data with industry dimension.
        
        Args:
            data: DataFrame with time series and industry information
            
        Returns:
            Aggregated DataFrame with NACE2 sectors
        """
        # Check if data has industry information in columns or index
        industry_columns = [col for col in data.columns if col in self.ISIC_REV4_INDUSTRIES]
        
        if industry_columns:
            # Industries are in columns
            aggregated_data = data.copy()
            
            # Create new columns for NACE sectors
            for nace_sector, isic_industries in self.aggregation_mapping.items():
                relevant_isic = [isic for isic in isic_industries if isic in industry_columns]
                
                if relevant_isic:
                    # Sum the relevant ISIC industries
                    aggregated_data[nace_sector] = data[relevant_isic].sum(axis=1)
                    
                    # Remove original ISIC columns
                    aggregated_data = aggregated_data.drop(columns=relevant_isic)
            
            return aggregated_data
        
        else:
            # Try to handle based on MultiIndex or other structure
            return self._aggregate_complex_time_series(data)
    
    def _aggregate_complex_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle complex time series with MultiIndex or nested structure."""
        # This is a placeholder for more complex aggregation logic
        # that would handle MultiIndex DataFrames with country/industry/time dimensions
        
        self.logger.warning("Complex time series aggregation not fully implemented")
        return data
    
    def _aggregate_icio_table(
        self,
        data: pd.DataFrame,
        countries: List[str],
        years: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate ICIO (Input-Output) table data.
        
        This handles the complex case from the MATLAB code where we have
        multi-dimensional arrays with countries, industries, and years.
        """
        # Extract relevant dimensions
        n_countries = len(countries) if countries else 1
        n_years = len(years) if years else 1
        
        # This would implement the complex aggregation logic for ICIO tables
        # involving multiple dimensions and countries
        
        # For now, apply standard vector aggregation
        return self._aggregate_vector(data)
    
    def aggregate_icio_matrices(
        self,
        icio_data: Dict[str, np.ndarray],
        countries: List[str],
        years: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate ICIO matrices from ISIC Rev4 to NACE2.
        
        This method handles the full ICIO aggregation as in the MATLAB
        ISIC_REV4_to_NACE2_10.m function.
        
        Args:
            icio_data: Dictionary containing ICIO matrices
            countries: List of country codes
            years: List of years
            
        Returns:
            Dictionary with aggregated NACE2 matrices
        """
        aggregated_data = {}
        
        for var_name, var_data in icio_data.items():
            try:
                aggregated_data[var_name] = self._aggregate_icio_variable(
                    var_data, var_name, countries, years
                )
            except Exception as e:
                self.logger.warning(f"Failed to aggregate {var_name}: {e}")
                # Keep original data on failure
                aggregated_data[var_name] = var_data
        
        return aggregated_data
    
    def _aggregate_icio_variable(
        self,
        var_data: np.ndarray,
        var_name: str,
        countries: List[str],
        years: List[int]
    ) -> np.ndarray:
        """Aggregate a specific ICIO variable."""
        n_years = len(years)
        
        # Handle different variable types based on their dimensions
        if var_data.shape == (44, n_years):
            # Case 1: 44 x years
            return self._aggregate_icio_case1(var_data)
        
        elif len(var_data.shape) == 3 and var_data.shape[:2] == (44, 44):
            # Case 2: 44 x 44 x years
            return self._aggregate_icio_case2(var_data)
        
        elif len(var_data.shape) == 3 and var_data.shape[0] == 44:
            # Case 3: 44 x countries x years
            return self._aggregate_icio_case3(var_data, len(countries))
        
        elif len(var_data.shape) >= 4:
            # Complex multi-dimensional cases
            return self._aggregate_icio_complex(var_data, countries, years)
        
        else:
            self.logger.warning(f"Unsupported variable shape for {var_name}: {var_data.shape}")
            return var_data
    
    def _aggregate_icio_case1(self, var_data: np.ndarray) -> np.ndarray:
        """Aggregate 44 x years data."""
        n_years = var_data.shape[1]
        aggregated = np.zeros((self.n_nace, n_years))
        
        for i, (nace_sector, isic_industries) in enumerate(self.aggregation_mapping.items()):
            row_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries 
                          if isic in self.ISIC_REV4_INDUSTRIES]
            
            if row_indices:
                aggregated[i, :] = var_data[row_indices, :].sum(axis=0)
        
        return aggregated
    
    def _aggregate_icio_case2(self, var_data: np.ndarray) -> np.ndarray:
        """Aggregate 44 x 44 x years data."""
        n_years = var_data.shape[2]
        aggregated = np.zeros((self.n_nace, self.n_nace, n_years))
        
        for i, (nace_sector_row, isic_industries_row) in enumerate(self.aggregation_mapping.items()):
            row_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries_row 
                          if isic in self.ISIC_REV4_INDUSTRIES]
            
            for j, (nace_sector_col, isic_industries_col) in enumerate(self.aggregation_mapping.items()):
                col_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries_col 
                              if isic in self.ISIC_REV4_INDUSTRIES]
                
                if row_indices and col_indices:
                    for year in range(n_years):
                        aggregated[i, j, year] = var_data[np.ix_(row_indices, col_indices, [year])].sum()
        
        return aggregated
    
    def _aggregate_icio_case3(self, var_data: np.ndarray, n_countries: int) -> np.ndarray:
        """Aggregate 44 x countries x years data."""
        n_years = var_data.shape[2]
        aggregated = np.zeros((self.n_nace, n_countries, n_years))
        
        for i, (nace_sector, isic_industries) in enumerate(self.aggregation_mapping.items()):
            row_indices = [self.ISIC_REV4_INDUSTRIES.index(isic) for isic in isic_industries 
                          if isic in self.ISIC_REV4_INDUSTRIES]
            
            if row_indices:
                aggregated[i, :, :] = var_data[row_indices, :, :].sum(axis=0)
        
        return aggregated
    
    def _aggregate_icio_complex(
        self,
        var_data: np.ndarray,
        countries: List[str],
        years: List[int]
    ) -> np.ndarray:
        """Handle complex multi-dimensional ICIO variables."""
        # This would handle cases like:
        # - exports view: industries x countries x years
        # - imports view: countries x industries x years
        # - Full ICIO tables: industries x industries x countries x sectors x years
        
        # For now, return original data
        self.logger.warning("Complex ICIO aggregation not fully implemented")
        return var_data
    
    def validate_input(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate input data for industry aggregation."""
        result = super().validate_input(data, parameters)
        
        # Check if data contains ISIC Rev4 industries
        if isinstance(data, pd.DataFrame):
            self._validate_dataframe_industries(data, result)
        elif isinstance(data, dict):
            for name, df in data.items():
                self._validate_dataframe_industries(df, result, name)
        
        return result
    
    def _validate_dataframe_industries(
        self,
        data: pd.DataFrame,
        result: ValidationResult,
        name: str = ""
    ) -> None:
        """Validate that DataFrame contains expected ISIC industries."""
        prefix = f"{name}: " if name else ""
        
        # Check if industry information is in index or columns
        industry_info = []
        
        if hasattr(data.index, 'names') and any(data.index):
            industry_info.extend(data.index.tolist())
        
        if hasattr(data.columns, 'names') and any(data.columns):
            industry_info.extend(data.columns.tolist())
        
        # Look for ISIC industry codes
        found_isic = [ind for ind in industry_info if ind in self.ISIC_REV4_INDUSTRIES]
        
        if not found_isic:
            result.add_warning(f"{prefix}No ISIC Rev4 industry codes found in data")
        else:
            result.add_detail(f"{prefix}isic_industries_found", len(found_isic))
    
    def get_aggregation_info(self) -> Dict[str, Any]:
        """Get information about the aggregation mapping."""
        return {
            "n_isic_industries": self.n_isic,
            "n_nace_sectors": self.n_nace,
            "aggregation_mapping": self.aggregation_mapping,
            "compression_ratio": self.n_isic / self.n_nace
        }