"""
Calibration pipeline orchestrator.

This module implements the main calibration pipeline that orchestrates the entire
data processing workflow, replicating the sequence from the original MATLAB code:
a_data.m -> b_calibration_data.m -> c1_icios_data.m -> set_parameters_and_initial_conditions.m
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from .base import DataProcessor, ProcessingResult, ProcessingStatus, ProcessingMetadata
from .currency import CurrencyConverter
from .industry import IndustryAggregator
from .harmonizer import DataHarmonizer
from ..data_sources import DataSourceManager, DataSourceType
from ..models import Country, TimeFrame, FrequencyType
from ..config import CalibrationConfig
from ..utils.validation import ValidationResult


class CalibrationPipeline(DataProcessor):
    """
    Main calibration pipeline orchestrator.
    
    This class coordinates the entire data processing workflow, from raw data
    extraction to final calibration-ready datasets, following the sequence
    established in the original MATLAB code.
    """
    
    def __init__(
        self,
        config: CalibrationConfig,
        data_source_manager: Optional[DataSourceManager] = None,
        name: Optional[str] = None
    ):
        """
        Initialize calibration pipeline.
        
        Args:
            config: Calibration configuration
            data_source_manager: Data source manager (will create if None)
            name: Pipeline name
        """
        super().__init__(name or "CalibrationPipeline")
        self.config = config
        
        if data_source_manager:
            self.data_source_manager = data_source_manager
        else:
            self.data_source_manager = DataSourceManager.from_config(config)
        
        # Initialize processors
        self._initialize_processors()
        
        # Pipeline state
        self._intermediate_results: Dict[str, Any] = {}
        self._pipeline_metadata: Dict[str, ProcessingMetadata] = {}
    
    def _initialize_processors(self) -> None:
        """Initialize all data processors."""
        # Get data sources
        eurostat_source = self.data_source_manager.get_source(DataSourceType.EUROSTAT)
        
        # Initialize processors
        self.currency_converter = CurrencyConverter(
            eurostat_source=eurostat_source,
            target_currency=self.config.processing.base_currency
        )
        
        self.industry_aggregator = IndustryAggregator()
        
        self.data_harmonizer = DataHarmonizer(
            default_interpolation_method=self.config.processing.interpolation_method,
            handle_missing_data=self.config.processing.missing_data_strategy != "raise"
        )
    
    def process(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Run the complete calibration pipeline.
        
        Args:
            data: Input data (not used - pipeline fetches its own data)
            parameters: Pipeline parameters (optional overrides)
            
        Returns:
            ProcessingResult with final calibration datasets
        """
        operation_id = self._generate_operation_id()
        
        self.logger.info(f"Starting calibration pipeline {operation_id}")
        
        try:
            # Connect to data sources
            self.data_source_manager.connect_all()
            
            # Step 1: Extract raw data (equivalent to a_data.m)
            raw_data = self._step_1_extract_raw_data()
            
            # Step 2: Process calibration data (equivalent to b_calibration_data.m)
            calibration_data = self._step_2_process_calibration_data(raw_data)
            
            # Step 3: Process ICIO data (equivalent to c1_icios_data.m)
            icio_data = self._step_3_process_icio_data()
            
            # Step 4: Create final calibration datasets
            final_data = self._step_4_create_final_datasets(calibration_data, icio_data)
            
            # Create result
            metadata = ProcessingMetadata(
                processor_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            result = ProcessingResult(
                data=final_data,
                metadata=metadata,
                status=ProcessingStatus.COMPLETED
            )
            
            self.logger.info(f"Calibration pipeline {operation_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Calibration pipeline {operation_id} failed: {e}")
            raise
        
        finally:
            # Disconnect from data sources
            self.data_source_manager.disconnect_all()
    
    def _step_1_extract_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Step 1: Extract raw data from OECD and Eurostat sources.
        
        This step replicates the functionality of a_data.m, extracting:
        - GDP data (quarterly and annual)
        - Consumption data (household and government)
        - Investment data (fixed capital formation)
        - Trade data (exports and imports)
        - Exchange rates
        - Interest rates and unemployment
        """
        self.logger.info("Step 1: Extracting raw data")
        
        raw_data = {}
        
        # Get data sources
        oecd_source = self.data_source_manager.get_source(DataSourceType.OECD)
        eurostat_source = self.data_source_manager.get_source(DataSourceType.EUROSTAT)
        
        if not oecd_source or not eurostat_source:
            raise ValueError("OECD and Eurostat data sources are required")
        
        countries = self.config.countries
        estimation_period = self.config.estimation_period
        
        # Extract GDP data (quarterly and annual)
        self.logger.info("Extracting GDP data")
        for price_base in ["V", "L"]:  # V=current, L=constant
            for frequency in [FrequencyType.QUARTERLY, FrequencyType.ANNUAL]:
                gdp_data = oecd_source.fetch_gdp_data(
                    countries=countries,
                    start_year=estimation_period.start_year,
                    end_year=estimation_period.end_year,
                    frequency=frequency,
                    price_base=price_base
                )
                
                key = f"gdp_{frequency.value.lower()}_{price_base.lower()}"
                raw_data[key] = gdp_data
        
        # Extract consumption data
        self.logger.info("Extracting consumption data")
        for sector in ["S1M", "S13"]:  # S1M=households, S13=government
            for price_base in ["V", "L"]:
                for frequency in [FrequencyType.QUARTERLY, FrequencyType.ANNUAL]:
                    consumption_data = oecd_source.fetch_consumption_data(
                        countries=countries,
                        start_year=estimation_period.start_year,
                        end_year=estimation_period.end_year,
                        frequency=frequency,
                        sector=sector,
                        price_base=price_base
                    )
                    
                    sector_name = "household" if sector == "S1M" else "government"
                    key = f"{sector_name}_consumption_{frequency.value.lower()}_{price_base.lower()}"
                    raw_data[key] = consumption_data
        
        # Extract unemployment data
        self.logger.info("Extracting unemployment data")
        unemployment_data = oecd_source.fetch_unemployment_data(
            countries=countries,
            start_year=estimation_period.start_year,
            end_year=estimation_period.end_year,
            frequency=FrequencyType.QUARTERLY
        )
        raw_data["unemployment_quarterly"] = unemployment_data
        
        # Extract exchange rates
        self.logger.info("Extracting exchange rates")
        for frequency in [FrequencyType.QUARTERLY, FrequencyType.ANNUAL]:
            exchange_rates = eurostat_source.fetch_exchange_rates(
                countries=countries,
                start_year=estimation_period.start_year,
                end_year=estimation_period.end_year,
                frequency=frequency
            )
            raw_data[f"exchange_rates_{frequency.value.lower()}"] = exchange_rates
        
        self._intermediate_results["raw_data"] = raw_data
        self.logger.info(f"Step 1 completed: extracted {len(raw_data)} datasets")
        
        return raw_data
    
    def _step_2_process_calibration_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Step 2: Process calibration data.
        
        This step replicates b_calibration_data.m functionality:
        - Convert currencies to USD
        - Calculate deflators
        - Harmonize quarterly and annual data
        - Apply country-specific adjustments
        """
        self.logger.info("Step 2: Processing calibration data")
        
        processed_data = {}
        
        # Process each country separately
        for country_code in self.config.countries:
            self.logger.info(f"Processing calibration data for {country_code}")
            
            # Extract country-specific data
            country_data = self._extract_country_data(raw_data, country_code)
            
            # Convert currencies
            currency_params = {
                "countries": [country_code],
                "start_year": self.config.estimation_period.start_year,
                "end_year": self.config.estimation_period.end_year,
                "frequency": "Q",
                "value_columns": ["OBS_VALUE"]
            }
            
            currency_result = self.currency_converter.process_with_validation(
                country_data, currency_params
            )
            
            if not currency_result.is_success:
                self.logger.warning(f"Currency conversion failed for {country_code}")
                continue
            
            # Harmonize data
            harmonization_params = {
                "countries": [country_code],
                "calculate_deflators": True,
                "apply_adjustments": True
            }
            
            harmonization_result = self.data_harmonizer.process_with_validation(
                currency_result.data, harmonization_params
            )
            
            if harmonization_result.is_success:
                processed_data[country_code] = harmonization_result.data
                self.logger.info(f"Successfully processed calibration data for {country_code}")
            else:
                self.logger.warning(f"Data harmonization failed for {country_code}")
        
        self._intermediate_results["calibration_data"] = processed_data
        self.logger.info(f"Step 2 completed: processed {len(processed_data)} countries")
        
        return processed_data
    
    def _step_3_process_icio_data(self) -> Dict[str, Any]:
        """
        Step 3: Process ICIO data.
        
        This step replicates c1_icios_data.m functionality:
        - Load ICIO input-output tables
        - Shrink tables to target countries and industries
        - Aggregate from ISIC Rev4 to NACE2
        - Convert to USD
        """
        self.logger.info("Step 3: Processing ICIO data")
        
        # Get ICIO data source
        icio_source = self.data_source_manager.get_source(DataSourceType.ICIO)
        
        if not icio_source:
            raise ValueError("ICIO data source is required")
        
        # Process calibration years
        calibration_years = list(self.config.calibration_period.years)
        target_countries = self.config.countries + ["ROW"]  # Include ROW
        target_industries = self.config.industries
        
        # Shrink ICIO tables
        self.logger.info("Shrinking ICIO tables")
        shrunk_tables = icio_source.shrink_icio_tables(
            target_countries=self.config.countries,  # Exclude ROW for shrinking
            target_industries=target_industries,
            years=calibration_years
        )
        
        # Aggregate industries from ISIC Rev4 to NACE2
        self.logger.info("Aggregating industries to NACE2")
        aggregated_tables = self.industry_aggregator.aggregate_icio_matrices(
            shrunk_tables,
            target_countries,
            calibration_years
        )
        
        # Convert to USD (if not already converted)
        # This would involve applying exchange rates to the ICIO data
        
        self._intermediate_results["icio_data"] = aggregated_tables
        self.logger.info("Step 3 completed: processed ICIO data")
        
        return aggregated_tables
    
    def _step_4_create_final_datasets(
        self,
        calibration_data: Dict[str, pd.DataFrame],
        icio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Step 4: Create final calibration datasets.
        
        This step combines processed data into final datasets ready for
        model calibration and initialization.
        """
        self.logger.info("Step 4: Creating final datasets")
        
        final_data = {
            "country_data": calibration_data,
            "icio_data": icio_data,
            "metadata": {
                "countries": self.config.countries,
                "industries": self.config.industries,
                "estimation_period": self.config.estimation_period.dict(),
                "calibration_period": self.config.calibration_period.dict(),
                "processing_timestamp": datetime.now().isoformat()
            }
        }
        
        # Create summary statistics
        final_data["summary"] = self._create_summary_statistics(calibration_data, icio_data)
        
        # Validate final data
        validation_result = self._validate_final_data(final_data)
        final_data["validation"] = validation_result
        
        self.logger.info("Step 4 completed: created final datasets")
        
        return final_data
    
    def _extract_country_data(self, raw_data: Dict[str, pd.DataFrame], country_code: str) -> pd.DataFrame:
        """Extract data for a specific country from raw datasets."""
        country_data_list = []
        
        for dataset_name, dataset in raw_data.items():
            if "REF_AREA" in dataset.columns:
                country_subset = dataset[dataset["REF_AREA"] == country_code].copy()
                country_subset["dataset"] = dataset_name
                country_data_list.append(country_subset)
        
        if country_data_list:
            return pd.concat(country_data_list, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _create_summary_statistics(
        self,
        calibration_data: Dict[str, pd.DataFrame],
        icio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create summary statistics for the processed data."""
        summary = {
            "countries_processed": len(calibration_data),
            "icio_years_processed": len(icio_data) if isinstance(icio_data, dict) else 0,
            "data_completeness": {},
            "validation_summary": {}
        }
        
        # Calculate data completeness for each country
        for country, data in calibration_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                completeness = 1 - (data.isna().sum().sum() / data.size)
                summary["data_completeness"][country] = completeness
        
        return summary
    
    def _validate_final_data(self, final_data: Dict[str, Any]) -> ValidationResult:
        """Validate the final calibration datasets."""
        result = ValidationResult(is_valid=True)
        
        # Check country data
        country_data = final_data.get("country_data", {})
        
        if not country_data:
            result.add_error("No country data found")
        else:
            for country, data in country_data.items():
                if isinstance(data, pd.DataFrame):
                    if data.empty:
                        result.add_warning(f"Empty data for country {country}")
                    elif data.isna().all().all():
                        result.add_error(f"All NaN data for country {country}")
        
        # Check ICIO data
        icio_data = final_data.get("icio_data", {})
        
        if not icio_data:
            result.add_warning("No ICIO data found")
        
        # Add summary details
        result.add_detail("countries_with_data", len(country_data))
        result.add_detail("icio_datasets", len(icio_data))
        
        return result
    
    def run_full_pipeline(self) -> ProcessingResult:
        """
        Run the complete calibration pipeline.
        
        This is the main entry point for running the entire calibration process.
        
        Returns:
            ProcessingResult with final calibration datasets
        """
        return self.process_with_validation(
            data=pd.DataFrame(),  # Pipeline fetches its own data
            validate_input=False,  # Skip input validation
            validate_output=True
        )
    
    def run_partial_pipeline(self, steps: List[str]) -> Dict[str, Any]:
        """
        Run specific steps of the pipeline.
        
        Args:
            steps: List of step names to run
            
        Returns:
            Dictionary with results from specified steps
        """
        results = {}
        
        if "extract_raw_data" in steps:
            self.data_source_manager.connect_all()
            try:
                results["raw_data"] = self._step_1_extract_raw_data()
            finally:
                self.data_source_manager.disconnect_all()
        
        if "process_calibration_data" in steps:
            raw_data = results.get("raw_data") or self._intermediate_results.get("raw_data")
            if raw_data:
                results["calibration_data"] = self._step_2_process_calibration_data(raw_data)
        
        if "process_icio_data" in steps:
            self.data_source_manager.connect_all()
            try:
                results["icio_data"] = self._step_3_process_icio_data()
            finally:
                self.data_source_manager.disconnect_all()
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and intermediate results."""
        return {
            "config": self.config.dict(),
            "data_sources": self.data_source_manager.get_status(),
            "intermediate_results_available": list(self._intermediate_results.keys()),
            "processors": {
                "currency_converter": self.currency_converter.get_info(),
                "industry_aggregator": self.industry_aggregator.get_info(),
                "data_harmonizer": self.data_harmonizer.get_info()
            }
        }
    
    def save_intermediate_results(self, output_directory: Path) -> None:
        """Save intermediate results to files."""
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        
        for step_name, data in self._intermediate_results.items():
            if isinstance(data, dict):
                for sub_name, sub_data in data.items():
                    if isinstance(sub_data, pd.DataFrame):
                        file_path = output_directory / f"{step_name}_{sub_name}.parquet"
                        sub_data.to_parquet(file_path)
            elif isinstance(data, pd.DataFrame):
                file_path = output_directory / f"{step_name}.parquet"
                data.to_parquet(file_path)
        
        self.logger.info(f"Saved intermediate results to {output_directory}")