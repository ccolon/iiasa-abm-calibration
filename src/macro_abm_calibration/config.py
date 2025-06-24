"""
Configuration management for the macroeconomic ABM calibration system.

This module provides comprehensive configuration management using Pydantic
for validation, type checking, and environment variable integration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from pydantic import BaseSettings, Field, validator
from pydantic.types import DirectoryPath, FilePath

from .models import Country, FrequencyType, Industry, TimeFrame, OECD_COUNTRIES, NACE2_INDUSTRIES


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""
    
    sqlite_path: Optional[FilePath] = Field(
        default=None,
        description="Path to OECD SQLite database file"
    )
    
    connection_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Database connection timeout in seconds"
    )
    
    query_timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Query execution timeout in seconds"
    )
    
    class Config:
        env_prefix = "DB_"


class DataSourceConfig(BaseSettings):
    """Configuration for external data sources."""
    
    oecd_datasets: Dict[str, str] = Field(
        default_factory=lambda: {
            "gdp_expenditure": "OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_EXPENDITURE_NATIO_CURR_1_1",
            "gdp_quarterly": "OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_1_1",
            "interest_rates": "OECD_SDD_STES_DSD_STES_DF_FINMARK_",
            "unemployment": "OECD_SDD_TPS_DSD_LFS_DF_IALFS_UNE_M_1_0",
            "economic_outlook": "OECD_ECO_MAD_DSD_EO_114_DF_EO_114_1_0",
        },
        description="OECD dataset identifiers"
    )
    
    eurostat_datasets: Dict[str, str] = Field(
        default_factory=lambda: {
            "exchange_rates_quarterly": "estat_ert_bil_eur_q_filtered_en",
            "exchange_rates_annual": "estat_ert_bil_eur_a_filtered_en",
        },
        description="Eurostat dataset identifiers"
    )
    
    icio_data_path: Optional[DirectoryPath] = Field(
        default=None,
        description="Path to OECD ICIO data files"
    )
    
    class Config:
        env_prefix = "DATA_"


class ProcessingConfig(BaseSettings):
    """Configuration for data processing parameters."""
    
    base_currency: str = Field(
        default="USD",
        description="Base currency for all conversions"
    )
    
    missing_data_strategy: str = Field(
        default="interpolate",
        regex="^(interpolate|forward_fill|drop|raise)$",
        description="Strategy for handling missing data"
    )
    
    interpolation_method: str = Field(
        default="linear",
        regex="^(linear|nearest|spline|polynomial)$",
        description="Interpolation method for missing values"
    )
    
    validation_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Tolerance for data validation checks"
    )
    
    parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing where applicable"
    )
    
    max_workers: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Maximum number of worker processes"
    )
    
    class Config:
        env_prefix = "PROC_"


class OutputConfig(BaseSettings):
    """Configuration for output generation."""
    
    output_directory: DirectoryPath = Field(
        default=Path("./output"),
        description="Directory for output files"
    )
    
    save_intermediate: bool = Field(
        default=True,
        description="Save intermediate processing results"
    )
    
    output_formats: List[str] = Field(
        default=["parquet", "csv"],
        description="Output file formats"
    )
    
    compression: str = Field(
        default="snappy",
        regex="^(snappy|gzip|brotli|lz4|zstd|none)$",
        description="Compression algorithm for output files"
    )
    
    class Config:
        env_prefix = "OUTPUT_"


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(
        default="INFO",
        regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    file_path: Optional[Path] = Field(
        default=None,
        description="Path to log file (None for console only)"
    )
    
    rotation_size: str = Field(
        default="10MB",
        description="Log file rotation size"
    )
    
    retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days to retain log files"
    )
    
    class Config:
        env_prefix = "LOG_"


class CalibrationConfig(BaseSettings):
    """
    Main configuration class for the macroeconomic ABM calibration system.
    
    This class aggregates all configuration sections and provides validation
    for the overall calibration setup.
    """
    
    # Basic configuration
    project_name: str = Field(
        default="macro-abm-calibration",
        description="Project name for identification"
    )
    
    # Country and industry selection
    countries: List[str] = Field(
        default_factory=lambda: [c.oecd_code for c in OECD_COUNTRIES],
        description="List of country codes to include in calibration"
    )
    
    industries: List[str] = Field(
        default_factory=lambda: [i.code for i in NACE2_INDUSTRIES],
        description="List of industry codes to include"
    )
    
    # Time periods
    estimation_period: TimeFrame = Field(
        default=TimeFrame(start_year=1996, end_year=2024, frequency=FrequencyType.QUARTERLY),
        description="Time period for data estimation"
    )
    
    calibration_period: TimeFrame = Field(
        default=TimeFrame(start_year=2010, end_year=2017, frequency=FrequencyType.ANNUAL),
        description="Time period for model calibration"
    )
    
    # Nested configurations
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database connection settings"
    )
    
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig,
        description="External data source configuration"
    )
    
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Data processing parameters"
    )
    
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output generation settings"
    )
    
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Advanced options
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with additional validation"
    )
    
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching of intermediate results"
    )
    
    cache_directory: Optional[DirectoryPath] = Field(
        default=None,
        description="Directory for cache files"
    )
    
    @validator('countries')
    def validate_countries(cls, v):
        """Validate that all country codes are supported."""
        valid_codes = {c.oecd_code for c in OECD_COUNTRIES}
        invalid_codes = set(v) - valid_codes
        if invalid_codes:
            raise ValueError(f"Invalid country codes: {invalid_codes}")
        return v
    
    @validator('industries')
    def validate_industries(cls, v):
        """Validate that all industry codes are supported."""
        valid_codes = {i.code for i in NACE2_INDUSTRIES}
        invalid_codes = set(v) - valid_codes
        if invalid_codes:
            raise ValueError(f"Invalid industry codes: {invalid_codes}")
        return v
    
    @validator('calibration_period')
    def validate_calibration_period(cls, v, values):
        """Ensure calibration period is within estimation period."""
        if 'estimation_period' in values:
            est_period = values['estimation_period']
            if v.start_year < est_period.start_year or v.end_year > est_period.end_year:
                raise ValueError("Calibration period must be within estimation period")
        return v
    
    @property
    def country_objects(self) -> Set[Country]:
        """Get Country objects for selected countries."""
        country_map = {c.oecd_code: c for c in OECD_COUNTRIES}
        return {country_map[code] for code in self.countries}
    
    @property
    def industry_objects(self) -> Set[Industry]:
        """Get Industry objects for selected industries."""
        industry_map = {i.code: i for i in NACE2_INDUSTRIES}
        return {industry_map[code] for code in self.industries}
    
    def get_country(self, code: str) -> Country:
        """Get Country object by code."""
        for country in OECD_COUNTRIES:
            if country.oecd_code == code:
                return country
        raise ValueError(f"Country not found: {code}")
    
    def get_industry(self, code: str) -> Industry:
        """Get Industry object by code."""
        for industry in NACE2_INDUSTRIES:
            if industry.code == code:
                return industry
        raise ValueError(f"Industry not found: {code}")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # JSON encoders for custom types
        json_encoders = {
            Path: str,
            TimeFrame: lambda tf: tf.dict(),
        }
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> CalibrationConfig:
        """Load configuration from a file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() == '.json':
            import json
            with open(config_path, 'w') as f:
                json.dump(self.dict(), f, indent=2, default=str)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(self.dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    **overrides
) -> CalibrationConfig:
    """
    Load configuration with optional file and overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        **overrides: Configuration overrides
        
    Returns:
        CalibrationConfig instance
    """
    if config_path:
        config = CalibrationConfig.from_file(config_path)
        # Apply overrides
        if overrides:
            config_dict = config.dict()
            config_dict.update(overrides)
            config = CalibrationConfig(**config_dict)
        return config
    else:
        return CalibrationConfig(**overrides)