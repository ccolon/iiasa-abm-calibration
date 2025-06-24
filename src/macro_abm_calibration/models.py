"""
Core data models for the macroeconomic ABM calibration system.

This module defines the fundamental data structures used throughout
the calibration pipeline, including country mappings, industry
classifications, and time frame handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import pandas as pd
from pydantic import BaseModel, Field, validator


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"


@dataclass(frozen=True)
class Country:
    """
    Represents a country with its various code mappings.
    
    Attributes:
        oecd_code: 3-letter OECD country code
        eurostat_code: Eurostat country/currency code
        name: Full country name
        currency: Primary currency code
        is_oecd_member: Whether country is OECD member
        is_eurozone: Whether country uses EUR
    """
    oecd_code: str
    eurostat_code: str
    name: str
    currency: CurrencyCode
    is_oecd_member: bool = True
    is_eurozone: bool = False
    
    def __post_init__(self):
        """Validate country codes and currency consistency."""
        if len(self.oecd_code) != 3:
            raise ValueError(f"OECD code must be 3 characters: {self.oecd_code}")
        
        if self.is_eurozone and self.currency != CurrencyCode.EUR:
            raise ValueError(f"Eurozone country must use EUR: {self.oecd_code}")


class IndustryClassification(str, Enum):
    """Supported industry classification systems."""
    ISIC_REV4 = "ISIC_REV4"
    NACE2 = "NACE2"


@dataclass(frozen=True)
class Industry:
    """
    Represents an industry with classification codes.
    
    Attributes:
        code: Industry code (varies by classification)
        name: Descriptive name
        classification: Classification system used
        parent_code: Parent industry code for aggregation
        description: Detailed description
    """
    code: str
    name: str
    classification: IndustryClassification
    parent_code: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate industry code format."""
        if not self.code:
            raise ValueError("Industry code cannot be empty")


class FrequencyType(str, Enum):
    """Data frequency types."""
    ANNUAL = "A"
    QUARTERLY = "Q"
    MONTHLY = "M"


class TimeFrame(BaseModel):
    """
    Handles time periods for data processing.
    
    Attributes:
        start_year: Starting year
        end_year: Ending year (inclusive)
        frequency: Data frequency
        quarters_per_year: Number of quarters per year
    """
    start_year: int = Field(..., ge=1990, le=2030)
    end_year: int = Field(..., ge=1990, le=2030)
    frequency: FrequencyType = FrequencyType.ANNUAL
    quarters_per_year: int = Field(default=4, ge=1, le=4)
    
    @validator('end_year')
    def end_after_start(cls, v, values):
        """Ensure end year is after start year."""
        if 'start_year' in values and v < values['start_year']:
            raise ValueError('end_year must be >= start_year')
        return v
    
    @property
    def years(self) -> range:
        """Get range of years."""
        return range(self.start_year, self.end_year + 1)
    
    @property
    def num_years(self) -> int:
        """Get number of years."""
        return self.end_year - self.start_year + 1
    
    @property
    def num_periods(self) -> int:
        """Get total number of periods based on frequency."""
        if self.frequency == FrequencyType.ANNUAL:
            return self.num_years
        elif self.frequency == FrequencyType.QUARTERLY:
            return self.num_years * self.quarters_per_year
        elif self.frequency == FrequencyType.MONTHLY:
            return self.num_years * 12
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")
    
    def to_datetime_index(self) -> pd.DatetimeIndex:
        """Convert to pandas DatetimeIndex."""
        if self.frequency == FrequencyType.ANNUAL:
            return pd.date_range(
                start=f"{self.start_year}-12-31",
                end=f"{self.end_year}-12-31",
                freq="A"
            )
        elif self.frequency == FrequencyType.QUARTERLY:
            return pd.date_range(
                start=f"{self.start_year}-01-01",
                end=f"{self.end_year}-12-31",
                freq="Q"
            )
        elif self.frequency == FrequencyType.MONTHLY:
            return pd.date_range(
                start=f"{self.start_year}-01-01",
                end=f"{self.end_year}-12-31",
                freq="M"
            )
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")


class DatasetMetadata(BaseModel):
    """
    Metadata for datasets used in calibration.
    
    Attributes:
        name: Dataset name
        source: Data source (OECD, Eurostat, etc.)
        table_id: Database table identifier
        description: Dataset description
        variables: List of variable names
        last_updated: Last update timestamp
    """
    name: str
    source: str
    table_id: str
    description: str
    variables: List[str] = Field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# Standard country definitions based on MATLAB analysis
OECD_COUNTRIES = {
    Country("AUS", "AUD", "Australia", CurrencyCode.AUD),
    Country("AUT", "EUR", "Austria", CurrencyCode.EUR, is_eurozone=True),
    Country("BEL", "EUR", "Belgium", CurrencyCode.EUR, is_eurozone=True),
    Country("CAN", "CAD", "Canada", CurrencyCode.CAD),
    Country("CZE", "CZK", "Czech Republic", CurrencyCode.USD),  # Non-EUR mapped to USD in original
    Country("DNK", "DKK", "Denmark", CurrencyCode.DKK),
    Country("EST", "EUR", "Estonia", CurrencyCode.EUR, is_eurozone=True),
    Country("FIN", "EUR", "Finland", CurrencyCode.EUR, is_eurozone=True),
    Country("FRA", "EUR", "France", CurrencyCode.EUR, is_eurozone=True),
    Country("DEU", "EUR", "Germany", CurrencyCode.EUR, is_eurozone=True),
    Country("GRC", "EUR", "Greece", CurrencyCode.EUR, is_eurozone=True),
    Country("HUN", "HUF", "Hungary", CurrencyCode.USD),
    Country("IRL", "EUR", "Ireland", CurrencyCode.EUR, is_eurozone=True),
    Country("ITA", "EUR", "Italy", CurrencyCode.EUR, is_eurozone=True),
    Country("JPN", "JPY", "Japan", CurrencyCode.JPY),
    Country("KOR", "KRW", "South Korea", CurrencyCode.USD),
    Country("LVA", "EUR", "Latvia", CurrencyCode.EUR, is_eurozone=True),
    Country("LTU", "EUR", "Lithuania", CurrencyCode.EUR, is_eurozone=True),
    Country("LUX", "EUR", "Luxembourg", CurrencyCode.EUR, is_eurozone=True),
    Country("MEX", "MXN", "Mexico", CurrencyCode.USD),
    Country("NLD", "EUR", "Netherlands", CurrencyCode.EUR, is_eurozone=True),
    Country("NOR", "NOK", "Norway", CurrencyCode.NOK),
    Country("NZL", "NZD", "New Zealand", CurrencyCode.USD),
    Country("POL", "PLN", "Poland", CurrencyCode.USD),
    Country("PRT", "EUR", "Portugal", CurrencyCode.EUR, is_eurozone=True),
    Country("SVK", "EUR", "Slovakia", CurrencyCode.EUR, is_eurozone=True),
    Country("SVN", "EUR", "Slovenia", CurrencyCode.EUR, is_eurozone=True),
    Country("ESP", "EUR", "Spain", CurrencyCode.EUR, is_eurozone=True),
    Country("SWE", "SEK", "Sweden", CurrencyCode.SEK),
    Country("GBR", "GBP", "United Kingdom", CurrencyCode.GBP),
    Country("USA", "USD", "United States", CurrencyCode.USD),
}

# NACE2 Industry definitions (18 sectors, excluding T)
NACE2_INDUSTRIES = {
    Industry("A", "Agriculture, forestry and fishing", IndustryClassification.NACE2),
    Industry("B", "Mining and quarrying", IndustryClassification.NACE2),
    Industry("C", "Manufacturing", IndustryClassification.NACE2),
    Industry("D", "Electricity, gas, steam and air conditioning supply", IndustryClassification.NACE2),
    Industry("E", "Water supply; sewerage, waste management", IndustryClassification.NACE2),
    Industry("F", "Construction", IndustryClassification.NACE2),
    Industry("G", "Wholesale and retail trade", IndustryClassification.NACE2),
    Industry("H", "Transportation and storage", IndustryClassification.NACE2),
    Industry("I", "Accommodation and food service activities", IndustryClassification.NACE2),
    Industry("J", "Information and communication", IndustryClassification.NACE2),
    Industry("K", "Financial and insurance activities", IndustryClassification.NACE2),
    Industry("L", "Real estate activities", IndustryClassification.NACE2),
    Industry("M", "Professional, scientific and technical activities", IndustryClassification.NACE2),
    Industry("N", "Administrative and support service activities", IndustryClassification.NACE2),
    Industry("O", "Public administration and defence", IndustryClassification.NACE2),
    Industry("P", "Education", IndustryClassification.NACE2),
    Industry("Q", "Human health and social work activities", IndustryClassification.NACE2),
    Industry("R", "Arts, entertainment and recreation", IndustryClassification.NACE2),
    Industry("S", "Other service activities", IndustryClassification.NACE2),
}

# ISIC Rev4 to NACE2 aggregation mapping (from MATLAB analysis)
ISIC_TO_NACE2_MAPPING = {
    "A": ["A01_02", "A03"],
    "B": ["B05_06", "B07_08", "B09"],
    "C": ["C10T12", "C13T15", "C16", "C17_18", "C19", "C20", "C21", "C22", 
          "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31T33"],
    "D": ["D"],
    "E": ["E"],
    "F": ["F"],
    "G": ["G"],
    "H": ["H49", "H50", "H51", "H52", "H53"],
    "I": ["I"],
    "J": ["J58T60", "J61", "J62_63"],
    "K": ["K"],
    "L": ["L"],
    "M": ["M"],
    "N": ["N"],
    "O": ["O"],
    "P": ["P"],
    "Q": ["Q"],
    "R_S": ["R", "S"],  # Note: R and S are aggregated in original
}