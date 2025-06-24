"""
Currency conversion processor.

This module implements currency conversion logic that replicates the exchange rate
processing from the original MATLAB code, including USD conversion and special
handling for different currencies.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import DataProcessor, ProcessingResult, ProcessingStatus, ProcessingMetadata
from .utils import interpolate_missing_data, extrapolate_missing_data
from ..data_sources import EurostatDataSource, DataSourceType
from ..models import Country, FrequencyType, OECD_COUNTRIES
from ..utils.validation import ValidationResult


class CurrencyConverter(DataProcessor):
    """
    Currency conversion processor.
    
    This class handles conversion of economic data to a common currency (USD),
    replicating the complex exchange rate logic from the original MATLAB code.
    """
    
    # Currency mappings from OECD countries to Eurostat currency codes
    CURRENCY_MAPPINGS = {
        "AUS": "AUD", "AUT": "EUR", "BEL": "EUR", "CAN": "CAD", "CZE": "CZK",
        "DNK": "DKK", "EST": "EUR", "FIN": "EUR", "FRA": "EUR", "DEU": "EUR",
        "GRC": "EUR", "HUN": "HUF", "IRL": "EUR", "ITA": "EUR", "JPN": "JPY",
        "KOR": "KRW", "LVA": "EUR", "LTU": "EUR", "LUX": "EUR", "MEX": "MXN",
        "NLD": "EUR", "NOR": "NOK", "NZL": "NZD", "POL": "PLN", "PRT": "EUR",
        "SVK": "EUR", "SVN": "EUR", "ESP": "EUR", "SWE": "SEK", "GBR": "GBP",
        "USA": "USD",
    }
    
    # Currencies with limited historical data requiring special handling
    LIMITED_DATA_CURRENCIES = ["ILS", "KRW", "MXN"]
    
    def __init__(
        self,
        eurostat_source: EurostatDataSource,
        target_currency: str = "USD",
        name: Optional[str] = None
    ):
        """
        Initialize currency converter.
        
        Args:
            eurostat_source: Eurostat data source for exchange rates
            target_currency: Target currency for conversion (default: USD)
            name: Processor name
        """
        super().__init__(name or "CurrencyConverter")
        self.eurostat_source = eurostat_source
        self.target_currency = target_currency
        self._exchange_rate_cache: Dict[str, pd.DataFrame] = {}
    
    def process(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Convert currency data to target currency.
        
        Args:
            data: Economic data to convert (with country information)
            parameters: Processing parameters including:
                - countries: List of country codes
                - start_year: Start year for exchange rates
                - end_year: End year for exchange rates
                - frequency: Data frequency ('Q' or 'A')
                - value_columns: Columns containing values to convert
                
        Returns:
            ProcessingResult with currency-converted data
        """
        operation_id = self._generate_operation_id()
        
        # Extract parameters
        countries = parameters.get("countries", []) if parameters else []
        start_year = parameters.get("start_year", 2010) if parameters else 2010
        end_year = parameters.get("end_year", 2020) if parameters else 2020
        frequency = parameters.get("frequency", "Q") if parameters else "Q"
        value_columns = parameters.get("value_columns", []) if parameters else []
        
        if not countries:
            raise ValueError("Countries parameter is required")
        
        # Get exchange rates for all countries
        exchange_rates = self._get_exchange_rates(
            countries, start_year, end_year, frequency
        )
        
        # Convert data
        if isinstance(data, pd.DataFrame):
            converted_data = self._convert_dataframe(data, exchange_rates, value_columns)
        else:
            converted_data = {}
            for name, df in data.items():
                converted_data[name] = self._convert_dataframe(df, exchange_rates, value_columns)
        
        # Create result
        metadata = ProcessingMetadata(
            processor_name=self.name,
            operation_id=operation_id,
            parameters=parameters or {}
        )
        
        return ProcessingResult(
            data=converted_data,
            metadata=metadata,
            status=ProcessingStatus.COMPLETED
        )
    
    def _get_exchange_rates(
        self,
        countries: List[str],
        start_year: int,
        end_year: int,
        frequency: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Get exchange rates for countries, with caching.
        
        This method replicates the complex exchange rate fetching logic
        from the original MATLAB code.
        """
        cache_key = f"{'-'.join(countries)}_{start_year}_{end_year}_{frequency}"
        
        if cache_key in self._exchange_rate_cache:
            return self._exchange_rate_cache[cache_key]
        
        self.logger.info(f"Fetching exchange rates for {len(countries)} countries")
        
        # Convert frequency
        freq_type = FrequencyType.QUARTERLY if frequency == "Q" else FrequencyType.ANNUAL
        
        # Get USD to EUR rates (base for all conversions)
        usd_eur_rates = self.eurostat_source.fetch_usd_eur_rates(
            start_year, end_year, freq_type
        )
        
        if usd_eur_rates.empty:
            raise ValueError("Failed to fetch USD/EUR exchange rates")
        
        # Process each country
        exchange_rates = {}
        
        for country_code in countries:
            try:
                rates = self._get_country_exchange_rate(
                    country_code, start_year, end_year, freq_type, usd_eur_rates
                )
                exchange_rates[country_code] = rates
                
            except Exception as e:
                self.logger.warning(f"Failed to get exchange rates for {country_code}: {e}")
                # Create default rates (1.0 for USD, skip for others)
                if country_code == "USA":
                    default_rates = self._create_default_rates(start_year, end_year, frequency, 1.0)
                    exchange_rates[country_code] = default_rates
        
        # Cache results
        self._exchange_rate_cache[cache_key] = exchange_rates
        
        return exchange_rates
    
    def _get_country_exchange_rate(
        self,
        country_code: str,
        start_year: int,
        end_year: int,
        frequency: FrequencyType,
        usd_eur_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get exchange rates for a specific country.
        
        This method implements the country-specific logic from the MATLAB code,
        including special handling for EUR countries and limited data currencies.
        """
        currency = self.CURRENCY_MAPPINGS.get(country_code)
        if not currency:
            raise ValueError(f"No currency mapping for country {country_code}")
        
        # Handle USD (target currency)
        if currency == "USD":
            return self._create_default_rates(start_year, end_year, frequency.value, 1.0)
        
        # Handle EUR countries
        if currency == "EUR":
            # For EUR countries: rate = 1/USD_to_EUR (EUR to USD)
            rates = usd_eur_rates.copy()
            rates["OBS_VALUE"] = 1.0 / rates["OBS_VALUE"]
            rates["currency"] = country_code
            return rates
        
        # Handle other currencies
        return self._get_non_eur_currency_rate(
            country_code, currency, start_year, end_year, frequency, usd_eur_rates
        )
    
    def _get_non_eur_currency_rate(
        self,
        country_code: str,
        currency: str,
        start_year: int,
        end_year: int,
        frequency: FrequencyType,
        usd_eur_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get exchange rates for non-EUR currencies.
        
        This implements the LC_to_USD = LC_to_EUR / USD_to_EUR logic.
        """
        # Fetch LC to EUR rates
        eur_rates = self.eurostat_source.fetch_exchange_rates(
            countries=[country_code],
            start_year=start_year,
            end_year=end_year,
            frequency=frequency
        )
        
        if eur_rates.empty:
            raise ValueError(f"No exchange rate data for {currency}")
        
        # Handle currencies with limited historical data
        if currency in self.LIMITED_DATA_CURRENCIES:
            eur_rates = self._handle_limited_data_currency(
                eur_rates, start_year, end_year, frequency
            )
        
        # Convert EUR rates to USD rates
        usd_rates = self._convert_eur_to_usd_rates(eur_rates, usd_eur_rates)
        usd_rates["currency"] = country_code
        
        return usd_rates
    
    def _handle_limited_data_currency(
        self,
        eur_rates: pd.DataFrame,
        start_year: int,
        end_year: int,
        frequency: FrequencyType
    ) -> pd.DataFrame:
        """
        Handle currencies with limited historical data.
        
        This replicates the MATLAB logic for ILS, KRW, MXN that uses
        interpolation and extrapolation for missing early periods.
        """
        # Create full time index
        if frequency == FrequencyType.QUARTERLY:
            full_index = pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                freq="Q"
            )
            time_periods = [f"{d.year}-Q{d.quarter}" for d in full_index]
        else:
            time_periods = [str(year) for year in range(start_year, end_year + 1)]
        
        # Create DataFrame with full time range
        full_data = pd.DataFrame({"TIME_PERIOD": time_periods})
        
        # Merge with available data
        merged = full_data.merge(eur_rates, on="TIME_PERIOD", how="left")
        
        # Apply extrapolation (nearest neighbor)
        if "OBS_VALUE" in merged.columns:
            merged["OBS_VALUE"] = extrapolate_missing_data(
                merged["OBS_VALUE"], method="nearest"
            )
        
        return merged.dropna()
    
    def _convert_eur_to_usd_rates(
        self,
        eur_rates: pd.DataFrame,
        usd_eur_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert EUR-based rates to USD-based rates.
        
        Implements: LC_to_USD = LC_to_EUR / USD_to_EUR
        """
        # Merge on time period
        merged = eur_rates.merge(
            usd_eur_rates[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"OBS_VALUE": "USD_EUR_RATE"}),
            on="TIME_PERIOD",
            how="left"
        )
        
        # Calculate USD rates
        merged["USD_RATE"] = merged["OBS_VALUE"] / merged["USD_EUR_RATE"]
        
        return merged[["TIME_PERIOD", "USD_RATE"]].rename(columns={"USD_RATE": "OBS_VALUE"})
    
    def _create_default_rates(
        self,
        start_year: int,
        end_year: int,
        frequency: str,
        rate_value: float
    ) -> pd.DataFrame:
        """Create default exchange rates (typically 1.0 for base currency)."""
        if frequency == "Q":
            periods = []
            for year in range(start_year, end_year + 1):
                for quarter in range(1, 5):
                    periods.append(f"{year}-Q{quarter}")
        else:
            periods = [str(year) for year in range(start_year, end_year + 1)]
        
        return pd.DataFrame({
            "TIME_PERIOD": periods,
            "OBS_VALUE": rate_value
        })
    
    def _convert_dataframe(
        self,
        data: pd.DataFrame,
        exchange_rates: Dict[str, pd.DataFrame],
        value_columns: List[str]
    ) -> pd.DataFrame:
        """
        Convert DataFrame values using exchange rates.
        
        Args:
            data: DataFrame with economic data
            exchange_rates: Dictionary of exchange rates by country
            value_columns: Columns containing values to convert
            
        Returns:
            DataFrame with converted values
        """
        converted_data = data.copy()
        
        # If no value columns specified, try to detect numeric columns
        if not value_columns:
            value_columns = list(data.select_dtypes(include=[np.number]).columns)
        
        # Convert each value column
        for column in value_columns:
            if column not in converted_data.columns:
                continue
            
            converted_data[column] = self._convert_column(
                converted_data, column, exchange_rates
            )
        
        return converted_data
    
    def _convert_column(
        self,
        data: pd.DataFrame,
        column: str,
        exchange_rates: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """Convert a specific column using exchange rates."""
        converted_values = data[column].copy()
        
        # Check if data has country and time information
        if "REF_AREA" in data.columns and "TIME_PERIOD" in data.columns:
            # Convert row by row
            for idx, row in data.iterrows():
                country = row["REF_AREA"]
                time_period = row["TIME_PERIOD"]
                
                if country in exchange_rates:
                    country_rates = exchange_rates[country]
                    
                    # Find matching time period
                    rate_row = country_rates[country_rates["TIME_PERIOD"] == time_period]
                    
                    if not rate_row.empty:
                        exchange_rate = rate_row["OBS_VALUE"].iloc[0]
                        
                        # Convert: value_in_USD = value_in_local_currency * local_to_USD_rate
                        if not pd.isna(exchange_rate) and exchange_rate != 0:
                            converted_values.iloc[idx] = row[column] * exchange_rate
        
        return converted_values
    
    def validate_input(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate input data for currency conversion."""
        result = super().validate_input(data, parameters)
        
        if parameters:
            # Check required parameters
            if "countries" not in parameters:
                result.add_error("Countries parameter is required")
            
            # Validate country codes
            countries = parameters.get("countries", [])
            invalid_countries = [c for c in countries if c not in self.CURRENCY_MAPPINGS]
            if invalid_countries:
                result.add_error(f"Invalid country codes: {invalid_countries}")
        
        return result
    
    def get_supported_currencies(self) -> Dict[str, str]:
        """Get dictionary of supported country codes and their currencies."""
        return self.CURRENCY_MAPPINGS.copy()
    
    def clear_cache(self) -> None:
        """Clear exchange rate cache."""
        self._exchange_rate_cache.clear()
        self.logger.info("Exchange rate cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached exchange rates."""
        return {
            "cached_combinations": len(self._exchange_rate_cache),
            "cache_keys": list(self._exchange_rate_cache.keys())
        }