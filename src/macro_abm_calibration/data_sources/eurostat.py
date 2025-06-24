"""
Eurostat data source connector.

This module provides access to Eurostat exchange rate data, replicating the
functionality from the original MATLAB code for bilateral EUR exchange rates.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base import CachedDataSource, DataSourceType, QueryError, ConnectionError
from ..models import Country, FrequencyType, OECD_COUNTRIES


class EurostatDataSource(CachedDataSource):
    """
    Eurostat data connector for exchange rates.
    
    This class provides access to Eurostat exchange rate data stored in the
    same SQLite database as OECD data, following the patterns from the original
    MATLAB code.
    """
    
    # Dataset mappings for Eurostat tables
    DATASET_MAPPINGS = {
        "exchange_rates_quarterly": "estat_ert_bil_eur_q_filtered_en",
        "exchange_rates_annual": "estat_ert_bil_eur_a_filtered_en",
        "national_accounts": "estat_nama_10_nfa_st_en",
    }
    
    # Currency code mappings from OECD countries to Eurostat codes
    CURRENCY_MAPPINGS = {
        "AUS": "AUD", "AUT": "EUR", "BEL": "EUR", "CAN": "CAD", "CZE": "CZK",
        "DNK": "DKK", "EST": "EUR", "FIN": "EUR", "FRA": "EUR", "DEU": "EUR",
        "GRC": "EUR", "HUN": "HUF", "IRL": "EUR", "ITA": "EUR", "JPN": "JPY",
        "KOR": "KRW", "LVA": "EUR", "LTU": "EUR", "LUX": "EUR", "MEX": "MXN",
        "NLD": "EUR", "NOR": "NOK", "NZL": "NZD", "POL": "PLN", "PRT": "EUR",
        "SVK": "EUR", "SVN": "EUR", "ESP": "EUR", "SWE": "SEK", "GBR": "GBP",
        "USA": "USD",
    }
    
    def __init__(
        self,
        database_path: Path,
        connection_timeout: int = 30,
        query_timeout: int = 120,
        **kwargs
    ):
        """
        Initialize Eurostat data source.
        
        Args:
            database_path: Path to SQLite database containing Eurostat tables
            connection_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
            **kwargs: Additional arguments for CachedDataSource
        """
        connection_params = {
            "database_path": database_path,
            "connection_timeout": connection_timeout,
            "query_timeout": query_timeout
        }
        
        super().__init__(DataSourceType.EUROSTAT, connection_params, **kwargs)
        
        self.database_path = Path(database_path)
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self._engine: Optional[Engine] = None
    
    def connect(self) -> None:
        """Establish connection to database."""
        try:
            if not self.database_path.exists():
                raise ConnectionError(
                    f"Database file not found: {self.database_path}",
                    self.source_type
                )
            
            # Create SQLAlchemy engine
            connection_string = f"sqlite:///{self.database_path}"
            self._engine = create_engine(
                connection_string,
                connect_args={"timeout": self.connection_timeout},
                echo=False
            )
            
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self._is_connected = True
            self.logger.info(f"Connected to Eurostat database: {self.database_path}")
            
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to Eurostat database: {e}", self.source_type) from e
    
    def disconnect(self) -> None:
        """Close connection to database."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        
        self._is_connected = False
        self.logger.info("Disconnected from Eurostat database")
    
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            if not self._is_connected or not self._engine:
                return False
            
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
            
        except Exception:
            return False
    
    def list_datasets(self) -> List[str]:
        """List available Eurostat datasets."""
        if not self._is_connected:
            self.connect()
        
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'estat_%'
                    ORDER BY name
                """))
                tables = [row[0] for row in result]
            
            return tables
            
        except Exception as e:
            raise QueryError(f"Failed to list datasets: {e}", self.source_type) from e
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        if not self._is_connected:
            self.connect()
        
        try:
            # Resolve dataset name
            table_name = self.DATASET_MAPPINGS.get(dataset_name, dataset_name)
            
            with self._engine.connect() as conn:
                # Get table schema
                schema_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [{"name": row[1], "type": row[2]} for row in schema_result]
                
                # Get row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.scalar()
                
                # Get sample currencies
                currency_result = conn.execute(text(f"""
                    SELECT DISTINCT currency FROM {table_name} 
                    WHERE currency IS NOT NULL 
                    LIMIT 20
                """))
                currencies = [row[0] for row in currency_result]
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": columns,
                "currencies": currencies
            }
            
        except Exception as e:
            raise QueryError(f"Failed to get dataset info for {dataset_name}: {e}", self.source_type) from e
    
    def _execute_query_impl(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute query against Eurostat database."""
        if not self._is_connected:
            self.connect()
        
        # Resolve dataset name if needed
        table_name = self.DATASET_MAPPINGS.get(query, query)
        
        # Build query if it's a dataset name
        if table_name in self.DATASET_MAPPINGS.values():
            sql_query = self._build_exchange_rate_query(table_name, parameters)
        else:
            sql_query = query
        
        try:
            query_timeout = timeout or self.query_timeout
            
            with self._engine.connect() as conn:
                conn.execute(text(f"PRAGMA busy_timeout = {query_timeout * 1000}"))
                result = pd.read_sql(sql_query, conn)
            
            return result
            
        except Exception as e:
            raise QueryError(f"Failed to execute query: {e}", self.source_type) from e
    
    def _build_exchange_rate_query(self, table_name: str, parameters: Optional[Dict[str, Any]]) -> str:
        """Build exchange rate query with parameters."""
        if not parameters:
            return f"SELECT * FROM {table_name} LIMIT 1000"
        
        query = f"SELECT currency, TIME_PERIOD, OBS_VALUE FROM {table_name} WHERE 1=1"
        
        # Add currency filter
        if "currencies" in parameters:
            currency_list = "'" + "','".join(parameters["currencies"]) + "'"
            query += f" AND currency IN ({currency_list})"
        
        # Add time period filter
        if "start_year" in parameters and "end_year" in parameters:
            frequency = parameters.get("frequency", "A")
            if frequency == "A":
                query += f" AND TIME_PERIOD BETWEEN '{parameters['start_year']}' AND '{parameters['end_year']}'"
            elif frequency == "Q":
                query += f" AND TIME_PERIOD BETWEEN '{parameters['start_year']}-Q1' AND '{parameters['end_year']}-Q4'"
        
        query += " ORDER BY currency, TIME_PERIOD"
        return query
    
    def fetch_exchange_rates(
        self,
        countries: List[str],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY,
        base_currency: str = "EUR"
    ) -> pd.DataFrame:
        """
        Fetch bilateral exchange rates to EUR.
        
        This function replicates the exchange rate fetching logic from the
        original MATLAB code, including special handling for certain currencies.
        
        Args:
            countries: List of country codes (OECD format)
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            base_currency: Base currency (should be EUR for Eurostat)
            
        Returns:
            DataFrame with exchange rates
        """
        # Convert country codes to currency codes
        currencies = []
        for country in countries:
            if country in self.CURRENCY_MAPPINGS:
                currencies.append(self.CURRENCY_MAPPINGS[country])
            else:
                self.logger.warning(f"No currency mapping for country: {country}")
        
        # Remove duplicates and add USD for conversion
        currencies = list(set(currencies))
        if "USD" not in currencies:
            currencies.append("USD")
        
        # Determine dataset and time format
        if frequency == FrequencyType.QUARTERLY:
            dataset = "exchange_rates_quarterly"
            time_suffix = "-Q1' AND '" + str(end_year) + "-Q4"
        else:
            dataset = "exchange_rates_annual"
            time_suffix = "'"
        
        table_name = self.DATASET_MAPPINGS[dataset]
        
        # Build query
        currency_list = "'" + "','".join(currencies) + "'"
        time_filter = f"'{start_year}{time_suffix if frequency == FrequencyType.QUARTERLY else ''}' AND '{end_year}{'' if frequency == FrequencyType.ANNUAL else '-Q4'}"
        
        sql_query = f"""
        SELECT currency, TIME_PERIOD, OBS_VALUE
        FROM {table_name}
        WHERE currency IN ({currency_list})
        AND TIME_PERIOD BETWEEN {time_filter}
        ORDER BY currency, TIME_PERIOD
        """
        
        result = self.execute_query(sql_query)
        return result.data
    
    def fetch_usd_eur_rates(
        self,
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY
    ) -> pd.DataFrame:
        """
        Fetch USD to EUR exchange rates.
        
        This is used as the base for converting other currencies to USD,
        following the original MATLAB logic.
        
        Args:
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            
        Returns:
            DataFrame with USD/EUR rates
        """
        return self.fetch_exchange_rates(
            countries=["USA"],
            start_year=start_year,
            end_year=end_year,
            frequency=frequency
        )
    
    def convert_to_usd_rates(
        self,
        eur_rates: pd.DataFrame,
        usd_eur_rate: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert EUR-based rates to USD-based rates.
        
        This function implements the conversion logic from the MATLAB code:
        LC_to_USD = LC_to_EUR / USD_to_EUR
        
        Args:
            eur_rates: DataFrame with rates to EUR
            usd_eur_rate: DataFrame with USD to EUR rate
            
        Returns:
            DataFrame with rates converted to USD base
        """
        # Merge the dataframes on time period
        merged = eur_rates.merge(
            usd_eur_rate[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"OBS_VALUE": "USD_EUR_RATE"}),
            on="TIME_PERIOD",
            how="left"
        )
        
        # Calculate USD rates: LC_to_USD = LC_to_EUR / USD_to_EUR
        merged["USD_RATE"] = merged["OBS_VALUE"] / merged["USD_EUR_RATE"]
        
        # Handle EUR currency (should be 1/USD_EUR_RATE)
        eur_mask = merged["currency"] == "EUR"
        merged.loc[eur_mask, "USD_RATE"] = 1.0 / merged.loc[eur_mask, "USD_EUR_RATE"]
        
        return merged[["currency", "TIME_PERIOD", "USD_RATE"]].rename(columns={"USD_RATE": "OBS_VALUE"})
    
    def fetch_country_exchange_rates(
        self,
        countries: List[Country],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY,
        target_currency: str = "USD"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch exchange rates for multiple countries with special handling.
        
        This function replicates the complex exchange rate logic from the
        original MATLAB code, including special cases for different currencies
        and time periods.
        
        Args:
            countries: List of Country objects
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            target_currency: Target currency (USD)
            
        Returns:
            Dictionary mapping country codes to exchange rate DataFrames
        """
        results = {}
        
        # First, get USD to EUR rates for conversion
        usd_eur_rates = self.fetch_usd_eur_rates(start_year, end_year, frequency)
        
        for country in countries:
            country_code = country.oecd_code
            currency = self.CURRENCY_MAPPINGS.get(country_code)
            
            if not currency:
                self.logger.warning(f"No currency mapping for {country_code}")
                continue
            
            try:
                if currency == "EUR":
                    # For EUR countries, rate is 1/USD_EUR_RATE
                    rates = usd_eur_rates.copy()
                    rates["OBS_VALUE"] = 1.0 / rates["OBS_VALUE"]
                    rates["currency"] = "EUR"
                
                elif currency in ["ILS", "KRW", "MXN"]:
                    # Special handling for currencies with limited historical data
                    rates = self._fetch_currency_with_interpolation(
                        currency, start_year, end_year, frequency, usd_eur_rates
                    )
                
                else:
                    # Standard currency handling
                    eur_rates = self.fetch_exchange_rates(
                        countries=[country_code],
                        start_year=start_year,
                        end_year=end_year,
                        frequency=frequency
                    )
                    
                    if not eur_rates.empty:
                        rates = self.convert_to_usd_rates(eur_rates, usd_eur_rates)
                    else:
                        self.logger.warning(f"No exchange rate data for {currency}")
                        continue
                
                results[country_code] = rates
                
            except Exception as e:
                self.logger.error(f"Failed to fetch exchange rates for {country_code}: {e}")
                continue
        
        return results
    
    def _fetch_currency_with_interpolation(
        self,
        currency: str,
        start_year: int,
        end_year: int,
        frequency: FrequencyType,
        usd_eur_rates: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fetch currency rates with interpolation for missing early periods.
        
        This implements the MATLAB logic for currencies like ILS, KRW, MXN
        that have limited historical data availability.
        """
        # Fetch available data
        eur_rates = self.fetch_exchange_rates(
            countries=[currency],  # Using currency directly
            start_year=start_year,
            end_year=end_year,
            frequency=frequency
        )
        
        if eur_rates.empty:
            raise QueryError(f"No data available for currency {currency}")
        
        # Convert to USD rates
        usd_rates = self.convert_to_usd_rates(eur_rates, usd_eur_rates)
        
        # Create full time series
        if frequency == FrequencyType.QUARTERLY:
            time_index = pd.date_range(
                start=f"{start_year}-01-01",
                end=f"{end_year}-12-31",
                freq="Q"
            )
            time_periods = [f"{d.year}-Q{d.quarter}" for d in time_index]
        else:
            time_periods = [str(year) for year in range(start_year, end_year + 1)]
        
        # Create full DataFrame
        full_series = pd.DataFrame({
            "TIME_PERIOD": time_periods,
            "currency": currency
        })
        
        # Merge with available data
        merged = full_series.merge(usd_rates, on=["TIME_PERIOD", "currency"], how="left")
        
        # Interpolate missing values using nearest neighbor extrapolation
        # This matches the MATLAB interp1 'nearest' 'extrap' behavior
        merged["OBS_VALUE"] = merged["OBS_VALUE"].fillna(method="bfill").fillna(method="ffill")
        
        return merged