"""
OECD data source connector.

This module provides access to OECD databases, replicating the functionality
of the original MATLAB code for querying national accounts, financial markets,
and labor force statistics.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base import CachedDataSource, DataSourceType, QueryError, ConnectionError
from ..models import Country, FrequencyType


class OECDDataSource(CachedDataSource):
    """
    OECD SQLite database connector.
    
    This class provides access to OECD data stored in SQLite format,
    matching the datasets and query patterns from the original MATLAB code.
    """
    
    # Dataset mappings from MATLAB analysis
    DATASET_MAPPINGS = {
        "gdp_expenditure": "OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_EXPENDITURE_NATIO_CURR_1_1",
        "gdp_quarterly": "OECD_SDD_NAD_DSD_NAMAIN1_DF_QNA_1_1",
        "interest_rates": "OECD_SDD_STES_DSD_STES_DF_FINMARK_",
        "unemployment": "OECD_SDD_TPS_DSD_LFS_DF_IALFS_UNE_M_1_0",
        "economic_outlook": "OECD_ECO_MAD_DSD_EO_114_DF_EO_114_1_0",
        "national_accounts": "OECD_SDD_NAD_DSD_NAMAIN10_DF_TABLE6_1_0",
        "sector_accounts": "OECD_SDD_NAD_DSD_NASEC20_DF_T720R_A_1_1",
        "sector_accounts_quarterly": "OECD_SDD_NAD_DSD_NASEC20_DF_T720R_Q_1_1",
        "government_accounts": "OECD_SDD_NAD_DSD_NASEC20_DF_T720GOV_A_1_1",
        "employment": "OECD_SDD_TPS_DSD_ALFS_DF_ALFS_EMP_EES",
    }
    
    def __init__(
        self,
        database_path: Path,
        connection_timeout: int = 30,
        query_timeout: int = 120,
        **kwargs
    ):
        """
        Initialize OECD data source.
        
        Args:
            database_path: Path to OECD SQLite database
            connection_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
            **kwargs: Additional arguments for CachedDataSource
        """
        connection_params = {
            "database_path": database_path,
            "connection_timeout": connection_timeout,
            "query_timeout": query_timeout
        }
        
        super().__init__(DataSourceType.OECD, connection_params, **kwargs)
        
        self.database_path = Path(database_path)
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self._engine: Optional[Engine] = None
    
    def connect(self) -> None:
        """Establish connection to OECD SQLite database."""
        try:
            if not self.database_path.exists():
                raise ConnectionError(
                    f"OECD database file not found: {self.database_path}",
                    self.source_type
                )
            
            # Create SQLAlchemy engine for SQLite
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
            self.logger.info(f"Connected to OECD database: {self.database_path}")
            
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Failed to connect to OECD database: {e}", self.source_type) from e
    
    def disconnect(self) -> None:
        """Close connection to database."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
        
        self._is_connected = False
        self.logger.info("Disconnected from OECD database")
    
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
        """List available datasets in the database."""
        if not self._is_connected:
            self.connect()
        
        try:
            # Get all table names
            with self._engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'OECD_%'
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
            # Resolve dataset name if it's an alias
            table_name = self.DATASET_MAPPINGS.get(dataset_name, dataset_name)
            
            with self._engine.connect() as conn:
                # Get table schema
                schema_result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = [{"name": row[1], "type": row[2]} for row in schema_result]
                
                # Get row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.scalar()
                
                # Get sample of unique values for key columns
                sample_data = {}
                key_columns = ["REF_AREA", "TIME_PERIOD", "MEASURE", "_TRANSACTION"]
                
                for col in key_columns:
                    if any(c["name"] == col for c in columns):
                        unique_result = conn.execute(text(f"""
                            SELECT DISTINCT {col} FROM {table_name} 
                            WHERE {col} IS NOT NULL 
                            LIMIT 20
                        """))
                        sample_data[col] = [row[0] for row in unique_result]
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "columns": columns,
                "sample_data": sample_data
            }
            
        except Exception as e:
            raise QueryError(f"Failed to get dataset info for {dataset_name}: {e}", self.source_type) from e
    
    def _execute_query_impl(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> pd.DataFrame:
        """Execute SQL query against OECD database."""
        if not self._is_connected:
            self.connect()
        
        # Resolve dataset name if needed
        table_name = self.DATASET_MAPPINGS.get(query, query)
        
        # If query is a dataset name, build default query
        if table_name in self.DATASET_MAPPINGS.values():
            sql_query = self._build_dataset_query(table_name, parameters)
        else:
            # Assume it's a direct SQL query
            sql_query = query
        
        try:
            # Execute query with timeout
            query_timeout = timeout or self.query_timeout
            
            with self._engine.connect() as conn:
                # Set query timeout
                conn.execute(text(f"PRAGMA busy_timeout = {query_timeout * 1000}"))
                
                # Execute main query
                result = pd.read_sql(sql_query, conn)
            
            return result
            
        except Exception as e:
            raise QueryError(f"Failed to execute query: {e}", self.source_type) from e
    
    def _build_dataset_query(self, table_name: str, parameters: Optional[Dict[str, Any]]) -> str:
        """Build SQL query for dataset with parameters."""
        if not parameters:
            return f"SELECT * FROM {table_name} LIMIT 1000"
        
        # Base query
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        
        # Add country filter
        if "countries" in parameters:
            country_list = "'" + "','".join(parameters["countries"]) + "'"
            query += f" AND REF_AREA IN ({country_list})"
        
        # Add time period filter
        if "start_year" in parameters and "end_year" in parameters:
            freq = parameters.get("frequency", "A")
            if freq == "A":
                query += f" AND TIME_PERIOD BETWEEN '{parameters['start_year']}' AND '{parameters['end_year']}'"
            elif freq == "Q":
                query += f" AND TIME_PERIOD BETWEEN '{parameters['start_year']}-Q1' AND '{parameters['end_year']}-Q4'"
        
        # Add variable filter
        if "variables" in parameters:
            var_list = "'" + "','".join(parameters["variables"]) + "'"
            
            # Try different column names for variables
            for var_col in ["MEASURE", "_TRANSACTION", "INDICATOR"]:
                query += f" AND {var_col} IN ({var_list})"
                break
        
        # Add ordering
        query += " ORDER BY REF_AREA, TIME_PERIOD"
        
        return query
    
    def fetch_gdp_data(
        self,
        countries: List[str],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY,
        price_base: str = "V"  # V=current prices, L=constant prices
    ) -> pd.DataFrame:
        """
        Fetch GDP data matching MATLAB functionality.
        
        Args:
            countries: List of country codes
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            price_base: Price base (V=current, L=constant, Q=quarterly for MEX)
            
        Returns:
            DataFrame with GDP data
        """
        parameters = {
            "countries": countries,
            "start_year": start_year,
            "end_year": end_year,
            "frequency": frequency.value
        }
        
        # Build specific GDP query based on MATLAB logic
        freq_filter = "Q" if frequency == FrequencyType.QUARTERLY else "A"
        time_filter = f"'{start_year}-Q1' AND '{end_year}-Q4'" if freq_filter == "Q" else f"'{start_year}' AND '{end_year}'"
        
        sql_query = f"""
        SELECT REF_AREA, TIME_PERIOD, OBS_VALUE, PRICE_BASE
        FROM {self.DATASET_MAPPINGS['gdp_expenditure']}
        WHERE FREQ = '{freq_filter}'
        AND ADJUSTMENT = 'Y'
        AND REF_AREA IN ('{"','".join(countries)}')
        AND SECTOR = 'S1'
        AND COUNTERPART_SECTOR = 'S1'
        AND _TRANSACTION = 'B1GQ'
        AND INSTR_ASSET = '_Z'
        AND ACTIVITY = '_Z'
        AND EXPENDITURE = '_Z'
        AND UNIT_MEASURE = 'XDC'
        AND PRICE_BASE = '{price_base}'
        AND TIME_PERIOD BETWEEN {time_filter}
        ORDER BY REF_AREA, TIME_PERIOD
        """
        
        result = self.execute_query(sql_query)
        return result.data
    
    def fetch_exchange_rates(
        self,
        currencies: List[str],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY,
        base_currency: str = "USD"
    ) -> pd.DataFrame:
        """
        Fetch exchange rate data.
        
        Args:
            currencies: List of currency codes
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            base_currency: Base currency for rates
            
        Returns:
            DataFrame with exchange rate data
        """
        # Note: Exchange rates are typically in Eurostat data
        # This is a placeholder for OECD interest rate data
        parameters = {
            "countries": currencies,
            "start_year": start_year,
            "end_year": end_year,
            "frequency": frequency.value
        }
        
        freq_filter = "Q" if frequency == FrequencyType.QUARTERLY else "A"
        time_filter = f"'{start_year}-Q1' AND '{end_year}-Q4'" if freq_filter == "Q" else f"'{start_year}' AND '{end_year}'"
        
        sql_query = f"""
        SELECT REF_AREA, TIME_PERIOD, OBS_VALUE
        FROM {self.DATASET_MAPPINGS['interest_rates']}
        WHERE REF_AREA IN ('{"','".join(currencies)}')
        AND FREQ = '{freq_filter}'
        AND MEASURE = 'IR3TIB'
        AND UNIT_MEASURE = 'PA'
        AND TIME_PERIOD BETWEEN {time_filter}
        ORDER BY REF_AREA, TIME_PERIOD
        """
        
        result = self.execute_query(sql_query)
        return result.data
    
    def fetch_unemployment_data(
        self,
        countries: List[str],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY
    ) -> pd.DataFrame:
        """
        Fetch unemployment rate data.
        
        Args:
            countries: List of country codes
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            
        Returns:
            DataFrame with unemployment data
        """
        freq_filter = "Q" if frequency == FrequencyType.QUARTERLY else "A"
        time_filter = f"'{start_year}-Q1' AND '{end_year}-Q4'" if freq_filter == "Q" else f"'{start_year}' AND '{end_year}'"
        
        sql_query = f"""
        SELECT REF_AREA, TIME_PERIOD, OBS_VALUE
        FROM {self.DATASET_MAPPINGS['unemployment']}
        WHERE FREQ = '{freq_filter}'
        AND ADJUSTMENT = 'Y'
        AND REF_AREA IN ('{"','".join(countries)}')
        AND MEASURE = 'UNE_LF_M'
        AND UNIT_MEASURE = 'PT_LF_SUB'
        AND TRANSFORMATION = '_Z'
        AND SEX = '_T'
        AND AGE = 'Y_GE15'
        AND TIME_PERIOD BETWEEN {time_filter}
        ORDER BY REF_AREA, TIME_PERIOD
        """
        
        result = self.execute_query(sql_query)
        return result.data
    
    def fetch_consumption_data(
        self,
        countries: List[str],
        start_year: int,
        end_year: int,
        frequency: FrequencyType = FrequencyType.QUARTERLY,
        sector: str = "S1M",  # S1M=households, S13=government
        price_base: str = "V"
    ) -> pd.DataFrame:
        """
        Fetch consumption data (household or government).
        
        Args:
            countries: List of country codes
            start_year: Start year
            end_year: End year
            frequency: Data frequency
            sector: Sector (S1M=households, S13=government)
            price_base: Price base
            
        Returns:
            DataFrame with consumption data
        """
        freq_filter = "Q" if frequency == FrequencyType.QUARTERLY else "A"
        time_filter = f"'{start_year}-Q1' AND '{end_year}-Q4'" if freq_filter == "Q" else f"'{start_year}' AND '{end_year}'"
        
        sql_query = f"""
        SELECT REF_AREA, TIME_PERIOD, OBS_VALUE, SECTOR
        FROM {self.DATASET_MAPPINGS['gdp_expenditure']}
        WHERE FREQ = '{freq_filter}'
        AND ADJUSTMENT = 'Y'
        AND REF_AREA IN ('{"','".join(countries)}')
        AND SECTOR = '{sector}'
        AND COUNTERPART_SECTOR = 'S1'
        AND _TRANSACTION = 'P3'
        AND INSTR_ASSET = '_Z'
        AND ACTIVITY = '_Z'
        AND EXPENDITURE = '_T'
        AND UNIT_MEASURE = 'XDC'
        AND PRICE_BASE = '{price_base}'
        AND TIME_PERIOD BETWEEN {time_filter}
        ORDER BY REF_AREA, TIME_PERIOD
        """
        
        result = self.execute_query(sql_query)
        return result.data