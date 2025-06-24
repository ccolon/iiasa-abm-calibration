"""
Base interfaces and abstract classes for data sources.

This module defines the common interface that all data sources must implement,
providing consistency and extensibility across different data providers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from ..models import Country, Industry, TimeFrame
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataSourceType(str, Enum):
    """Types of data sources."""
    OECD = "oecd"
    EUROSTAT = "eurostat"
    ICIO = "icio"
    CUSTOM = "custom"


class QueryStatus(str, Enum):
    """Status of a data query."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class QueryMetadata:
    """Metadata for data queries."""
    query_id: str
    source_type: DataSourceType
    dataset_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    row_count: Optional[int] = None
    error_message: Optional[str] = None


class QueryResult(BaseModel):
    """
    Result of a data source query.
    
    Attributes:
        data: Retrieved data as DataFrame
        metadata: Query metadata
        status: Query execution status
        cache_hit: Whether result came from cache
    """
    data: pd.DataFrame
    metadata: QueryMetadata
    status: QueryStatus = QueryStatus.COMPLETED
    cache_hit: bool = False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda df: df.to_dict(),
            datetime: lambda dt: dt.isoformat(),
        }
    
    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return self.data.empty
    
    @property
    def shape(self) -> tuple:
        """Get data shape."""
        return self.data.shape
    
    def validate_data(self) -> bool:
        """Validate data quality."""
        if self.data.empty:
            return False
        
        # Check for basic data integrity
        if self.data.isnull().all().all():
            return False
        
        return True


class DataSourceError(Exception):
    """Base exception for data source errors."""
    
    def __init__(self, message: str, source_type: Optional[DataSourceType] = None, query_id: Optional[str] = None):
        """Initialize error."""
        super().__init__(message)
        self.source_type = source_type
        self.query_id = query_id
        self.timestamp = datetime.now()


class ConnectionError(DataSourceError):
    """Error connecting to data source."""
    pass


class QueryError(DataSourceError):
    """Error executing query."""
    pass


class ValidationError(DataSourceError):
    """Error validating data."""
    pass


class DataSource(ABC):
    """
    Abstract base class for all data sources.
    
    This class defines the common interface that all data source implementations
    must follow, ensuring consistency and interoperability.
    """
    
    def __init__(self, source_type: DataSourceType, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize data source.
        
        Args:
            source_type: Type of data source
            connection_params: Connection parameters
        """
        self.source_type = source_type
        self.connection_params = connection_params or {}
        self._connection = None
        self._is_connected = False
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if connection is working."""
        pass
    
    @abstractmethod
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        pass
    
    @abstractmethod
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        pass
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """
        Execute a query and return results.
        
        Args:
            query: Query string or identifier
            parameters: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            QueryResult object
        """
        query_id = self._generate_query_id()
        start_time = time.time()
        
        metadata = QueryMetadata(
            query_id=query_id,
            source_type=self.source_type,
            dataset_name=query,
            parameters=parameters or {}
        )
        
        try:
            self.logger.info(f"Executing query {query_id}", extra={
                "query": query,
                "parameters": parameters,
                "source_type": self.source_type.value
            })
            
            # Ensure connection
            if not self._is_connected:
                self.connect()
            
            # Execute the actual query (implemented by subclasses)
            data = self._execute_query_impl(query, parameters, timeout)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            metadata.execution_time = execution_time
            metadata.row_count = len(data) if not data.empty else 0
            
            self.logger.info(f"Query {query_id} completed successfully", extra={
                "execution_time": execution_time,
                "row_count": metadata.row_count
            })
            
            return QueryResult(
                data=data,
                metadata=metadata,
                status=QueryStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            metadata.execution_time = execution_time
            metadata.error_message = str(e)
            
            self.logger.error(f"Query {query_id} failed", extra={
                "error": str(e),
                "execution_time": execution_time
            })
            
            raise QueryError(f"Query execution failed: {e}", self.source_type, query_id) from e
    
    @abstractmethod
    def _execute_query_impl(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> pd.DataFrame:
        """Implementation-specific query execution."""
        pass
    
    def fetch_country_data(
        self,
        dataset: str,
        countries: List[Country],
        time_frame: TimeFrame,
        variables: Optional[List[str]] = None
    ) -> QueryResult:
        """
        Fetch data for specific countries and time period.
        
        Args:
            dataset: Dataset identifier
            countries: List of countries
            time_frame: Time period
            variables: Specific variables to fetch
            
        Returns:
            QueryResult with country data
        """
        country_codes = [c.oecd_code for c in countries]
        
        parameters = {
            "countries": country_codes,
            "start_year": time_frame.start_year,
            "end_year": time_frame.end_year,
            "frequency": time_frame.frequency.value,
            "variables": variables
        }
        
        return self.execute_query(dataset, parameters)
    
    def fetch_industry_data(
        self,
        dataset: str,
        industries: List[Industry],
        countries: List[Country],
        time_frame: TimeFrame
    ) -> QueryResult:
        """
        Fetch industry-level data.
        
        Args:
            dataset: Dataset identifier
            industries: List of industries
            countries: List of countries
            time_frame: Time period
            
        Returns:
            QueryResult with industry data
        """
        industry_codes = [i.code for i in industries]
        country_codes = [c.oecd_code for c in countries]
        
        parameters = {
            "industries": industry_codes,
            "countries": country_codes,
            "start_year": time_frame.start_year,
            "end_year": time_frame.end_year,
            "frequency": time_frame.frequency.value
        }
        
        return self.execute_query(dataset, parameters)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def _generate_query_id(self) -> str:
        """Generate unique query identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{self.source_type.value}_{timestamp}"
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._is_connected


class CachedDataSource(DataSource):
    """
    Data source with caching capabilities.
    
    This class extends the base DataSource to add caching functionality,
    reducing redundant queries and improving performance.
    """
    
    def __init__(
        self,
        source_type: DataSourceType,
        connection_params: Optional[Dict[str, Any]] = None,
        cache_enabled: bool = True,
        cache_ttl: int = 3600  # 1 hour default
    ):
        """
        Initialize cached data source.
        
        Args:
            source_type: Type of data source
            connection_params: Connection parameters
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(source_type, connection_params)
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[QueryResult, datetime]] = {}
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> QueryResult:
        """Execute query with caching support."""
        if not self.cache_enabled:
            return super().execute_query(query, parameters, timeout)
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, parameters)
        
        # Check cache
        if cache_key in self._cache:
            cached_result, cache_time = self._cache[cache_key]
            cache_age = (datetime.now() - cache_time).total_seconds()
            
            if cache_age < self.cache_ttl:
                self.logger.debug(f"Cache hit for query: {cache_key}")
                cached_result.cache_hit = True
                cached_result.status = QueryStatus.CACHED
                return cached_result
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        
        # Execute query and cache result
        result = super().execute_query(query, parameters, timeout)
        self._cache[cache_key] = (result, datetime.now())
        
        return result
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        current_time = datetime.now()
        
        expired_entries = sum(
            1 for _, cache_time in self._cache.values()
            if (current_time - cache_time).total_seconds() > self.cache_ttl
        )
        
        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "cache_ttl": self.cache_ttl,
            "cache_enabled": self.cache_enabled
        }
    
    def _generate_cache_key(self, query: str, parameters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query."""
        import hashlib
        import json
        
        # Create deterministic string representation
        cache_data = {
            "query": query,
            "parameters": parameters or {},
            "source_type": self.source_type.value
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()