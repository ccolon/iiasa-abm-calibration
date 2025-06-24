"""
Data source factory for creating and managing data source instances.

This module provides a factory pattern for creating data source instances
and managing their lifecycle, making it easy to switch between different
data sources and configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import DataSource, DataSourceType
from .oecd import OECDDataSource
from .eurostat import EurostatDataSource
from .icio import ICIODataSource
from ..config import CalibrationConfig


class DataSourceFactory:
    """
    Factory for creating data source instances.
    
    This class provides a centralized way to create and configure
    data source instances based on configuration settings.
    """
    
    @staticmethod
    def create_oecd_source(
        database_path: Union[str, Path],
        **kwargs
    ) -> OECDDataSource:
        """
        Create OECD data source instance.
        
        Args:
            database_path: Path to OECD SQLite database
            **kwargs: Additional configuration options
            
        Returns:
            Configured OECDDataSource instance
        """
        return OECDDataSource(
            database_path=Path(database_path),
            **kwargs
        )
    
    @staticmethod
    def create_eurostat_source(
        database_path: Union[str, Path],
        **kwargs
    ) -> EurostatDataSource:
        """
        Create Eurostat data source instance.
        
        Args:
            database_path: Path to database containing Eurostat tables
            **kwargs: Additional configuration options
            
        Returns:
            Configured EurostatDataSource instance
        """
        return EurostatDataSource(
            database_path=Path(database_path),
            **kwargs
        )
    
    @staticmethod
    def create_icio_source(
        data_directory: Union[str, Path],
        **kwargs
    ) -> ICIODataSource:
        """
        Create ICIO data source instance.
        
        Args:
            data_directory: Path to directory containing ICIO data files
            **kwargs: Additional configuration options
            
        Returns:
            Configured ICIODataSource instance
        """
        return ICIODataSource(
            data_directory=Path(data_directory),
            **kwargs
        )
    
    @staticmethod
    def create_from_config(
        source_type: DataSourceType,
        config: CalibrationConfig
    ) -> DataSource:
        """
        Create data source from configuration.
        
        Args:
            source_type: Type of data source to create
            config: Calibration configuration
            
        Returns:
            Configured data source instance
        """
        if source_type == DataSourceType.OECD:
            if not config.database.sqlite_path:
                raise ValueError("OECD database path not configured")
            
            return DataSourceFactory.create_oecd_source(
                database_path=config.database.sqlite_path,
                connection_timeout=config.database.connection_timeout,
                query_timeout=config.database.query_timeout,
                cache_enabled=config.cache_enabled
            )
        
        elif source_type == DataSourceType.EUROSTAT:
            if not config.database.sqlite_path:
                raise ValueError("Eurostat database path not configured")
            
            return DataSourceFactory.create_eurostat_source(
                database_path=config.database.sqlite_path,
                connection_timeout=config.database.connection_timeout,
                query_timeout=config.database.query_timeout,
                cache_enabled=config.cache_enabled
            )
        
        elif source_type == DataSourceType.ICIO:
            if not config.data_sources.icio_data_path:
                raise ValueError("ICIO data path not configured")
            
            return DataSourceFactory.create_icio_source(
                data_directory=config.data_sources.icio_data_path,
                cache_enabled=config.cache_enabled
            )
        
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    @staticmethod
    def create_all_sources(config: CalibrationConfig) -> Dict[DataSourceType, DataSource]:
        """
        Create all configured data sources.
        
        Args:
            config: Calibration configuration
            
        Returns:
            Dictionary mapping source types to instances
        """
        sources = {}
        
        # Try to create each source type
        for source_type in [DataSourceType.OECD, DataSourceType.EUROSTAT, DataSourceType.ICIO]:
            try:
                source = DataSourceFactory.create_from_config(source_type, config)
                sources[source_type] = source
            except ValueError as e:
                # Log warning but continue with other sources
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not create {source_type.value} source: {e}")
        
        return sources


def create_data_source(
    source_type: Union[str, DataSourceType],
    **kwargs
) -> DataSource:
    """
    Convenience function to create a data source.
    
    Args:
        source_type: Type of data source
        **kwargs: Configuration parameters
        
    Returns:
        Configured data source instance
    """
    if isinstance(source_type, str):
        source_type = DataSourceType(source_type)
    
    if source_type == DataSourceType.OECD:
        return DataSourceFactory.create_oecd_source(**kwargs)
    elif source_type == DataSourceType.EUROSTAT:
        return DataSourceFactory.create_eurostat_source(**kwargs)
    elif source_type == DataSourceType.ICIO:
        return DataSourceFactory.create_icio_source(**kwargs)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")


class DataSourceManager:
    """
    Manager for multiple data sources.
    
    This class provides a unified interface for working with multiple
    data sources, handling connections, and coordinating queries.
    """
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize data source manager.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self._sources: Dict[DataSourceType, DataSource] = {}
        self._connected: Dict[DataSourceType, bool] = {}
    
    def add_source(self, source_type: DataSourceType, source: DataSource) -> None:
        """Add a data source to the manager."""
        self._sources[source_type] = source
        self._connected[source_type] = False
    
    def get_source(self, source_type: DataSourceType) -> Optional[DataSource]:
        """Get a data source by type."""
        return self._sources.get(source_type)
    
    def connect_all(self) -> None:
        """Connect to all data sources."""
        for source_type, source in self._sources.items():
            try:
                source.connect()
                self._connected[source_type] = True
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to connect to {source_type.value}: {e}")
                self._connected[source_type] = False
    
    def disconnect_all(self) -> None:
        """Disconnect from all data sources."""
        for source_type, source in self._sources.items():
            try:
                source.disconnect()
                self._connected[source_type] = False
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to disconnect from {source_type.value}: {e}")
    
    def test_all_connections(self) -> Dict[DataSourceType, bool]:
        """Test all data source connections."""
        results = {}
        for source_type, source in self._sources.items():
            results[source_type] = source.test_connection()
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        status = {
            "sources": {},
            "total_sources": len(self._sources),
            "connected_sources": sum(self._connected.values())
        }
        
        for source_type, source in self._sources.items():
            status["sources"][source_type.value] = {
                "connected": self._connected.get(source_type, False),
                "type": source_type.value,
                "class": source.__class__.__name__
            }
        
        return status
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()
    
    @classmethod
    def from_config(cls, config: CalibrationConfig) -> DataSourceManager:
        """
        Create data source manager from configuration.
        
        Args:
            config: Calibration configuration
            
        Returns:
            Configured DataSourceManager instance
        """
        manager = cls(config)
        
        # Create and add all configured sources
        sources = DataSourceFactory.create_all_sources(config)
        for source_type, source in sources.items():
            manager.add_source(source_type, source)
        
        return manager