"""
Integration tests for data sources.

These tests verify that data sources can connect to actual data files
and retrieve expected data structures.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from macro_abm_calibration.data_sources import (
    OECDDataSource, EurostatDataSource, ICIODataSource,
    DataSourceManager, DataSourceFactory
)
from macro_abm_calibration.data_sources.base import DataSourceType, QueryResult
from macro_abm_calibration.config import CalibrationConfig
from macro_abm_calibration.models import FrequencyType


class TestOECDDataSource:
    """Test OECD data source integration."""
    
    @pytest.fixture
    def mock_database_path(self, tmp_path):
        """Create a mock SQLite database file."""
        db_path = tmp_path / "test_oecd.sqlite"
        db_path.touch()  # Create empty file
        return db_path
    
    def test_oecd_source_creation(self, mock_database_path):
        """Test OECD data source can be created."""
        source = OECDDataSource(
            database_path=mock_database_path,
            connection_timeout=10,
            query_timeout=30
        )
        
        assert source.database_path == mock_database_path
        assert source.connection_timeout == 10
        assert source.query_timeout == 30
        assert source.source_type == DataSourceType.OECD
        assert not source.is_connected
    
    @patch('macro_abm_calibration.data_sources.oecd.create_engine')
    def test_oecd_connection(self, mock_create_engine, mock_database_path):
        """Test OECD database connection."""
        # Mock SQLAlchemy engine
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        source = OECDDataSource(database_path=mock_database_path)
        
        # Test connection
        source.connect()
        
        assert source.is_connected
        mock_create_engine.assert_called_once()
        mock_engine.connect.assert_called()
    
    @patch('macro_abm_calibration.data_sources.oecd.pd.read_sql')
    @patch('macro_abm_calibration.data_sources.oecd.create_engine')
    def test_oecd_gdp_query(self, mock_create_engine, mock_read_sql, mock_database_path):
        """Test OECD GDP data query."""
        # Mock data
        mock_data = pd.DataFrame({
            'REF_AREA': ['USA', 'USA', 'DEU', 'DEU'],
            'TIME_PERIOD': ['2020-Q1', '2020-Q2', '2020-Q1', '2020-Q2'],
            'OBS_VALUE': [100.0, 102.0, 80.0, 81.0],
            'PRICE_BASE': ['V', 'V', 'V', 'V']
        })
        mock_read_sql.return_value = mock_data
        
        # Mock engine
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        source = OECDDataSource(database_path=mock_database_path)
        source.connect()
        
        # Test GDP data fetch
        result = source.fetch_gdp_data(
            countries=['USA', 'DEU'],
            start_year=2020,
            end_year=2020,
            frequency=FrequencyType.QUARTERLY
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert 'REF_AREA' in result.columns
        assert 'OBS_VALUE' in result.columns


class TestEurostatDataSource:
    """Test Eurostat data source integration."""
    
    @pytest.fixture
    def mock_database_path(self, tmp_path):
        """Create a mock database file."""
        db_path = tmp_path / "test_eurostat.sqlite"
        db_path.touch()
        return db_path
    
    def test_eurostat_source_creation(self, mock_database_path):
        """Test Eurostat data source creation."""
        source = EurostatDataSource(database_path=mock_database_path)
        
        assert source.database_path == mock_database_path
        assert source.source_type == DataSourceType.EUROSTAT
        assert not source.is_connected
    
    @patch('macro_abm_calibration.data_sources.eurostat.pd.read_sql')
    @patch('macro_abm_calibration.data_sources.eurostat.create_engine')
    def test_eurostat_exchange_rates(self, mock_create_engine, mock_read_sql, mock_database_path):
        """Test Eurostat exchange rate query."""
        # Mock exchange rate data
        mock_data = pd.DataFrame({
            'currency': ['USD', 'USD', 'GBP', 'GBP'],
            'TIME_PERIOD': ['2020-Q1', '2020-Q2', '2020-Q1', '2020-Q2'],
            'OBS_VALUE': [1.10, 1.12, 0.85, 0.87]
        })
        mock_read_sql.return_value = mock_data
        
        # Mock engine
        mock_engine = Mock()
        mock_conn = Mock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        
        source = EurostatDataSource(database_path=mock_database_path)
        source.connect()
        
        # Test exchange rate fetch
        result = source.fetch_exchange_rates(
            countries=['USA', 'GBR'],
            start_year=2020,
            end_year=2020,
            frequency=FrequencyType.QUARTERLY
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'currency' in result.columns
        assert 'OBS_VALUE' in result.columns


class TestICIODataSource:
    """Test ICIO data source integration."""
    
    @pytest.fixture
    def mock_icio_directory(self, tmp_path):
        """Create a mock ICIO data directory."""
        icio_dir = tmp_path / "icio_data"
        icio_dir.mkdir()
        
        # Create mock .mat file
        mat_file = icio_dir / "oecd_ICIOs_SML_double.mat"
        mat_file.touch()
        
        return icio_dir
    
    def test_icio_source_creation(self, mock_icio_directory):
        """Test ICIO data source creation."""
        source = ICIODataSource(data_directory=mock_icio_directory)
        
        assert source.data_directory == mock_icio_directory
        assert source.source_type == DataSourceType.ICIO
        assert not source.is_connected
    
    @patch('macro_abm_calibration.data_sources.icio.scipy.io.loadmat')
    def test_icio_connection(self, mock_loadmat, mock_icio_directory):
        """Test ICIO data loading."""
        # Mock MATLAB data
        mock_data = {
            'oecd_ICIOs_SML_double': [[np.random.rand(1000, 1000) for _ in range(26)]],
            'rowNames': [[np.array([f'ROW_{i}']) for i in range(1000)] for _ in range(26)],
            'columnNames': [[np.array([f'COL_{i}']) for i in range(1000)] for _ in range(26)]
        }
        mock_loadmat.return_value = mock_data
        
        source = ICIODataSource(data_directory=mock_icio_directory)
        source.connect()
        
        assert source.is_connected
        assert source._available_years == list(range(1995, 2021))
        mock_loadmat.assert_called_once()


class TestDataSourceFactory:
    """Test data source factory."""
    
    def test_create_oecd_source(self, tmp_path):
        """Test OECD source creation via factory."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        
        source = DataSourceFactory.create_oecd_source(
            database_path=db_path,
            connection_timeout=20
        )
        
        assert isinstance(source, OECDDataSource)
        assert source.database_path == db_path
        assert source.connection_timeout == 20
    
    def test_create_from_config(self, tmp_path):
        """Test source creation from configuration."""
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        
        # Create minimal config
        config = CalibrationConfig()
        config.database.sqlite_path = db_path
        
        source = DataSourceFactory.create_from_config(DataSourceType.OECD, config)
        
        assert isinstance(source, OECDDataSource)
        assert source.database_path == db_path
    
    def test_create_all_sources(self, tmp_path):
        """Test creating all sources from config."""
        # Create mock data files
        db_path = tmp_path / "test.sqlite"
        db_path.touch()
        
        icio_dir = tmp_path / "icio"
        icio_dir.mkdir()
        (icio_dir / "oecd_ICIOs_SML_double.mat").touch()
        
        # Create config
        config = CalibrationConfig()
        config.database.sqlite_path = db_path
        config.data_sources.icio_data_path = icio_dir
        
        sources = DataSourceFactory.create_all_sources(config)
        
        assert DataSourceType.OECD in sources
        assert DataSourceType.EUROSTAT in sources
        assert DataSourceType.ICIO in sources
        assert isinstance(sources[DataSourceType.OECD], OECDDataSource)
        assert isinstance(sources[DataSourceType.EUROSTAT], EurostatDataSource)
        assert isinstance(sources[DataSourceType.ICIO], ICIODataSource)


class TestDataSourceManager:
    """Test data source manager."""
    
    def test_manager_creation(self):
        """Test manager creation."""
        config = CalibrationConfig()
        manager = DataSourceManager(config)
        
        assert manager.config == config
        assert len(manager._sources) == 0
    
    def test_add_and_get_source(self):
        """Test adding and retrieving sources."""
        config = CalibrationConfig()
        manager = DataSourceManager(config)
        
        # Create mock source
        mock_source = Mock(spec=OECDDataSource)
        manager.add_source(DataSourceType.OECD, mock_source)
        
        retrieved = manager.get_source(DataSourceType.OECD)
        assert retrieved == mock_source
    
    def test_manager_status(self):
        """Test manager status reporting."""
        config = CalibrationConfig()
        manager = DataSourceManager(config)
        
        # Add mock source
        mock_source = Mock(spec=OECDDataSource)
        manager.add_source(DataSourceType.OECD, mock_source)
        
        status = manager.get_status()
        
        assert status["total_sources"] == 1
        assert status["connected_sources"] == 0
        assert DataSourceType.OECD.value in status["sources"]
    
    @patch('macro_abm_calibration.data_sources.factory.DataSourceFactory.create_all_sources')
    def test_manager_from_config(self, mock_create_sources):
        """Test manager creation from config."""
        # Mock sources
        mock_oecd = Mock(spec=OECDDataSource)
        mock_eurostat = Mock(spec=EurostatDataSource)
        mock_create_sources.return_value = {
            DataSourceType.OECD: mock_oecd,
            DataSourceType.EUROSTAT: mock_eurostat
        }
        
        config = CalibrationConfig()
        manager = DataSourceManager.from_config(config)
        
        assert len(manager._sources) == 2
        assert manager.get_source(DataSourceType.OECD) == mock_oecd
        assert manager.get_source(DataSourceType.EUROSTAT) == mock_eurostat


@pytest.mark.integration
class TestDataSourceIntegration:
    """Integration tests requiring actual data files."""
    
    def test_full_data_pipeline(self, tmp_path):
        """Test complete data source pipeline."""
        # This test would require actual data files
        # For now, it's a placeholder for future integration testing
        pytest.skip("Requires actual OECD/ICIO data files")
    
    def test_cross_source_consistency(self):
        """Test data consistency across sources."""
        # This would test that exchange rates from Eurostat
        # are consistent with those used in OECD calculations
        pytest.skip("Requires actual data files and complex validation")
    
    def test_performance_benchmarks(self):
        """Test data source performance."""
        # This would benchmark query performance against
        # the original MATLAB implementation
        pytest.skip("Performance testing requires full dataset")