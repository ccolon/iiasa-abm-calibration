"""
Unit tests for configuration system.
"""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from macro_abm_calibration.config import CalibrationConfig, DatabaseConfig
from macro_abm_calibration.models import TimeFrame, FrequencyType


class TestCalibrationConfig:
    """Test CalibrationConfig class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = CalibrationConfig()
        
        assert config.project_name == "macro-abm-calibration"
        assert len(config.countries) == 31  # All OECD countries
        assert len(config.industries) == 19  # All NACE2 industries
        assert config.estimation_period.start_year == 1996
        assert config.calibration_period.start_year == 2010
        assert config.debug_mode is False
        assert config.cache_enabled is True
    
    def test_custom_countries(self):
        """Test configuration with custom countries."""
        config = CalibrationConfig(
            countries=["USA", "DEU", "JPN"]
        )
        
        assert config.countries == ["USA", "DEU", "JPN"]
        assert len(config.country_objects) == 3
    
    def test_custom_industries(self):
        """Test configuration with custom industries."""
        config = CalibrationConfig(
            industries=["A", "C", "G"]
        )
        
        assert config.industries == ["A", "C", "G"]
        assert len(config.industry_objects) == 3
    
    def test_invalid_country_codes(self):
        """Test validation of invalid country codes."""
        with pytest.raises(ValueError, match="Invalid country codes"):
            CalibrationConfig(
                countries=["USA", "XXX", "YYY"]  # Invalid codes
            )
    
    def test_invalid_industry_codes(self):
        """Test validation of invalid industry codes."""
        with pytest.raises(ValueError, match="Invalid industry codes"):
            CalibrationConfig(
                industries=["A", "Z", "AA"]  # Invalid codes
            )
    
    def test_calibration_period_validation(self):
        """Test calibration period must be within estimation period."""
        with pytest.raises(ValueError, match="Calibration period must be within estimation period"):
            CalibrationConfig(
                estimation_period=TimeFrame(start_year=2000, end_year=2010),
                calibration_period=TimeFrame(start_year=2015, end_year=2020)  # Outside range
            )
    
    def test_get_country_method(self):
        """Test get_country method."""
        config = CalibrationConfig()
        
        country = config.get_country("USA")
        assert country.oecd_code == "USA"
        assert country.name == "United States"
        
        with pytest.raises(ValueError, match="Country not found"):
            config.get_country("XXX")
    
    def test_get_industry_method(self):
        """Test get_industry method."""
        config = CalibrationConfig()
        
        industry = config.get_industry("C")
        assert industry.code == "C"
        assert industry.name == "Manufacturing"
        
        with pytest.raises(ValueError, match="Industry not found"):
            config.get_industry("Z")


class TestConfigurationFiles:
    """Test configuration file loading and saving."""
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        config = CalibrationConfig(
            countries=["USA", "DEU"],
            industries=["A", "C"]
        )
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = Path(f.name)
        
        try:
            # Save configuration
            config.save_to_file(config_path)
            
            # Load configuration
            loaded_config = CalibrationConfig.from_file(config_path)
            
            assert loaded_config.countries == ["USA", "DEU"]
            assert loaded_config.industries == ["A", "C"]
            
        finally:
            config_path.unlink()  # Clean up
    
    def test_unsupported_file_format(self):
        """Test error with unsupported file format."""
        config = CalibrationConfig()
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            config.save_to_file("config.txt")
        
        with pytest.raises(ValueError, match="Unsupported config file format"):
            CalibrationConfig.from_file("config.txt")


class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_default_database_config(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        
        assert config.sqlite_path is None
        assert config.connection_timeout == 30
        assert config.query_timeout == 120
    
    def test_custom_database_config(self):
        """Test custom database configuration."""
        config = DatabaseConfig(
            connection_timeout=60,
            query_timeout=300
        )
        
        assert config.connection_timeout == 60
        assert config.query_timeout == 300
    
    def test_invalid_timeout_values(self):
        """Test validation of timeout values."""
        with pytest.raises(ValueError):
            DatabaseConfig(connection_timeout=0)  # Too low
        
        with pytest.raises(ValueError):
            DatabaseConfig(query_timeout=1000)  # Too high