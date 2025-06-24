"""
Unit tests for data processors.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from macro_abm_calibration.processors import (
    CurrencyConverter, IndustryAggregator, DataHarmonizer, CalibrationPipeline
)
from macro_abm_calibration.processors.base import ProcessingStatus
from macro_abm_calibration.processors.utils import calculate_deflator, interpolate_missing_data
from macro_abm_calibration.data_sources import EurostatDataSource
from macro_abm_calibration.config import CalibrationConfig


class TestProcessorUtils:
    """Test processor utility functions."""
    
    def test_calculate_deflator_series(self):
        """Test deflator calculation with pandas Series."""
        nominal = pd.Series([100, 110, 121])
        real = pd.Series([100, 105, 110])
        
        deflator = calculate_deflator(nominal, real)
        
        expected = pd.Series([1.0, 1.048, 1.1])
        pd.testing.assert_series_equal(deflator, expected, rtol=1e-3)
    
    def test_calculate_deflator_arrays(self):
        """Test deflator calculation with numpy arrays."""
        nominal = np.array([100, 110, 121])
        real = np.array([100, 105, 110])
        
        deflator = calculate_deflator(nominal, real)
        
        expected = np.array([1.0, 1.048, 1.1])
        np.testing.assert_array_almost_equal(deflator, expected, decimal=3)
    
    def test_calculate_deflator_zero_division(self):
        """Test deflator calculation with zero values."""
        nominal = pd.Series([100, 110, 121])
        real = pd.Series([100, 0, 110])
        
        deflator = calculate_deflator(nominal, real)
        
        assert deflator.iloc[0] == 1.0
        assert pd.isna(deflator.iloc[1])  # Should be NaN for zero real value
        assert deflator.iloc[2] == 1.1
    
    def test_interpolate_missing_data_linear(self):
        """Test linear interpolation of missing data."""
        data = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        
        result = interpolate_missing_data(data, method="linear")
        
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_interpolate_missing_data_nearest(self):
        """Test nearest neighbor interpolation."""
        data = pd.Series([1.0, np.nan, np.nan, 4.0])
        
        result = interpolate_missing_data(data, method="nearest")
        
        # Should forward/backward fill
        assert not result.isna().any()
        assert result.iloc[0] == 1.0
        assert result.iloc[-1] == 4.0


class TestCurrencyConverter:
    """Test currency conversion processor."""
    
    @pytest.fixture
    def mock_eurostat_source(self):
        """Create mock Eurostat data source."""
        mock_source = Mock(spec=EurostatDataSource)
        
        # Mock USD/EUR rates
        usd_eur_rates = pd.DataFrame({
            "TIME_PERIOD": ["2020-Q1", "2020-Q2"],
            "OBS_VALUE": [1.1, 1.12]
        })
        mock_source.fetch_usd_eur_rates.return_value = usd_eur_rates
        
        # Mock exchange rates
        exchange_rates = pd.DataFrame({
            "TIME_PERIOD": ["2020-Q1", "2020-Q2"],
            "OBS_VALUE": [0.85, 0.87],
            "currency": "GBP"
        })
        mock_source.fetch_exchange_rates.return_value = exchange_rates
        
        return mock_source
    
    def test_currency_converter_creation(self, mock_eurostat_source):
        """Test currency converter initialization."""
        converter = CurrencyConverter(
            eurostat_source=mock_eurostat_source,
            target_currency="USD"
        )
        
        assert converter.target_currency == "USD"
        assert converter.eurostat_source == mock_eurostat_source
        assert "USA" in converter.get_supported_currencies()
    
    def test_currency_conversion_usd(self, mock_eurostat_source):
        """Test conversion for USD (should be 1.0)."""
        converter = CurrencyConverter(mock_eurostat_source)
        
        data = pd.DataFrame({
            "REF_AREA": ["USA", "USA"],
            "TIME_PERIOD": ["2020-Q1", "2020-Q2"],
            "OBS_VALUE": [100.0, 110.0]
        })
        
        parameters = {
            "countries": ["USA"],
            "start_year": 2020,
            "end_year": 2020,
            "frequency": "Q",
            "value_columns": ["OBS_VALUE"]
        }
        
        result = converter.process(data, parameters)
        
        assert result.is_success
        assert isinstance(result.data, pd.DataFrame)
    
    def test_input_validation(self, mock_eurostat_source):
        """Test input validation for currency converter."""
        converter = CurrencyConverter(mock_eurostat_source)
        
        # Test missing countries parameter
        validation = converter.validate_input(pd.DataFrame(), {})
        assert not validation.is_valid
        assert "Countries parameter is required" in validation.errors
        
        # Test invalid country codes
        validation = converter.validate_input(pd.DataFrame(), {"countries": ["XXX"]})
        assert not validation.is_valid
        assert "Invalid country codes" in validation.errors[0]


class TestIndustryAggregator:
    """Test industry aggregation processor."""
    
    def test_industry_aggregator_creation(self):
        """Test industry aggregator initialization."""
        aggregator = IndustryAggregator()
        
        assert aggregator.n_isic == 44
        assert aggregator.n_nace == 18
        assert "A" in aggregator.aggregation_mapping
    
    def test_aggregate_vector(self):
        """Test aggregation of industry vector."""
        aggregator = IndustryAggregator()
        
        # Create test data with ISIC Rev4 industries
        isic_data = np.random.rand(44, 3)  # 44 industries, 3 time periods
        data = pd.DataFrame(
            isic_data,
            index=aggregator.ISIC_REV4_INDUSTRIES
        )
        
        parameters = {"data_type": "vector"}
        result = aggregator.process(data, parameters)
        
        assert result.is_success
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.shape[0] == 18  # NACE2 sectors
    
    def test_aggregate_matrix(self):
        """Test aggregation of industry matrix."""
        aggregator = IndustryAggregator()
        
        # Create test matrix
        isic_matrix = np.random.rand(44, 44)
        data = pd.DataFrame(
            isic_matrix,
            index=aggregator.ISIC_REV4_INDUSTRIES,
            columns=aggregator.ISIC_REV4_INDUSTRIES
        )
        
        parameters = {"data_type": "matrix"}
        result = aggregator.process(data, parameters)
        
        assert result.is_success
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.shape == (18, 18)  # NACE2 x NACE2
    
    def test_aggregation_info(self):
        """Test aggregation information."""
        aggregator = IndustryAggregator()
        
        info = aggregator.get_aggregation_info()
        
        assert info["n_isic_industries"] == 44
        assert info["n_nace_sectors"] == 18
        assert info["compression_ratio"] > 1


class TestDataHarmonizer:
    """Test data harmonization processor."""
    
    def test_data_harmonizer_creation(self):
        """Test data harmonizer initialization."""
        harmonizer = DataHarmonizer(
            default_interpolation_method="linear",
            handle_missing_data=True
        )
        
        assert harmonizer.default_interpolation_method == "linear"
        assert harmonizer.handle_missing_data is True
    
    def test_deflator_calculation(self):
        """Test deflator calculation in harmonizer."""
        harmonizer = DataHarmonizer()
        
        data = pd.DataFrame({
            "nominal_gdp": [100, 110, 121],
            "real_gdp": [100, 105, 110],
            "TIME_PERIOD": ["2020", "2021", "2022"]
        })
        
        parameters = {
            "calculate_deflators": True,
            "variables": ["gdp"]
        }
        
        result = harmonizer.process(data, parameters)
        
        assert result.is_success
        assert "gdp_deflator" in result.data.columns
    
    def test_missing_data_handling(self):
        """Test missing data handling."""
        harmonizer = DataHarmonizer()
        
        data = pd.DataFrame({
            "value": [1.0, np.nan, 3.0, np.nan, 5.0],
            "TIME_PERIOD": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4", "2021-Q1"]
        })
        
        result = harmonizer.process(data)
        
        assert result.is_success
        # Should have interpolated missing values
        assert not result.data["value"].isna().any()
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency validation."""
        harmonizer = DataHarmonizer()
        
        # Regular time series
        regular_data = pd.DataFrame({
            "TIME_PERIOD": ["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"],
            "value": [1, 2, 3, 4]
        })
        
        validation = harmonizer.validate_temporal_consistency(regular_data)
        assert validation.is_valid
        
        # Irregular time series
        irregular_data = pd.DataFrame({
            "TIME_PERIOD": ["2020-Q1", "2020-Q3", "2021-Q1"],  # Missing Q2 and Q4
            "value": [1, 3, 5]
        })
        
        validation = harmonizer.validate_temporal_consistency(irregular_data)
        # Should have warnings about missing periods
        assert len(validation.warnings) > 0


class TestCalibrationPipeline:
    """Test calibration pipeline orchestrator."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = CalibrationConfig(
            countries=["USA", "DEU"],
            industries=["A", "C", "G"]
        )
        return config
    
    @pytest.fixture
    def mock_data_source_manager(self):
        """Create mock data source manager."""
        manager = Mock()
        
        # Mock OECD source
        oecd_source = Mock()
        oecd_source.fetch_gdp_data.return_value = pd.DataFrame({
            "REF_AREA": ["USA", "USA"],
            "TIME_PERIOD": ["2020-Q1", "2020-Q2"],
            "OBS_VALUE": [100, 105]
        })
        oecd_source.fetch_consumption_data.return_value = pd.DataFrame()
        oecd_source.fetch_unemployment_data.return_value = pd.DataFrame()
        
        # Mock Eurostat source
        eurostat_source = Mock()
        eurostat_source.fetch_exchange_rates.return_value = pd.DataFrame()
        
        # Mock ICIO source
        icio_source = Mock()
        icio_source.shrink_icio_tables.return_value = {}
        
        manager.get_source.side_effect = lambda source_type: {
            "oecd": oecd_source,
            "eurostat": eurostat_source,
            "icio": icio_source
        }.get(source_type.value)
        
        manager.connect_all.return_value = None
        manager.disconnect_all.return_value = None
        
        return manager
    
    def test_pipeline_creation(self, mock_config, mock_data_source_manager):
        """Test pipeline initialization."""
        pipeline = CalibrationPipeline(
            config=mock_config,
            data_source_manager=mock_data_source_manager
        )
        
        assert pipeline.config == mock_config
        assert pipeline.data_source_manager == mock_data_source_manager
        assert pipeline.currency_converter is not None
        assert pipeline.industry_aggregator is not None
        assert pipeline.data_harmonizer is not None
    
    def test_pipeline_status(self, mock_config, mock_data_source_manager):
        """Test pipeline status reporting."""
        pipeline = CalibrationPipeline(
            config=mock_config,
            data_source_manager=mock_data_source_manager
        )
        
        status = pipeline.get_pipeline_status()
        
        assert "config" in status
        assert "data_sources" in status
        assert "processors" in status
        assert "currency_converter" in status["processors"]
    
    @patch('macro_abm_calibration.processors.pipeline.CalibrationPipeline._step_1_extract_raw_data')
    @patch('macro_abm_calibration.processors.pipeline.CalibrationPipeline._step_2_process_calibration_data')
    @patch('macro_abm_calibration.processors.pipeline.CalibrationPipeline._step_3_process_icio_data')
    @patch('macro_abm_calibration.processors.pipeline.CalibrationPipeline._step_4_create_final_datasets')
    def test_full_pipeline_execution(
        self, mock_step4, mock_step3, mock_step2, mock_step1,
        mock_config, mock_data_source_manager
    ):
        """Test full pipeline execution."""
        # Mock step returns
        mock_step1.return_value = {"raw_data": pd.DataFrame()}
        mock_step2.return_value = {"processed_data": pd.DataFrame()}
        mock_step3.return_value = {"icio_data": {}}
        mock_step4.return_value = {"final_data": {}}
        
        pipeline = CalibrationPipeline(
            config=mock_config,
            data_source_manager=mock_data_source_manager
        )
        
        result = pipeline.run_full_pipeline()
        
        assert result.is_success
        assert result.status == ProcessingStatus.COMPLETED
        
        # Verify all steps were called
        mock_step1.assert_called_once()
        mock_step2.assert_called_once()
        mock_step3.assert_called_once()
        mock_step4.assert_called_once()
    
    def test_partial_pipeline_execution(self, mock_config, mock_data_source_manager):
        """Test partial pipeline execution."""
        pipeline = CalibrationPipeline(
            config=mock_config,
            data_source_manager=mock_data_source_manager
        )
        
        # Mock intermediate results
        pipeline._intermediate_results["raw_data"] = {"test": pd.DataFrame()}
        
        with patch.object(pipeline, '_step_2_process_calibration_data') as mock_step2:
            mock_step2.return_value = {"processed": pd.DataFrame()}
            
            results = pipeline.run_partial_pipeline(["process_calibration_data"])
            
            assert "calibration_data" in results
            mock_step2.assert_called_once()


@pytest.mark.integration
class TestProcessorIntegration:
    """Integration tests for processor pipeline."""
    
    def test_processor_chaining(self):
        """Test chaining processors together."""
        # This would test the actual integration between processors
        pytest.skip("Integration testing requires full data setup")
    
    def test_end_to_end_processing(self):
        """Test end-to-end data processing."""
        # This would test the complete pipeline with real data
        pytest.skip("End-to-end testing requires actual data files")