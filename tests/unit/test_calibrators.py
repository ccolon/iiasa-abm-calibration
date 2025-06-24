"""
Unit tests for calibrators module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from macro_abm_calibration.calibrators import (
    Calibrator, CalibrationResult, CalibrationStatus,
    ABMParameterEstimator, InitialConditionsSetter, ModelValidator
)
from macro_abm_calibration.calibrators.base import (
    EconomicCalibrator, CalibrationManager, CalibrationMetadata
)
from macro_abm_calibration.calibrators.initial_conditions import AgentType, MarketConditions
from macro_abm_calibration.calibrators.validation import ValidationLevel, ValidationCheck, ValidationSummary
from macro_abm_calibration.calibrators.export import CalibrationExporter, CalibrationVisualizer
from macro_abm_calibration.calibrators.utils import (
    hp_filter, calculate_output_gap, estimate_ar_process, calculate_gini_coefficient
)


class TestCalibrationBase:
    """Test base calibration classes."""
    
    def test_calibration_metadata_creation(self):
        """Test calibration metadata creation."""
        metadata = CalibrationMetadata(
            calibrator_name="TestCalibrator",
            operation_id="test_001",
            parameters={"param1": 1.0}
        )
        
        assert metadata.calibrator_name == "TestCalibrator"
        assert metadata.operation_id == "test_001"
        assert metadata.parameters["param1"] == 1.0
        assert isinstance(metadata.timestamp, datetime)
    
    def test_calibration_result_creation(self):
        """Test calibration result creation."""
        metadata = CalibrationMetadata(
            calibrator_name="TestCalibrator",
            operation_id="test_001"
        )
        
        result = CalibrationResult(
            status=CalibrationStatus.COMPLETED,
            data={"test": "data"},
            metadata=metadata
        )
        
        assert result.is_success
        assert not result.has_errors
        assert not result.has_warnings
        
        result.add_error("Test error")
        assert result.has_errors
        assert result.status == CalibrationStatus.FAILED
    
    def test_economic_calibrator_validation(self):
        """Test economic calibrator validation methods."""
        calibrator = EconomicCalibrator("TestEconomicCalibrator")
        
        # Test economic data validation
        data = pd.DataFrame({
            "REF_AREA": ["USA", "USA", "GBR", "GBR"],
            "TIME_PERIOD": ["2020", "2021", "2020", "2021"],
            "gdp": [100, 105, 80, 82],
            "consumption": [70, 73, 60, 61]
        })
        
        errors = calibrator.validate_economic_data(data)
        assert len(errors) == 0  # Should pass validation
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            "gdp": [100, 105, 0, 0],  # All zeros should trigger error
            "consumption": [70, 73, 60, 61]
        })
        
        errors = calibrator.validate_economic_data(invalid_data)
        assert len(errors) > 0
    
    def test_calibration_manager(self):
        """Test calibration manager functionality."""
        manager = CalibrationManager()
        
        # Mock calibrator
        mock_calibrator = Mock(spec=Calibrator)
        mock_calibrator.calibrate.return_value = CalibrationResult(
            status=CalibrationStatus.COMPLETED,
            data={"result": "success"},
            metadata=CalibrationMetadata("MockCalibrator", "op_001")
        )
        
        # Register calibrator
        manager.register_calibrator("mock", mock_calibrator)
        assert "mock" in manager.calibrators
        
        # Run calibration sequence
        data = {"test": pd.DataFrame()}
        results = manager.run_calibration_sequence(data, ["mock"])
        
        assert "mock" in results
        assert results["mock"].is_success


class TestABMParameterEstimator:
    """Test ABM parameter estimator."""
    
    @pytest.fixture
    def sample_economic_data(self):
        """Create sample economic data for testing."""
        dates = pd.date_range("2010", "2020", freq="Q")
        
        data = pd.DataFrame({
            "REF_AREA": ["USA"] * len(dates),
            "TIME_PERIOD": [d.strftime("%Y-Q%q") for d in dates],
            "gdp": np.cumsum(np.random.normal(0.02, 0.01, len(dates))) + 100,
            "consumption": np.cumsum(np.random.normal(0.015, 0.008, len(dates))) + 70,
            "interest_rate": 2.0 + np.random.normal(0, 0.5, len(dates)),
            "inflation_rate": 2.0 + np.random.normal(0, 0.3, len(dates)),
            "unemployment_rate": 5.0 + np.random.normal(0, 0.5, len(dates))
        })
        
        return data
    
    def test_parameter_estimator_creation(self):
        """Test parameter estimator initialization."""
        estimator = ABMParameterEstimator()
        
        assert estimator.name == "ABMParameterEstimator"
        assert isinstance(estimator.PARAMETER_BOUNDS, dict)
        assert "taylor_inflation_response" in estimator.PARAMETER_BOUNDS
    
    def test_taylor_rule_estimation(self, sample_economic_data):
        """Test Taylor rule parameter estimation."""
        estimator = ABMParameterEstimator()
        
        parameters = {
            "countries": ["USA"],
            "parameter_groups": ["taylor_rule"]
        }
        
        result = estimator.calibrate(sample_economic_data, parameters)
        
        assert result.is_success
        assert "estimated_parameters" in result.data
        assert "USA" in result.data["estimated_parameters"]
        
        usa_params = result.data["estimated_parameters"]["USA"]
        assert "taylor_inflation_response" in usa_params
        assert "taylor_output_response" in usa_params
        assert "taylor_smoothing" in usa_params
        
        # Check bounds
        assert 1.0 <= usa_params["taylor_inflation_response"] <= 5.0
        assert 0.0 <= usa_params["taylor_output_response"] <= 2.0
    
    def test_firm_parameter_estimation(self, sample_economic_data):
        """Test firm parameter estimation."""
        estimator = ABMParameterEstimator()
        
        parameters = {
            "countries": ["USA"],
            "parameter_groups": ["firm_behavior"]
        }
        
        result = estimator.calibrate(sample_economic_data, parameters)
        
        assert result.is_success
        usa_params = result.data["estimated_parameters"]["USA"]
        
        assert "price_adjustment_speed" in usa_params
        assert "investment_sensitivity" in usa_params
        assert "markup_elasticity" in usa_params
    
    def test_household_parameter_estimation(self, sample_economic_data):
        """Test household parameter estimation."""
        estimator = ABMParameterEstimator()
        
        parameters = {
            "countries": ["USA"],
            "parameter_groups": ["household_behavior"]
        }
        
        result = estimator.calibrate(sample_economic_data, parameters)
        
        assert result.is_success
        usa_params = result.data["estimated_parameters"]["USA"]
        
        assert "marginal_propensity_consume" in usa_params
        assert "wealth_effect" in usa_params
        assert "labor_supply_elasticity" in usa_params
        
        # Check MPC bounds
        mpc = usa_params["marginal_propensity_consume"]
        assert 0.3 <= mpc <= 0.95
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        estimator = ABMParameterEstimator()
        
        # Valid parameters
        valid_params = {
            "USA": {
                "taylor_inflation_response": 1.5,
                "taylor_output_response": 0.5,
                "marginal_propensity_consume": 0.7
            }
        }
        
        warnings = estimator.validate_estimated_parameters(valid_params)
        assert len(warnings) == 0
        
        # Invalid parameters
        invalid_params = {
            "USA": {
                "taylor_inflation_response": 0.8,  # Below 1.0
                "marginal_propensity_consume": 1.2  # Above 0.95
            }
        }
        
        warnings = estimator.validate_estimated_parameters(invalid_params)
        assert len(warnings) > 0


class TestInitialConditionsSetter:
    """Test initial conditions setter."""
    
    @pytest.fixture
    def sample_calibration_data(self):
        """Create sample calibration data."""
        return pd.DataFrame({
            "REF_AREA": ["USA"] * 10,
            "TIME_PERIOD": [f"2020-Q{i%4+1}" for i in range(10)],
            "gdp": [25e12] * 10,  # $25 trillion
            "unemployment_rate": [5.0] * 10,
            "interest_rate": [2.0] * 10
        })
    
    def test_initial_conditions_setter_creation(self):
        """Test initial conditions setter initialization."""
        setter = InitialConditionsSetter()
        
        assert setter.name == "InitialConditionsSetter"
        assert AgentType.HOUSEHOLD in setter.DEFAULT_POPULATIONS
        assert AgentType.FIRM in setter.DEFAULT_POPULATIONS
    
    def test_agent_population_setup(self, sample_calibration_data):
        """Test agent population setup."""
        setter = InitialConditionsSetter()
        
        parameters = {
            "countries": ["USA"],
            "population_scaling": {"USA": 1.0}
        }
        
        result = setter.calibrate(sample_calibration_data, parameters)
        
        assert result.is_success
        assert "initial_conditions" in result.data
        assert "USA" in result.data["initial_conditions"]
        
        usa_conditions = result.data["initial_conditions"]["USA"]
        assert "agent_populations" in usa_conditions
        
        populations = usa_conditions["agent_populations"]
        assert AgentType.HOUSEHOLD in populations
        assert AgentType.FIRM in populations
        assert AgentType.BANK in populations
        
        # Check population counts
        household_pop = populations[AgentType.HOUSEHOLD]
        assert household_pop.count > 0
        assert household_pop.agent_type == AgentType.HOUSEHOLD
    
    def test_market_conditions_setup(self, sample_calibration_data):
        """Test market conditions setup."""
        setter = InitialConditionsSetter()
        
        parameters = {"countries": ["USA"]}
        result = setter.calibrate(sample_calibration_data, parameters)
        
        usa_conditions = result.data["initial_conditions"]["USA"]
        market_conditions = usa_conditions["market_conditions"]
        
        assert isinstance(market_conditions, MarketConditions)
        assert market_conditions.interest_rate > 0
        assert 0 <= market_conditions.unemployment_rate <= 1
        assert market_conditions.price_level > 0
    
    def test_balance_sheet_creation(self, sample_calibration_data):
        """Test balance sheet creation."""
        setter = InitialConditionsSetter()
        
        parameters = {"countries": ["USA"]}
        result = setter.calibrate(sample_calibration_data, parameters)
        
        usa_conditions = result.data["initial_conditions"]["USA"]
        balance_sheets = usa_conditions["balance_sheets"]
        
        assert "households" in balance_sheets
        assert "firms" in balance_sheets
        assert "banks" in balance_sheets
        assert "government" in balance_sheets
        
        # Check household balance sheet
        household_bs = balance_sheets["households"]
        assert household_bs["count"] > 0
        assert household_bs["total_wealth"] > 0
        assert len(household_bs["wealth_distribution"]) == household_bs["count"]
    
    def test_initial_conditions_validation(self):
        """Test initial conditions validation."""
        setter = InitialConditionsSetter()
        
        # Valid initial conditions
        valid_ic = {
            "USA": {
                "agent_populations": {
                    AgentType.HOUSEHOLD: Mock(count=10000),
                    AgentType.FIRM: Mock(count=1000)
                },
                "balance_sheets": {
                    "households": {"total_wealth": 1000000},
                    "firms": {"total_assets": 500000}
                }
            }
        }
        
        warnings = setter.validate_initial_conditions(valid_ic)
        assert len(warnings) == 0
        
        # Invalid initial conditions
        invalid_ic = {
            "USA": {
                "agent_populations": {
                    AgentType.HOUSEHOLD: Mock(count=0),  # Zero population
                    AgentType.FIRM: Mock(count=1)
                }
            }
        }
        
        warnings = setter.validate_initial_conditions(invalid_ic)
        assert len(warnings) > 0


class TestModelValidator:
    """Test model validator."""
    
    @pytest.fixture
    def sample_calibration_results(self):
        """Create sample calibration results for validation."""
        return {
            "estimated_parameters": {
                "USA": {
                    "taylor_inflation_response": 1.5,
                    "taylor_output_response": 0.5,
                    "marginal_propensity_consume": 0.7
                },
                "GBR": {
                    "taylor_inflation_response": 1.3,
                    "taylor_output_response": 0.3,
                    "marginal_propensity_consume": 0.6
                }
            },
            "initial_conditions": {
                "USA": {
                    "market_conditions": Mock(unemployment_rate=0.05, interest_rate=0.02)
                }
            }
        }
    
    def test_model_validator_creation(self):
        """Test model validator initialization."""
        validator = ModelValidator()
        
        assert validator.name == "ModelValidator"
        assert isinstance(validator.THEORY_BOUNDS, dict)
        assert "taylor_inflation_response" in validator.THEORY_BOUNDS
    
    def test_economic_theory_validation(self, sample_calibration_results):
        """Test economic theory validation."""
        validator = ModelValidator()
        
        result = validator.calibrate(sample_calibration_results)
        
        assert result.is_success or result.status == CalibrationStatus.VALIDATION_FAILED
        assert "validation_summary" in result.data
        
        summary = result.data["validation_summary"]
        assert isinstance(summary, ValidationSummary)
        assert summary.total_checks > 0
    
    def test_parameter_range_validation(self, sample_calibration_results):
        """Test parameter range validation."""
        validator = ModelValidator()
        
        # Add out-of-range parameter
        sample_calibration_results["estimated_parameters"]["USA"]["taylor_inflation_response"] = 0.5  # Below 1.0
        
        result = validator.calibrate(sample_calibration_results)
        summary = result.data["validation_summary"]
        
        # Should have at least one failed check
        assert summary.failed_checks > 0
    
    def test_validation_summary_compilation(self):
        """Test validation summary compilation."""
        validator = ModelValidator()
        
        checks = [
            ValidationCheck(
                name="test_check_1",
                level=ValidationLevel.ERROR,
                passed=True,
                message="Test passed"
            ),
            ValidationCheck(
                name="test_check_2",
                level=ValidationLevel.WARNING,
                passed=False,
                message="Test warning"
            ),
            ValidationCheck(
                name="test_check_3",
                level=ValidationLevel.ERROR,
                passed=False,
                message="Test failed"
            )
        ]
        
        summary = validator._compile_validation_summary(checks)
        
        assert summary.total_checks == 3
        assert summary.passed_checks == 1
        assert summary.failed_checks == 2
        assert summary.warnings == 1
        assert summary.errors == 1
        assert not summary.overall_passed  # Should fail due to error


class TestCalibrationUtils:
    """Test calibration utility functions."""
    
    def test_hp_filter(self):
        """Test Hodrick-Prescott filter."""
        # Create trending series
        t = np.arange(100)
        trend = 0.02 * t
        cycle = 0.5 * np.sin(0.3 * t)
        series = pd.Series(trend + cycle)
        
        # Apply HP filter
        filtered_cycle = hp_filter(series)
        
        assert len(filtered_cycle) == len(series)
        assert isinstance(filtered_cycle, pd.Series)
        
        # Test with return_trend=True
        cycle_result, trend_result = hp_filter(series, return_trend=True)
        assert len(cycle_result) == len(trend_result) == len(series)
    
    def test_calculate_output_gap(self):
        """Test output gap calculation."""
        # Create GDP series with trend and cycle
        t = np.arange(50)
        log_gdp = 10 + 0.02 * t + 0.1 * np.sin(0.5 * t)
        gdp = pd.Series(np.exp(log_gdp))
        
        # Calculate output gap
        gap = calculate_output_gap(gdp)
        
        assert len(gap) == len(gdp)
        assert isinstance(gap, pd.Series)
        
        # Test different methods
        gap_linear = calculate_output_gap(gdp, method="linear_trend")
        gap_quad = calculate_output_gap(gdp, method="quadratic_trend")
        
        assert len(gap_linear) == len(gap_quad) == len(gdp)
    
    def test_estimate_ar_process(self):
        """Test AR process estimation."""
        # Generate AR(1) process
        np.random.seed(42)
        n = 100
        phi = 0.7
        sigma = 1.0
        
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t-1] + np.random.normal(0, sigma)
        
        # Estimate AR parameters
        result = estimate_ar_process(y, max_lags=3)
        
        assert "coefficients" in result
        assert "intercept" in result
        assert "sigma" in result
        assert "selected_lags" in result
        
        # Check if estimated coefficient is close to true value
        estimated_phi = result["coefficients"][0]
        assert abs(estimated_phi - phi) < 0.2  # Allow some estimation error
    
    def test_calculate_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_dist = np.ones(100)
        gini_equal = calculate_gini_coefficient(equal_dist)
        assert abs(gini_equal) < 0.01  # Should be close to 0
        
        # Perfect inequality
        unequal_dist = np.zeros(100)
        unequal_dist[-1] = 100  # One person has everything
        gini_unequal = calculate_gini_coefficient(unequal_dist)
        assert gini_unequal > 0.9  # Should be close to 1
        
        # Normal distribution
        np.random.seed(42)
        normal_dist = np.abs(np.random.normal(1, 0.3, 1000))
        gini_normal = calculate_gini_coefficient(normal_dist)
        assert 0.1 < gini_normal < 0.4  # Reasonable range for normal distribution


class TestCalibrationExport:
    """Test calibration export functionality."""
    
    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return tmp_path / "output"
    
    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters for export testing."""
        return {
            "USA": {
                "taylor_inflation_response": 1.5,
                "taylor_output_response": 0.5,
                "marginal_propensity_consume": 0.7
            },
            "GBR": {
                "taylor_inflation_response": 1.3,
                "taylor_output_response": 0.3,
                "marginal_propensity_consume": 0.6
            }
        }
    
    def test_calibration_exporter_creation(self, temp_output_dir):
        """Test calibration exporter initialization."""
        exporter = CalibrationExporter(temp_output_dir)
        
        assert exporter.output_directory == temp_output_dir
        assert temp_output_dir.exists()
    
    def test_parameter_export_json(self, temp_output_dir, sample_parameters):
        """Test parameter export to JSON."""
        exporter = CalibrationExporter(temp_output_dir)
        
        created_files = exporter.export_parameters(sample_parameters, format="json")
        
        assert len(created_files) == 1
        json_file = created_files[0]
        assert json_file.endswith(".json")
        
        # Verify file exists and contains data
        import json
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert "USA" in loaded_data
        assert "taylor_inflation_response" in loaded_data["USA"]
    
    def test_parameter_export_csv(self, temp_output_dir, sample_parameters):
        """Test parameter export to CSV."""
        exporter = CalibrationExporter(temp_output_dir)
        
        created_files = exporter.export_parameters(sample_parameters, format="csv")
        
        assert len(created_files) == 1
        csv_file = created_files[0]
        assert csv_file.endswith(".csv")
        
        # Verify CSV content
        df = pd.read_csv(csv_file)
        assert "country" in df.columns
        assert "parameter" in df.columns
        assert "value" in df.columns
        assert len(df) == 6  # 2 countries × 3 parameters each
    
    @pytest.mark.parametrize("format_type", ["json", "csv", "all"])
    def test_parameter_export_formats(self, temp_output_dir, sample_parameters, format_type):
        """Test parameter export in different formats."""
        exporter = CalibrationExporter(temp_output_dir)
        
        created_files = exporter.export_parameters(sample_parameters, format=format_type)
        
        assert len(created_files) > 0
        
        if format_type == "all":
            # Should create multiple files
            assert len(created_files) >= 3  # At least JSON, CSV, and potentially others
        else:
            assert len(created_files) == 1


@pytest.mark.integration
class TestCalibrationIntegration:
    """Integration tests for calibration components."""
    
    @pytest.fixture
    def complete_economic_data(self):
        """Create complete economic dataset for integration testing."""
        countries = ["USA", "GBR", "DEU"]
        dates = pd.date_range("2010", "2020", freq="Q")
        
        data_list = []
        for country in countries:
            for date in dates:
                data_list.append({
                    "REF_AREA": country,
                    "TIME_PERIOD": date.strftime("%Y-Q%q"),
                    "gdp": 1000 + np.random.normal(0, 50),
                    "consumption": 700 + np.random.normal(0, 30),
                    "interest_rate": 2.0 + np.random.normal(0, 0.5),
                    "inflation_rate": 2.0 + np.random.normal(0, 0.3),
                    "unemployment_rate": 5.0 + np.random.normal(0, 0.5)
                })
        
        return pd.DataFrame(data_list)
    
    def test_full_calibration_pipeline(self, complete_economic_data):
        """Test complete calibration pipeline integration."""
        # Initialize calibrators
        parameter_estimator = ABMParameterEstimator()
        initial_conditions_setter = InitialConditionsSetter()
        validator = ModelValidator()
        
        # Set up calibration manager
        manager = CalibrationManager()
        manager.register_calibrator("parameters", parameter_estimator)
        manager.register_calibrator("initial_conditions", initial_conditions_setter)
        manager.register_calibrator("validation", validator)
        
        # Run calibration sequence
        countries = ["USA", "GBR", "DEU"]
        parameters = {
            "parameters": {
                "countries": countries,
                "parameter_groups": ["taylor_rule", "firm_behavior", "household_behavior"]
            },
            "initial_conditions": {
                "countries": countries
            },
            "validation": {
                "validation_level": "standard"
            }
        }
        
        data = {"economic_data": complete_economic_data}
        results = manager.run_calibration_sequence(
            data, 
            ["parameters", "initial_conditions", "validation"],
            parameters
        )
        
        # Verify results
        assert len(results) == 3
        assert "parameters" in results
        assert "initial_conditions" in results
        assert "validation" in results
        
        # Check parameter estimation results
        param_result = results["parameters"]
        assert param_result.is_success
        assert "estimated_parameters" in param_result.data
        
        for country in countries:
            assert country in param_result.data["estimated_parameters"]
        
        # Check initial conditions results
        ic_result = results["initial_conditions"]
        assert ic_result.is_success
        assert "initial_conditions" in ic_result.data
        
        # Check validation results
        validation_result = results["validation"]
        assert validation_result.status in [CalibrationStatus.COMPLETED, CalibrationStatus.VALIDATION_FAILED]
    
    def test_calibration_export_integration(self, complete_economic_data, tmp_path):
        """Test calibration export integration."""
        # Run parameter estimation
        estimator = ABMParameterEstimator()
        result = estimator.calibrate(
            complete_economic_data,
            {"countries": ["USA", "GBR"], "parameter_groups": ["taylor_rule"]}
        )
        
        assert result.is_success
        
        # Export results
        exporter = CalibrationExporter(tmp_path)
        created_files = exporter.export_parameters(
            result.data["estimated_parameters"], 
            format="all"
        )
        
        assert len(created_files) >= 2  # Should create multiple format files
        
        # Verify files exist
        for file_path in created_files:
            assert Path(file_path).exists()
    
    def test_end_to_end_workflow(self, complete_economic_data, tmp_path):
        """Test complete end-to-end calibration workflow."""
        # This test would normally require actual data files
        pytest.skip("End-to-end testing requires full data setup")