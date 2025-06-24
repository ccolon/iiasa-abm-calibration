"""
Model validation framework.

This module implements validation of ABM calibration results against the original
MATLAB outputs and economic theory, ensuring consistency and accuracy of the
calibrated model parameters and initial conditions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats, io as scipy_io
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base import EconomicCalibrator, CalibrationResult, CalibrationStatus, CalibrationMetadata
from ..utils.validation import ValidationResult


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationCheck:
    """Individual validation check result."""
    name: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of all validation checks."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    critical_issues: int
    overall_passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)


class ModelValidator(EconomicCalibrator):
    """
    Model validation framework.
    
    This class validates ABM calibration results by:
    - Comparing against MATLAB reference outputs
    - Checking economic theory consistency
    - Validating parameter ranges and relationships
    - Ensuring model stability and convergence properties
    """
    
    # Economic theory bounds for validation
    THEORY_BOUNDS = {
        "taylor_inflation_response": (1.0, 5.0),
        "taylor_output_response": (0.0, 2.0),
        "marginal_propensity_consume": (0.3, 0.95),
        "unemployment_rate": (0.01, 0.25),
        "inflation_rate": (-0.05, 0.15),
        "interest_rate": (0.0, 0.20),
        "debt_to_gdp_ratio": (0.0, 2.0),
        "capacity_utilization": (0.5, 1.0)
    }
    
    # Statistical thresholds for comparisons
    COMPARISON_THRESHOLDS = {
        "correlation_threshold": 0.7,
        "relative_error_threshold": 0.1,  # 10%
        "statistical_significance": 0.05
    }
    
    def __init__(self, name: Optional[str] = None):
        """Initialize model validator."""
        super().__init__(name or "ModelValidator")
        self.validation_history: List[ValidationSummary] = []
        self.reference_data: Dict[str, Any] = {}
    
    def calibrate(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Validate calibration results.
        
        Args:
            data: Calibration results to validate
            parameters: Validation parameters including:
                - reference_data_path: Path to MATLAB reference data
                - validation_level: Strictness of validation
                - output_path: Path to save validation report
                
        Returns:
            CalibrationResult with validation summary
        """
        operation_id = self._generate_operation_id()
        
        # Extract parameters
        reference_path = parameters.get("reference_data_path") if parameters else None
        validation_level = parameters.get("validation_level", "standard") if parameters else "standard"
        output_path = parameters.get("output_path") if parameters else None
        
        self.logger.info("Starting model validation")
        
        try:
            # Load reference data if provided
            if reference_path:
                self.reference_data = self._load_reference_data(reference_path)
            
            # Perform validation checks
            validation_summary = self._perform_validation_suite(data, validation_level)
            
            # Generate validation report
            if output_path:
                self._generate_validation_report(validation_summary, output_path)
            
            # Store validation history
            self.validation_history.append(validation_summary)
            
            # Determine overall status
            if validation_summary.critical_issues > 0:
                status = CalibrationStatus.VALIDATION_FAILED
            elif validation_summary.overall_passed:
                status = CalibrationStatus.COMPLETED
            else:
                status = CalibrationStatus.VALIDATION_FAILED
            
            # Create metadata
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {},
                validation_results={
                    "total_checks": validation_summary.total_checks,
                    "passed_checks": validation_summary.passed_checks,
                    "overall_passed": validation_summary.overall_passed
                }
            )
            
            result = CalibrationResult(
                status=status,
                data={"validation_summary": validation_summary},
                metadata=metadata
            )
            
            self.logger.info(f"Validation completed: {validation_summary.passed_checks}/{validation_summary.total_checks} checks passed")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            return CalibrationResult(
                status=CalibrationStatus.FAILED,
                data={},
                metadata=metadata,
                errors=[str(e)]
            )
    
    def _load_reference_data(self, reference_path: Union[str, Path]) -> Dict[str, Any]:
        """Load MATLAB reference data for comparison."""
        reference_path = Path(reference_path)
        
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference data not found: {reference_path}")
        
        if reference_path.suffix == ".mat":
            # Load MATLAB file
            return scipy_io.loadmat(str(reference_path))
        
        elif reference_path.suffix == ".json":
            # Load JSON file
            import json
            with open(reference_path, 'r') as f:
                return json.load(f)
        
        elif reference_path.suffix == ".pkl":
            # Load pickle file
            import pickle
            with open(reference_path, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported reference data format: {reference_path.suffix}")
    
    def _perform_validation_suite(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        validation_level: str
    ) -> ValidationSummary:
        """Perform comprehensive validation suite."""
        checks = []
        
        # Economic theory validation
        theory_checks = self._validate_economic_theory(data)
        checks.extend(theory_checks)
        
        # Parameter range validation
        range_checks = self._validate_parameter_ranges(data)
        checks.extend(range_checks)
        
        # Consistency validation
        consistency_checks = self._validate_internal_consistency(data)
        checks.extend(consistency_checks)
        
        # Reference data comparison (if available)
        if self.reference_data:
            comparison_checks = self._validate_against_reference(data)
            checks.extend(comparison_checks)
        
        # Stability validation
        stability_checks = self._validate_model_stability(data)
        checks.extend(stability_checks)
        
        # Statistical validation
        if validation_level in ["strict", "comprehensive"]:
            statistical_checks = self._validate_statistical_properties(data)
            checks.extend(statistical_checks)
        
        # Compile summary
        summary = self._compile_validation_summary(checks)
        
        return summary
    
    def _validate_economic_theory(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate against economic theory."""
        checks = []
        
        # Extract parameter data
        if isinstance(data, dict) and "estimated_parameters" in data:
            params_data = data["estimated_parameters"]
            
            for country, params in params_data.items():
                # Taylor rule principle
                if "taylor_inflation_response" in params:
                    inflation_response = params["taylor_inflation_response"]
                    
                    check = ValidationCheck(
                        name=f"taylor_principle_{country}",
                        level=ValidationLevel.ERROR,
                        passed=inflation_response > 1.0,
                        message=f"Taylor principle: inflation response ({inflation_response:.2f}) should be > 1.0",
                        details={"country": country, "value": inflation_response}
                    )
                    checks.append(check)
                
                # Consumption theory
                if "marginal_propensity_consume" in params:
                    mpc = params["marginal_propensity_consume"]
                    
                    check = ValidationCheck(
                        name=f"consumption_theory_{country}",
                        level=ValidationLevel.WARNING,
                        passed=0.3 <= mpc <= 0.95,
                        message=f"MPC ({mpc:.2f}) should be between 0.3 and 0.95",
                        details={"country": country, "value": mpc}
                    )
                    checks.append(check)
        
        # Validate initial conditions economic logic
        if isinstance(data, dict) and "initial_conditions" in data:
            ic_data = data["initial_conditions"]
            
            for country, conditions in ic_data.items():
                # Unemployment rate
                if "market_conditions" in conditions:
                    market = conditions["market_conditions"]
                    if hasattr(market, 'unemployment_rate'):
                        unemp_rate = market.unemployment_rate
                        
                        check = ValidationCheck(
                            name=f"unemployment_bound_{country}",
                            level=ValidationLevel.WARNING,
                            passed=0.01 <= unemp_rate <= 0.25,
                            message=f"Unemployment rate ({unemp_rate:.2%}) outside typical range [1%, 25%]",
                            details={"country": country, "value": unemp_rate}
                        )
                        checks.append(check)
        
        return checks
    
    def _validate_parameter_ranges(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate parameter values are within reasonable ranges."""
        checks = []
        
        if isinstance(data, dict) and "estimated_parameters" in data:
            params_data = data["estimated_parameters"]
            
            for country, params in params_data.items():
                for param_name, param_value in params.items():
                    if param_name in self.THEORY_BOUNDS:
                        min_val, max_val = self.THEORY_BOUNDS[param_name]
                        
                        check = ValidationCheck(
                            name=f"range_{param_name}_{country}",
                            level=ValidationLevel.ERROR,
                            passed=min_val <= param_value <= max_val,
                            message=f"{param_name} ({param_value:.3f}) outside bounds [{min_val}, {max_val}]",
                            details={
                                "country": country,
                                "parameter": param_name,
                                "value": param_value,
                                "bounds": (min_val, max_val)
                            }
                        )
                        checks.append(check)
        
        return checks
    
    def _validate_internal_consistency(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate internal consistency of parameters and conditions."""
        checks = []
        
        if isinstance(data, dict):
            # Check consistency between parameters and initial conditions
            if "estimated_parameters" in data and "initial_conditions" in data:
                params_data = data["estimated_parameters"]
                ic_data = data["initial_conditions"]
                
                common_countries = set(params_data.keys()) & set(ic_data.keys())
                
                for country in common_countries:
                    params = params_data[country]
                    conditions = ic_data[country]
                    
                    # Check Taylor rule consistency with initial interest rate
                    if ("taylor_neutral_rate" in params and 
                        "market_conditions" in conditions and
                        hasattr(conditions["market_conditions"], 'interest_rate')):
                        
                        neutral_rate = params["taylor_neutral_rate"]
                        initial_rate = conditions["market_conditions"].interest_rate
                        
                        rate_diff = abs(neutral_rate - initial_rate)
                        
                        check = ValidationCheck(
                            name=f"rate_consistency_{country}",
                            level=ValidationLevel.WARNING,
                            passed=rate_diff < 0.02,  # 2 percentage points
                            message=f"Taylor neutral rate ({neutral_rate:.2%}) differs from initial rate ({initial_rate:.2%})",
                            details={
                                "country": country,
                                "neutral_rate": neutral_rate,
                                "initial_rate": initial_rate,
                                "difference": rate_diff
                            }
                        )
                        checks.append(check)
        
        return checks
    
    def _validate_against_reference(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate against MATLAB reference data."""
        checks = []
        
        if not self.reference_data:
            return checks
        
        # Compare estimated parameters
        if isinstance(data, dict) and "estimated_parameters" in data:
            python_params = data["estimated_parameters"]
            
            for country in python_params.keys():
                # Look for corresponding country in reference data
                ref_country_key = self._find_reference_country_key(country)
                
                if ref_country_key and ref_country_key in self.reference_data:
                    ref_params = self.reference_data[ref_country_key]
                    
                    # Compare specific parameters
                    param_comparisons = self._compare_parameters(
                        python_params[country], ref_params, country
                    )
                    checks.extend(param_comparisons)
        
        return checks
    
    def _compare_parameters(
        self,
        python_params: Dict[str, float],
        ref_params: Dict[str, Any],
        country: str
    ) -> List[ValidationCheck]:
        """Compare Python parameters with MATLAB reference."""
        checks = []
        
        for param_name, python_value in python_params.items():
            # Find corresponding parameter in reference data
            ref_key = self._map_parameter_name(param_name)
            
            if ref_key in ref_params:
                ref_value = float(ref_params[ref_key])
                
                # Calculate relative error
                rel_error = abs(python_value - ref_value) / abs(ref_value) if ref_value != 0 else abs(python_value)
                
                threshold = self.COMPARISON_THRESHOLDS["relative_error_threshold"]
                
                check = ValidationCheck(
                    name=f"matlab_comparison_{param_name}_{country}",
                    level=ValidationLevel.WARNING,
                    passed=rel_error <= threshold,
                    message=f"{param_name}: Python={python_value:.3f}, MATLAB={ref_value:.3f}, error={rel_error:.1%}",
                    details={
                        "country": country,
                        "parameter": param_name,
                        "python_value": python_value,
                        "matlab_value": ref_value,
                        "relative_error": rel_error
                    }
                )
                checks.append(check)
        
        return checks
    
    def _validate_model_stability(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate model stability properties."""
        checks = []
        
        if isinstance(data, dict) and "estimated_parameters" in data:
            params_data = data["estimated_parameters"]
            
            for country, params in params_data.items():
                # Check stability conditions for Taylor rule
                if ("taylor_inflation_response" in params and 
                    "taylor_output_response" in params and
                    "taylor_smoothing" in params):
                    
                    phi_pi = params["taylor_inflation_response"]
                    phi_y = params["taylor_output_response"]
                    rho = params["taylor_smoothing"]
                    
                    # Simplified stability condition
                    stability_condition = phi_pi > 1.0 and phi_y >= 0 and 0 <= rho < 1
                    
                    check = ValidationCheck(
                        name=f"taylor_stability_{country}",
                        level=ValidationLevel.ERROR,
                        passed=stability_condition,
                        message=f"Taylor rule stability condition violated",
                        details={
                            "country": country,
                            "phi_pi": phi_pi,
                            "phi_y": phi_y,
                            "rho": rho
                        }
                    )
                    checks.append(check)
        
        return checks
    
    def _validate_statistical_properties(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> List[ValidationCheck]:
        """Validate statistical properties of estimates."""
        checks = []
        
        if isinstance(data, dict) and "estimated_parameters" in data:
            params_data = data["estimated_parameters"]
            
            # Collect parameter values across countries
            param_values = {}
            for country, params in params_data.items():
                for param_name, param_value in params.items():
                    if param_name not in param_values:
                        param_values[param_name] = []
                    param_values[param_name].append(param_value)
            
            # Check for outliers in cross-country distribution
            for param_name, values in param_values.items():
                if len(values) > 3:  # Need at least 4 countries
                    values_array = np.array(values)
                    
                    # Z-score test for outliers
                    z_scores = np.abs(stats.zscore(values_array))
                    outliers = z_scores > 3.0
                    
                    if outliers.any():
                        outlier_count = outliers.sum()
                        
                        check = ValidationCheck(
                            name=f"outlier_test_{param_name}",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"{outlier_count} outlier(s) detected in {param_name} distribution",
                            details={
                                "parameter": param_name,
                                "outlier_count": outlier_count,
                                "values": values,
                                "z_scores": z_scores.tolist()
                            }
                        )
                        checks.append(check)
        
        return checks
    
    def _find_reference_country_key(self, country: str) -> Optional[str]:
        """Find corresponding country key in reference data."""
        # Direct match
        if country in self.reference_data:
            return country
        
        # Try variations
        variations = [
            country.lower(),
            country.upper(),
            f"country_{country}",
            f"{country}_data"
        ]
        
        for var in variations:
            if var in self.reference_data:
                return var
        
        return None
    
    def _map_parameter_name(self, param_name: str) -> str:
        """Map Python parameter name to MATLAB equivalent."""
        # Parameter name mapping
        mapping = {
            "taylor_inflation_response": "phi_pi",
            "taylor_output_response": "phi_y", 
            "taylor_smoothing": "rho_i",
            "marginal_propensity_consume": "mpc",
            "price_adjustment_speed": "kappa"
        }
        
        return mapping.get(param_name, param_name)
    
    def _compile_validation_summary(self, checks: List[ValidationCheck]) -> ValidationSummary:
        """Compile validation summary from individual checks."""
        total_checks = len(checks)
        passed_checks = sum(1 for check in checks if check.passed)
        failed_checks = total_checks - passed_checks
        
        warnings = sum(1 for check in checks if check.level == ValidationLevel.WARNING)
        errors = sum(1 for check in checks if check.level == ValidationLevel.ERROR and not check.passed)
        critical_issues = sum(1 for check in checks if check.level == ValidationLevel.CRITICAL and not check.passed)
        
        # Overall pass: no critical issues and <20% failed checks
        overall_passed = (critical_issues == 0 and 
                         errors == 0 and
                         (failed_checks / total_checks if total_checks > 0 else 0) < 0.2)
        
        return ValidationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
            critical_issues=critical_issues,
            overall_passed=overall_passed,
            checks=checks
        )
    
    def _generate_validation_report(
        self,
        summary: ValidationSummary,
        output_path: Union[str, Path]
    ) -> None:
        """Generate detailed validation report."""
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            f.write("# Model Validation Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Checks**: {summary.total_checks}\n")
            f.write(f"- **Passed**: {summary.passed_checks}\n")
            f.write(f"- **Failed**: {summary.failed_checks}\n")
            f.write(f"- **Warnings**: {summary.warnings}\n")
            f.write(f"- **Errors**: {summary.errors}\n")
            f.write(f"- **Critical Issues**: {summary.critical_issues}\n")
            f.write(f"- **Overall Result**: {'✅ PASSED' if summary.overall_passed else '❌ FAILED'}\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for check in summary.checks:
                status = "✅" if check.passed else "❌"
                level_emoji = {
                    ValidationLevel.INFO: "ℹ️",
                    ValidationLevel.WARNING: "⚠️",
                    ValidationLevel.ERROR: "❌",
                    ValidationLevel.CRITICAL: "🚨"
                }
                
                f.write(f"### {status} {check.name}\n")
                f.write(f"{level_emoji[check.level]} **{check.level.value.upper()}**: {check.message}\n\n")
                
                if check.details:
                    f.write("**Details:**\n")
                    for key, value in check.details.items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
        
        self.logger.info(f"Validation report saved to {output_path}")
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation runs."""
        if not self.validation_history:
            return pd.DataFrame()
        
        summary_data = []
        for i, summary in enumerate(self.validation_history):
            summary_data.append({
                "run": i + 1,
                "total_checks": summary.total_checks,
                "passed_checks": summary.passed_checks,
                "failed_checks": summary.failed_checks,
                "warnings": summary.warnings,
                "errors": summary.errors,
                "critical_issues": summary.critical_issues,
                "overall_passed": summary.overall_passed
            })
        
        return pd.DataFrame(summary_data)
    
    def compare_with_benchmark(
        self,
        current_results: Dict[str, Any],
        benchmark_results: Dict[str, Any],
        tolerance: float = 0.1
    ) -> ValidationSummary:
        """
        Compare current results with benchmark results.
        
        Args:
            current_results: Current calibration results
            benchmark_results: Benchmark results to compare against
            tolerance: Tolerance for comparison (relative error)
            
        Returns:
            ValidationSummary with comparison results
        """
        checks = []
        
        # Compare estimated parameters
        if ("estimated_parameters" in current_results and 
            "estimated_parameters" in benchmark_results):
            
            current_params = current_results["estimated_parameters"]
            benchmark_params = benchmark_results["estimated_parameters"]
            
            common_countries = set(current_params.keys()) & set(benchmark_params.keys())
            
            for country in common_countries:
                current_country = current_params[country]
                benchmark_country = benchmark_params[country]
                
                common_params = set(current_country.keys()) & set(benchmark_country.keys())
                
                for param_name in common_params:
                    current_val = current_country[param_name]
                    benchmark_val = benchmark_country[param_name]
                    
                    rel_error = abs(current_val - benchmark_val) / abs(benchmark_val) if benchmark_val != 0 else abs(current_val)
                    
                    check = ValidationCheck(
                        name=f"benchmark_{param_name}_{country}",
                        level=ValidationLevel.WARNING,
                        passed=rel_error <= tolerance,
                        message=f"{param_name}: current={current_val:.3f}, benchmark={benchmark_val:.3f}, error={rel_error:.1%}",
                        details={
                            "country": country,
                            "parameter": param_name,
                            "current_value": current_val,
                            "benchmark_value": benchmark_val,
                            "relative_error": rel_error
                        }
                    )
                    checks.append(check)
        
        return self._compile_validation_summary(checks)