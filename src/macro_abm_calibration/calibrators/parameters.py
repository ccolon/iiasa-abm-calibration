"""
ABM parameter estimation.

This module implements parameter estimation for agent-based macroeconomic models,
including monetary policy rules (Taylor rule), firm behavioral parameters,
household consumption parameters, and market dynamics parameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import EconomicCalibrator, CalibrationResult, CalibrationStatus, CalibrationMetadata
from ..models import Country, OECD_COUNTRIES


@dataclass
class EstimationResults:
    """Results from parameter estimation."""
    parameters: Dict[str, float]
    standard_errors: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    goodness_of_fit: Dict[str, float] = field(default_factory=dict)
    residuals: Optional[pd.Series] = None
    fitted_values: Optional[pd.Series] = None


class ABMParameterEstimator(EconomicCalibrator):
    """
    Agent-Based Model parameter estimator.
    
    This class estimates key parameters for ABM models including:
    - Monetary policy rules (Taylor rule coefficients)
    - Firm behavioral parameters (pricing, investment decisions)
    - Household consumption parameters (marginal propensities)
    - Market dynamics parameters (adjustment speeds)
    """
    
    # Parameter bounds for optimization
    PARAMETER_BOUNDS = {
        # Taylor rule parameters
        "taylor_inflation_response": (1.01, 3.0),      # Greater than 1 for stability
        "taylor_output_response": (0.0, 2.0),
        "taylor_smoothing": (0.0, 0.99),
        
        # Firm parameters
        "price_adjustment_speed": (0.01, 1.0),
        "investment_sensitivity": (0.1, 2.0),
        "markup_elasticity": (0.1, 5.0),
        
        # Household parameters
        "marginal_propensity_consume": (0.3, 0.95),
        "wealth_effect": (0.01, 0.1),
        "labor_supply_elasticity": (0.1, 2.0),
        
        # Market dynamics
        "wage_adjustment_speed": (0.01, 0.5),
        "employment_adjustment_speed": (0.05, 0.8)
    }
    
    def __init__(self, name: Optional[str] = None):
        """Initialize ABM parameter estimator."""
        super().__init__(name or "ABMParameterEstimator")
        self.estimation_results: Dict[str, EstimationResults] = {}
    
    def calibrate(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Estimate ABM parameters from economic data.
        
        Args:
            data: Economic time series data
            parameters: Estimation parameters including:
                - countries: List of countries to estimate
                - estimation_method: Method for parameter estimation
                - parameter_groups: Groups of parameters to estimate
                
        Returns:
            CalibrationResult with estimated parameters
        """
        operation_id = self._generate_operation_id()
        
        # Validate inputs
        errors = self.validate_inputs(data, parameters)
        if errors:
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            return CalibrationResult(
                status=CalibrationStatus.FAILED,
                data={},
                metadata=metadata,
                errors=errors
            )
        
        # Extract parameters
        countries = parameters.get("countries", []) if parameters else []
        estimation_method = parameters.get("estimation_method", "ols") if parameters else "ols"
        parameter_groups = parameters.get("parameter_groups", ["taylor_rule", "firm_behavior", "household_behavior"]) if parameters else ["taylor_rule", "firm_behavior", "household_behavior"]
        
        self.logger.info(f"Estimating ABM parameters for {len(countries)} countries")
        
        try:
            estimated_parameters = {}
            
            # Process each country
            for country in countries:
                country_data = self._extract_country_data(data, country)
                
                if country_data.empty:
                    self.logger.warning(f"No data available for country {country}")
                    continue
                
                country_params = {}
                
                # Estimate parameter groups
                for group in parameter_groups:
                    if group == "taylor_rule":
                        taylor_params = self._estimate_taylor_rule(country_data, country)
                        country_params.update(taylor_params)
                    
                    elif group == "firm_behavior":
                        firm_params = self._estimate_firm_parameters(country_data, country)
                        country_params.update(firm_params)
                    
                    elif group == "household_behavior":
                        household_params = self._estimate_household_parameters(country_data, country)
                        country_params.update(household_params)
                
                estimated_parameters[country] = country_params
                self.logger.info(f"Estimated {len(country_params)} parameters for {country}")
            
            # Create metadata
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            # Create result
            result = CalibrationResult(
                status=CalibrationStatus.COMPLETED,
                data={"estimated_parameters": estimated_parameters},
                metadata=metadata
            )
            
            self.logger.info(f"Parameter estimation completed for {len(estimated_parameters)} countries")
            return result
            
        except Exception as e:
            self.logger.error(f"Parameter estimation failed: {e}")
            
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
    
    def _extract_country_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        country: str
    ) -> pd.DataFrame:
        """Extract data for a specific country."""
        if isinstance(data, pd.DataFrame):
            if "REF_AREA" in data.columns:
                return data[data["REF_AREA"] == country].copy()
            else:
                return data.copy()
        
        elif isinstance(data, dict):
            if country in data:
                return data[country].copy()
            else:
                # Try to find country data in any of the DataFrames
                for df in data.values():
                    if isinstance(df, pd.DataFrame) and "REF_AREA" in df.columns:
                        country_data = df[df["REF_AREA"] == country]
                        if not country_data.empty:
                            return country_data.copy()
        
        return pd.DataFrame()
    
    def _estimate_taylor_rule(self, data: pd.DataFrame, country: str) -> Dict[str, float]:
        """
        Estimate Taylor rule parameters.
        
        The Taylor rule: i_t = r* + π_t + α(π_t - π*) + β(y_t - y*)
        Where:
        - i_t = nominal interest rate
        - r* = neutral real interest rate
        - π_t = inflation rate
        - π* = inflation target
        - y_t - y* = output gap
        - α = response to inflation gap
        - β = response to output gap
        """
        self.logger.info(f"Estimating Taylor rule for {country}")
        
        # Prepare data
        taylor_data = self._prepare_taylor_rule_data(data)
        
        if taylor_data.empty or len(taylor_data) < 10:
            self.logger.warning(f"Insufficient data for Taylor rule estimation in {country}")
            return self._get_default_taylor_parameters()
        
        try:
            # Estimate using OLS
            y = taylor_data["interest_rate"].values
            X = taylor_data[["inflation_gap", "output_gap", "lagged_interest_rate"]].values
            
            # Remove NaN values
            valid_idx = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            y_clean = y[valid_idx]
            X_clean = X[valid_idx]
            
            if len(y_clean) < 5:
                self.logger.warning(f"Too few valid observations for Taylor rule in {country}")
                return self._get_default_taylor_parameters()
            
            # Fit regression
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            
            # Extract parameters
            inflation_response = model.coef_[0] + 1.0  # Add 1 for total response
            output_response = model.coef_[1]
            smoothing = model.coef_[2]
            neutral_rate = model.intercept_
            
            # Apply bounds
            inflation_response = np.clip(
                inflation_response,
                *self.PARAMETER_BOUNDS["taylor_inflation_response"]
            )
            output_response = np.clip(
                output_response,
                *self.PARAMETER_BOUNDS["taylor_output_response"]
            )
            smoothing = np.clip(
                smoothing,
                *self.PARAMETER_BOUNDS["taylor_smoothing"]
            )
            
            # Calculate goodness of fit
            y_pred = model.predict(X_clean)
            r_squared = model.score(X_clean, y_clean)
            
            self.logger.info(f"Taylor rule R² for {country}: {r_squared:.3f}")
            
            return {
                "taylor_inflation_response": inflation_response,
                "taylor_output_response": output_response,
                "taylor_smoothing": smoothing,
                "taylor_neutral_rate": neutral_rate,
                "taylor_r_squared": r_squared
            }
            
        except Exception as e:
            self.logger.warning(f"Taylor rule estimation failed for {country}: {e}")
            return self._get_default_taylor_parameters()
    
    def _prepare_taylor_rule_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Taylor rule estimation."""
        # Required columns for Taylor rule
        required_cols = ["interest_rate", "inflation_rate", "gdp"]
        
        # Check if we have the required data
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        # Create working DataFrame
        taylor_data = pd.DataFrame()
        
        # Interest rate (dependent variable)
        if "interest_rate" in data.columns:
            taylor_data["interest_rate"] = data["interest_rate"]
        elif "short_term_rate" in data.columns:
            taylor_data["interest_rate"] = data["short_term_rate"]
        else:
            # Use a proxy if available
            return pd.DataFrame()
        
        # Inflation rate
        if "inflation_rate" in data.columns:
            taylor_data["inflation"] = data["inflation_rate"]
        elif "gdp_deflator" in data.columns:
            taylor_data["inflation"] = data["gdp_deflator"].pct_change() * 100
        else:
            return pd.DataFrame()
        
        # GDP for output gap calculation
        if "gdp" in data.columns:
            # Calculate output gap using HP filter approximation
            gdp_log = np.log(data["gdp"])
            taylor_data["output_gap"] = self._calculate_output_gap(gdp_log)
        else:
            taylor_data["output_gap"] = 0  # Assume no output gap
        
        # Calculate inflation gap (assume 2% target)
        inflation_target = 2.0
        taylor_data["inflation_gap"] = taylor_data["inflation"] - inflation_target
        
        # Lagged interest rate for smoothing
        taylor_data["lagged_interest_rate"] = taylor_data["interest_rate"].shift(1)
        
        # Remove rows with NaN values
        taylor_data = taylor_data.dropna()
        
        return taylor_data
    
    def _calculate_output_gap(self, log_gdp: pd.Series, lambda_param: float = 1600) -> pd.Series:
        """
        Calculate output gap using simplified HP filter.
        
        This is a simplified version of the Hodrick-Prescott filter.
        """
        if len(log_gdp) < 4:
            return pd.Series(0, index=log_gdp.index)
        
        # Simple linear trend as proxy for HP filter
        x = np.arange(len(log_gdp))
        coeffs = np.polyfit(x, log_gdp.values, 1)
        trend = np.polyval(coeffs, x)
        
        output_gap = (log_gdp.values - trend) * 100  # Convert to percentage
        
        return pd.Series(output_gap, index=log_gdp.index)
    
    def _get_default_taylor_parameters(self) -> Dict[str, float]:
        """Get default Taylor rule parameters."""
        return {
            "taylor_inflation_response": 1.5,
            "taylor_output_response": 0.5,
            "taylor_smoothing": 0.8,
            "taylor_neutral_rate": 2.0,
            "taylor_r_squared": 0.0
        }
    
    def _estimate_firm_parameters(self, data: pd.DataFrame, country: str) -> Dict[str, float]:
        """Estimate firm behavioral parameters."""
        self.logger.info(f"Estimating firm parameters for {country}")
        
        firm_params = {}
        
        # Price adjustment speed (from inflation persistence)
        if "inflation_rate" in data.columns:
            inflation = data["inflation_rate"].dropna()
            if len(inflation) > 5:
                # Estimate AR(1) coefficient for inflation persistence
                inflation_lagged = inflation.shift(1).dropna()
                inflation_current = inflation[1:].reset_index(drop=True)
                inflation_lagged = inflation_lagged.reset_index(drop=True)
                
                if len(inflation_current) == len(inflation_lagged) and len(inflation_current) > 0:
                    correlation = np.corrcoef(inflation_current, inflation_lagged)[0, 1]
                    # Convert persistence to adjustment speed
                    price_adjustment = 1 - abs(correlation)
                    firm_params["price_adjustment_speed"] = np.clip(
                        price_adjustment,
                        *self.PARAMETER_BOUNDS["price_adjustment_speed"]
                    )
        
        # Investment sensitivity (from investment-GDP volatility)
        if "investment" in data.columns and "gdp" in data.columns:
            investment_ratio = data["investment"] / data["gdp"]
            if not investment_ratio.empty:
                investment_volatility = investment_ratio.std()
                # Scale to reasonable bounds
                investment_sensitivity = min(max(investment_volatility * 10, 0.1), 2.0)
                firm_params["investment_sensitivity"] = investment_sensitivity
        
        # Markup elasticity (from price-cost relationship proxy)
        if "gdp_deflator" in data.columns:
            deflator = data["gdp_deflator"].dropna()
            if len(deflator) > 5:
                deflator_volatility = deflator.pct_change().std()
                markup_elasticity = min(max(deflator_volatility * 20, 0.1), 5.0)
                firm_params["markup_elasticity"] = markup_elasticity
        
        # Apply default values for missing parameters
        default_firm_params = {
            "price_adjustment_speed": 0.3,
            "investment_sensitivity": 1.0,
            "markup_elasticity": 2.0
        }
        
        for param, default_val in default_firm_params.items():
            if param not in firm_params:
                firm_params[param] = default_val
        
        return firm_params
    
    def _estimate_household_parameters(self, data: pd.DataFrame, country: str) -> Dict[str, float]:
        """Estimate household behavioral parameters."""
        self.logger.info(f"Estimating household parameters for {country}")
        
        household_params = {}
        
        # Marginal propensity to consume (from consumption-GDP relationship)
        if "consumption" in data.columns and "gdp" in data.columns:
            consumption = data["consumption"].dropna()
            gdp = data["gdp"].dropna()
            
            # Align series
            common_index = consumption.index.intersection(gdp.index)
            if len(common_index) > 5:
                cons_aligned = consumption.loc[common_index]
                gdp_aligned = gdp.loc[common_index]
                
                # Calculate changes
                cons_change = cons_aligned.diff().dropna()
                gdp_change = gdp_aligned.diff().dropna()
                
                if len(cons_change) > 3 and len(gdp_change) > 3:
                    # Regression of consumption change on GDP change
                    common_idx = cons_change.index.intersection(gdp_change.index)
                    if len(common_idx) > 3:
                        X = gdp_change.loc[common_idx].values.reshape(-1, 1)
                        y = cons_change.loc[common_idx].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        mpc = model.coef_[0]
                        
                        # Apply bounds
                        mpc = np.clip(mpc, *self.PARAMETER_BOUNDS["marginal_propensity_consume"])
                        household_params["marginal_propensity_consume"] = mpc
        
        # Wealth effect (proxy from consumption volatility)
        if "consumption" in data.columns:
            consumption = data["consumption"].dropna()
            if len(consumption) > 5:
                cons_volatility = consumption.pct_change().std()
                wealth_effect = min(max(cons_volatility, 0.01), 0.1)
                household_params["wealth_effect"] = wealth_effect
        
        # Labor supply elasticity (proxy from unemployment volatility)
        if "unemployment_rate" in data.columns:
            unemployment = data["unemployment_rate"].dropna()
            if len(unemployment) > 5:
                unemp_volatility = unemployment.std()
                # Inverse relationship: higher volatility = lower elasticity
                labor_elasticity = min(max(1 / (unemp_volatility + 0.1), 0.1), 2.0)
                household_params["labor_supply_elasticity"] = labor_elasticity
        
        # Apply default values for missing parameters
        default_household_params = {
            "marginal_propensity_consume": 0.7,
            "wealth_effect": 0.05,
            "labor_supply_elasticity": 0.5
        }
        
        for param, default_val in default_household_params.items():
            if param not in household_params:
                household_params[param] = default_val
        
        return household_params
    
    def estimate_single_equation(
        self,
        dependent_var: pd.Series,
        independent_vars: pd.DataFrame,
        method: str = "ols"
    ) -> EstimationResults:
        """
        Estimate a single equation using specified method.
        
        Args:
            dependent_var: Dependent variable
            independent_vars: Independent variables
            method: Estimation method ('ols', 'ridge', 'robust')
            
        Returns:
            EstimationResults with parameter estimates
        """
        # Align data
        common_index = dependent_var.index.intersection(independent_vars.index)
        y = dependent_var.loc[common_index].values
        X = independent_vars.loc[common_index].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        y_clean = y[valid_idx]
        X_clean = X[valid_idx]
        
        if len(y_clean) < len(independent_vars.columns) + 1:
            raise ValueError("Insufficient observations for estimation")
        
        # Estimate parameters
        if method == "ols":
            model = LinearRegression()
        elif method == "ridge":
            model = Ridge(alpha=0.1)
        else:
            model = LinearRegression()  # Default to OLS
        
        model.fit(X_clean, y_clean)
        
        # Calculate standard errors (simplified)
        y_pred = model.predict(X_clean)
        residuals = y_clean - y_pred
        mse = np.mean(residuals**2)
        
        # Approximate standard errors
        if hasattr(model, 'coef_'):
            n_params = len(model.coef_)
            se_approx = np.sqrt(mse / len(y_clean)) * np.ones(n_params)
            
            parameters = dict(zip(independent_vars.columns, model.coef_))
            standard_errors = dict(zip(independent_vars.columns, se_approx))
            
            # 95% confidence intervals
            confidence_intervals = {}
            for i, param_name in enumerate(independent_vars.columns):
                coef = model.coef_[i]
                se = se_approx[i]
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
                confidence_intervals[param_name] = (ci_lower, ci_upper)
        else:
            parameters = {}
            standard_errors = {}
            confidence_intervals = {}
        
        # Goodness of fit
        r_squared = model.score(X_clean, y_clean) if hasattr(model, 'score') else 0.0
        
        return EstimationResults(
            parameters=parameters,
            standard_errors=standard_errors,
            confidence_intervals=confidence_intervals,
            goodness_of_fit={"r_squared": r_squared, "mse": mse},
            residuals=pd.Series(residuals, index=common_index[valid_idx]),
            fitted_values=pd.Series(y_pred, index=common_index[valid_idx])
        )
    
    def get_parameter_summary(self, country: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary of estimated parameters.
        
        Args:
            country: Specific country (if None, all countries)
            
        Returns:
            DataFrame with parameter summary
        """
        if not hasattr(self, 'estimation_results') or not self.estimation_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for country_code, results in self.estimation_results.items():
            if country and country_code != country:
                continue
            
            for param_name, param_value in results.parameters.items():
                row = {
                    "country": country_code,
                    "parameter": param_name,
                    "value": param_value
                }
                
                # Add standard error if available
                if param_name in results.standard_errors:
                    row["standard_error"] = results.standard_errors[param_name]
                
                # Add confidence interval if available
                if param_name in results.confidence_intervals:
                    ci_lower, ci_upper = results.confidence_intervals[param_name]
                    row["ci_lower"] = ci_lower
                    row["ci_upper"] = ci_upper
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def validate_estimated_parameters(
        self,
        estimated_params: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """
        Validate estimated parameters against economic theory.
        
        Args:
            estimated_params: Dictionary of estimated parameters by country
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        for country, params in estimated_params.items():
            # Taylor rule validation
            if "taylor_inflation_response" in params:
                if params["taylor_inflation_response"] < 1.0:
                    warnings.append(
                        f"{country}: Taylor rule inflation response < 1.0 may lead to instability"
                    )
            
            if "taylor_output_response" in params:
                if params["taylor_output_response"] < 0:
                    warnings.append(
                        f"{country}: Negative Taylor rule output response is unusual"
                    )
            
            # Consumption parameters
            if "marginal_propensity_consume" in params:
                mpc = params["marginal_propensity_consume"]
                if mpc < 0.3 or mpc > 0.95:
                    warnings.append(
                        f"{country}: MPC of {mpc:.2f} is outside typical range [0.3, 0.95]"
                    )
        
        return warnings