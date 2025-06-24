"""
Calibration utilities and economic functions.

This module provides utility functions for economic calculations, parameter
transformations, and calibration support functions used across the calibration
modules.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats, special
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def hp_filter(
    series: Union[pd.Series, np.ndarray],
    lamb: float = 1600,
    return_trend: bool = False
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Hodrick-Prescott filter for trend extraction.
    
    Args:
        series: Time series data
        lamb: Smoothing parameter (1600 for quarterly, 100 for annual)
        return_trend: Whether to return trend component
        
    Returns:
        Detrended series or (detrended, trend) if return_trend=True
    """
    if isinstance(series, pd.Series):
        y = series.values
        index = series.index
    else:
        y = np.asarray(series)
        index = None
    
    n = len(y)
    
    if n < 4:
        # Not enough data for HP filter
        if return_trend:
            return pd.Series(np.zeros(n), index=index), pd.Series(y, index=index)
        else:
            return pd.Series(np.zeros(n), index=index)
    
    # Create difference matrix
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    
    # HP filter calculation
    I = np.eye(n)
    
    try:
        # Solve the HP filter equation: (I + λD'D)τ = y
        A = I + lamb * D.T @ D
        trend = np.linalg.solve(A, y)
        cycle = y - trend
        
        if index is not None:
            trend_series = pd.Series(trend, index=index)
            cycle_series = pd.Series(cycle, index=index)
        else:
            trend_series = trend
            cycle_series = cycle
        
        if return_trend:
            return cycle_series, trend_series
        else:
            return cycle_series
            
    except np.linalg.LinAlgError:
        # Fallback to linear detrending
        warnings.warn("HP filter failed, using linear detrending")
        x = np.arange(n)
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        cycle = y - trend
        
        if index is not None:
            trend_series = pd.Series(trend, index=index)
            cycle_series = pd.Series(cycle, index=index)
        else:
            trend_series = trend
            cycle_series = cycle
        
        if return_trend:
            return cycle_series, trend_series
        else:
            return cycle_series


def calculate_output_gap(
    gdp: Union[pd.Series, np.ndarray],
    method: str = "hp_filter",
    **kwargs
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate output gap using various methods.
    
    Args:
        gdp: GDP time series
        method: Method to use ('hp_filter', 'linear_trend', 'quadratic_trend')
        **kwargs: Additional arguments for the method
        
    Returns:
        Output gap as percentage of trend
    """
    if isinstance(gdp, pd.Series):
        log_gdp = np.log(gdp.replace(0, np.nan).dropna())
        index = log_gdp.index
    else:
        log_gdp = np.log(np.maximum(gdp, 1e-10))  # Avoid log(0)
        index = None
    
    if method == "hp_filter":
        lamb = kwargs.get("lambda", 1600)
        gap = hp_filter(log_gdp, lamb=lamb)
        
    elif method == "linear_trend":
        x = np.arange(len(log_gdp))
        coeffs = np.polyfit(x, log_gdp.values if hasattr(log_gdp, 'values') else log_gdp, 1)
        trend = np.polyval(coeffs, x)
        gap = log_gdp.values - trend if hasattr(log_gdp, 'values') else log_gdp - trend
        
    elif method == "quadratic_trend":
        x = np.arange(len(log_gdp))
        coeffs = np.polyfit(x, log_gdp.values if hasattr(log_gdp, 'values') else log_gdp, 2)
        trend = np.polyval(coeffs, x)
        gap = log_gdp.values - trend if hasattr(log_gdp, 'values') else log_gdp - trend
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to percentage
    gap_pct = gap * 100
    
    if index is not None:
        return pd.Series(gap_pct, index=index)
    else:
        return gap_pct


def estimate_ar_process(
    series: Union[pd.Series, np.ndarray],
    max_lags: int = 4,
    criterion: str = "aic"
) -> Dict[str, Any]:
    """
    Estimate AR process parameters.
    
    Args:
        series: Time series data
        max_lags: Maximum number of lags to consider
        criterion: Information criterion ('aic', 'bic')
        
    Returns:
        Dictionary with AR parameters and diagnostics
    """
    if isinstance(series, pd.Series):
        y = series.dropna().values
    else:
        y = np.asarray(series)
        y = y[~np.isnan(y)]
    
    n = len(y)
    
    if n < max_lags + 5:
        # Not enough data
        return {
            "coefficients": [0.5],
            "intercept": np.mean(y),
            "sigma": np.std(y),
            "aic": np.inf,
            "bic": np.inf,
            "selected_lags": 1
        }
    
    best_ic = np.inf
    best_result = None
    
    for p in range(1, max_lags + 1):
        try:
            # Create lagged variables
            Y = y[p:]
            X = np.column_stack([y[i:n-p+i] for i in range(p)])
            X = np.column_stack([np.ones(len(Y)), X])  # Add intercept
            
            # OLS estimation
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            
            # Calculate residuals and diagnostics
            Y_pred = X @ beta
            residuals = Y - Y_pred
            sigma = np.std(residuals)
            
            # Information criteria
            log_likelihood = -0.5 * len(Y) * (np.log(2 * np.pi) + 2 * np.log(sigma)) - 0.5 * np.sum((residuals / sigma)**2)
            k = p + 1  # Number of parameters
            
            aic = -2 * log_likelihood + 2 * k
            bic = -2 * log_likelihood + k * np.log(len(Y))
            
            ic_value = aic if criterion == "aic" else bic
            
            if ic_value < best_ic:
                best_ic = ic_value
                best_result = {
                    "coefficients": beta[1:],  # Exclude intercept
                    "intercept": beta[0],
                    "sigma": sigma,
                    "aic": aic,
                    "bic": bic,
                    "selected_lags": p,
                    "residuals": residuals
                }
                
        except np.linalg.LinAlgError:
            continue
    
    if best_result is None:
        # Fallback: simple AR(1)
        if n > 2:
            y_lag = y[:-1]
            y_current = y[1:]
            
            X = np.column_stack([np.ones(len(y_current)), y_lag])
            beta = np.linalg.lstsq(X, y_current, rcond=None)[0]
            
            best_result = {
                "coefficients": [beta[1]],
                "intercept": beta[0],
                "sigma": np.std(y_current - X @ beta),
                "aic": np.inf,
                "bic": np.inf,
                "selected_lags": 1
            }
        else:
            best_result = {
                "coefficients": [0.5],
                "intercept": np.mean(y),
                "sigma": np.std(y),
                "aic": np.inf,
                "bic": np.inf,
                "selected_lags": 1
            }
    
    return best_result


def calculate_inflation_persistence(
    inflation: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate inflation persistence using AR(1) coefficient.
    
    Args:
        inflation: Inflation time series
        
    Returns:
        Persistence coefficient (AR(1) parameter)
    """
    ar_result = estimate_ar_process(inflation, max_lags=1)
    return ar_result["coefficients"][0]


def estimate_taylor_rule_ols(
    interest_rate: pd.Series,
    inflation: pd.Series,
    output_gap: pd.Series,
    inflation_target: float = 2.0
) -> Dict[str, float]:
    """
    Estimate Taylor rule using OLS.
    
    Taylor rule: i_t = α + β₁(π_t - π*) + β₂y_t + β₃i_{t-1} + ε_t
    
    Args:
        interest_rate: Policy interest rate
        inflation: Inflation rate
        output_gap: Output gap
        inflation_target: Inflation target
        
    Returns:
        Dictionary with Taylor rule coefficients
    """
    # Align data
    data = pd.DataFrame({
        "interest_rate": interest_rate,
        "inflation": inflation,
        "output_gap": output_gap
    }).dropna()
    
    if len(data) < 10:
        # Not enough data, return default values
        return {
            "intercept": 2.0,
            "inflation_response": 1.5,
            "output_response": 0.5,
            "smoothing": 0.8,
            "r_squared": 0.0
        }
    
    # Create variables
    y = data["interest_rate"].values
    inflation_gap = (data["inflation"] - inflation_target).values
    output_gap_vals = data["output_gap"].values
    
    # Lagged interest rate for smoothing
    y_lag = np.r_[y[0], y[:-1]]  # Use first value as initial lag
    
    # Regression
    X = np.column_stack([
        np.ones(len(y)),     # Intercept
        inflation_gap,        # Inflation gap
        output_gap_vals,      # Output gap
        y_lag                 # Lagged interest rate
    ])
    
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "intercept": coeffs[0],
            "inflation_response": coeffs[1] + 1.0,  # Total response to inflation
            "output_response": coeffs[2],
            "smoothing": coeffs[3],
            "r_squared": r_squared
        }
        
    except np.linalg.LinAlgError:
        # Fallback to default values
        return {
            "intercept": 2.0,
            "inflation_response": 1.5,
            "output_response": 0.5,
            "smoothing": 0.8,
            "r_squared": 0.0
        }


def estimate_consumption_function(
    consumption: pd.Series,
    income: pd.Series,
    wealth: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Estimate consumption function parameters.
    
    C_t = α + β₁Y_t + β₂W_t + ε_t
    
    Args:
        consumption: Consumption time series
        income: Income time series
        wealth: Wealth time series (optional)
        
    Returns:
        Dictionary with consumption function parameters
    """
    # Align data
    if wealth is not None:
        data = pd.DataFrame({
            "consumption": consumption,
            "income": income,
            "wealth": wealth
        }).dropna()
    else:
        data = pd.DataFrame({
            "consumption": consumption,
            "income": income
        }).dropna()
    
    if len(data) < 5:
        # Not enough data
        return {
            "mpc": 0.7,
            "wealth_effect": 0.05 if wealth is not None else 0.0,
            "intercept": 0.0,
            "r_squared": 0.0
        }
    
    # Regression
    y = data["consumption"].values
    X = [np.ones(len(y)), data["income"].values]
    
    if wealth is not None:
        X.append(data["wealth"].values)
    
    X = np.column_stack(X)
    
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate R-squared
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        result = {
            "intercept": coeffs[0],
            "mpc": coeffs[1],
            "r_squared": r_squared
        }
        
        if wealth is not None:
            result["wealth_effect"] = coeffs[2]
        else:
            result["wealth_effect"] = 0.0
        
        return result
        
    except np.linalg.LinAlgError:
        return {
            "mpc": 0.7,
            "wealth_effect": 0.05 if wealth is not None else 0.0,
            "intercept": 0.0,
            "r_squared": 0.0
        }


def calculate_volatility_measures(
    series: Union[pd.Series, np.ndarray],
    window: int = 12
) -> Dict[str, float]:
    """
    Calculate various volatility measures.
    
    Args:
        series: Time series data
        window: Rolling window for calculations
        
    Returns:
        Dictionary with volatility measures
    """
    if isinstance(series, pd.Series):
        values = series.dropna()
    else:
        values = pd.Series(series).dropna()
    
    if len(values) < window:
        return {
            "unconditional_volatility": values.std(),
            "rolling_volatility_mean": values.std(),
            "rolling_volatility_std": 0.0,
            "garch_volatility": values.std()
        }
    
    # Unconditional volatility
    unconditional_vol = values.std()
    
    # Rolling volatility
    rolling_vol = values.rolling(window).std().dropna()
    rolling_vol_mean = rolling_vol.mean()
    rolling_vol_std = rolling_vol.std()
    
    # Simple GARCH(1,1) approximation
    returns = values.pct_change().dropna()
    
    if len(returns) > 10:
        # Estimate GARCH parameters using method of moments
        unconditional_var = returns.var()
        
        # Simple persistence measure
        returns_sq = returns**2
        
        if len(returns_sq) > 1:
            persistence = returns_sq.autocorr(lag=1)
            persistence = max(0, min(0.99, persistence))  # Bound between 0 and 0.99
        else:
            persistence = 0.5
        
        garch_volatility = np.sqrt(unconditional_var / (1 - persistence))
    else:
        garch_volatility = unconditional_vol
    
    return {
        "unconditional_volatility": unconditional_vol,
        "rolling_volatility_mean": rolling_vol_mean,
        "rolling_volatility_std": rolling_vol_std,
        "garch_volatility": garch_volatility
    }


def generate_pareto_distribution(
    alpha: float,
    size: int,
    scale: float = 1.0
) -> np.ndarray:
    """
    Generate Pareto-distributed random variables.
    
    Args:
        alpha: Shape parameter
        size: Number of samples
        scale: Scale parameter
        
    Returns:
        Array of Pareto-distributed values
    """
    # Use inverse transform sampling
    u = np.random.uniform(0, 1, size)
    x = scale * ((1 - u) ** (-1/alpha))
    return x


def calibrate_wealth_distribution(
    total_wealth: float,
    population_size: int,
    gini_coefficient: float = 0.8,
    pareto_tail: float = 0.2
) -> np.ndarray:
    """
    Calibrate wealth distribution to match empirical properties.
    
    Args:
        total_wealth: Total wealth in the economy
        population_size: Number of agents
        gini_coefficient: Target Gini coefficient
        pareto_tail: Fraction of population in Pareto tail
        
    Returns:
        Array of wealth values
    """
    # Calculate Pareto alpha from Gini coefficient
    # For pure Pareto: Gini = 1/(2α-1)
    # Solve for α: α = (1 + Gini)/(2 * Gini)
    if gini_coefficient >= 0.5:
        alpha = (1 + gini_coefficient) / (2 * gini_coefficient)
    else:
        alpha = 2.0  # Default value
    
    # Generate wealth distribution
    n_pareto = int(population_size * pareto_tail)
    n_regular = population_size - n_pareto
    
    if n_pareto > 0:
        # Pareto tail for rich agents
        pareto_wealth = generate_pareto_distribution(alpha, n_pareto, scale=1.0)
        pareto_wealth = pareto_wealth / np.sum(pareto_wealth) * total_wealth * 0.8  # Rich get 80% of wealth
    else:
        pareto_wealth = np.array([])
    
    if n_regular > 0:
        # Log-normal for the rest
        mu = 0
        sigma = 0.6
        lognormal_wealth = np.random.lognormal(mu, sigma, n_regular)
        lognormal_wealth = lognormal_wealth / np.sum(lognormal_wealth) * total_wealth * 0.2  # Rest get 20%
    else:
        lognormal_wealth = np.array([])
    
    # Combine and shuffle
    all_wealth = np.concatenate([pareto_wealth, lognormal_wealth])
    np.random.shuffle(all_wealth)
    
    # Ensure total wealth constraint
    all_wealth = all_wealth / np.sum(all_wealth) * total_wealth
    
    return all_wealth


def calculate_concentration_ratios(
    distribution: np.ndarray,
    ratios: List[float] = [0.01, 0.05, 0.1, 0.2]
) -> Dict[str, float]:
    """
    Calculate concentration ratios for a distribution.
    
    Args:
        distribution: Array of values (e.g., wealth, income)
        ratios: List of top percentiles to calculate
        
    Returns:
        Dictionary with concentration ratios
    """
    sorted_dist = np.sort(distribution)[::-1]  # Sort descending
    total = np.sum(sorted_dist)
    n = len(sorted_dist)
    
    concentration = {}
    
    for ratio in ratios:
        top_n = int(n * ratio)
        if top_n > 0:
            top_share = np.sum(sorted_dist[:top_n]) / total
            concentration[f"top_{ratio:.0%}"] = top_share
        else:
            concentration[f"top_{ratio:.0%}"] = 0.0
    
    return concentration


def calculate_gini_coefficient(distribution: np.ndarray) -> float:
    """
    Calculate Gini coefficient.
    
    Args:
        distribution: Array of values
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Remove negative values and sort
    dist = distribution[distribution >= 0]
    
    if len(dist) == 0:
        return 0.0
    
    dist = np.sort(dist)
    n = len(dist)
    
    if np.sum(dist) == 0:
        return 0.0
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * dist)) / (n * np.sum(dist)) - (n + 1) / n
    
    return gini


def optimize_parameter_bounds(
    objective_function: Callable,
    initial_params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    method: str = "L-BFGS-B"
) -> Dict[str, Any]:
    """
    Optimize parameters within bounds.
    
    Args:
        objective_function: Function to minimize
        initial_params: Initial parameter values
        bounds: Parameter bounds
        method: Optimization method
        
    Returns:
        Dictionary with optimization results
    """
    # Convert to arrays for scipy.optimize
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[name] for name in param_names])
    bounds_array = [bounds[name] for name in param_names]
    
    def objective_wrapper(x):
        params_dict = dict(zip(param_names, x))
        return objective_function(params_dict)
    
    try:
        result = optimize.minimize(
            objective_wrapper,
            x0,
            method=method,
            bounds=bounds_array
        )
        
        optimized_params = dict(zip(param_names, result.x))
        
        return {
            "success": result.success,
            "parameters": optimized_params,
            "objective_value": result.fun,
            "iterations": result.nit if hasattr(result, 'nit') else None,
            "message": result.message
        }
        
    except Exception as e:
        return {
            "success": False,
            "parameters": initial_params,
            "objective_value": np.inf,
            "iterations": 0,
            "message": str(e)
        }


def bootstrap_parameter_confidence_intervals(
    data: pd.DataFrame,
    estimation_function: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for parameters.
    
    Args:
        data: Original data
        estimation_function: Function that estimates parameters from data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary with confidence intervals for each parameter
    """
    n_obs = len(data)
    bootstrap_results = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(n_obs, n_obs, replace=True)
        bootstrap_data = data.iloc[bootstrap_indices]
        
        try:
            # Estimate parameters on bootstrap sample
            params = estimation_function(bootstrap_data)
            bootstrap_results.append(params)
        except:
            # Skip failed estimations
            continue
    
    if not bootstrap_results:
        return {}
    
    # Convert to DataFrame for easier processing
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    confidence_intervals = {}
    for param_name in bootstrap_df.columns:
        lower = np.percentile(bootstrap_df[param_name], lower_percentile)
        upper = np.percentile(bootstrap_df[param_name], upper_percentile)
        confidence_intervals[param_name] = (lower, upper)
    
    return confidence_intervals


def validate_parameter_stability(
    parameters: Dict[str, List[float]],
    window_size: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Validate parameter stability across different time windows.
    
    Args:
        parameters: Dictionary with parameter time series
        window_size: Size of rolling window
        
    Returns:
        Dictionary with stability statistics for each parameter
    """
    stability_stats = {}
    
    for param_name, param_values in parameters.items():
        if len(param_values) < window_size:
            stability_stats[param_name] = {
                "mean": np.mean(param_values),
                "std": np.std(param_values),
                "cv": np.inf,
                "stability_ratio": 0.0
            }
            continue
        
        # Rolling statistics
        param_series = pd.Series(param_values)
        rolling_mean = param_series.rolling(window_size).mean().dropna()
        rolling_std = param_series.rolling(window_size).std().dropna()
        
        # Overall statistics
        overall_mean = param_series.mean()
        overall_std = param_series.std()
        cv = overall_std / abs(overall_mean) if overall_mean != 0 else np.inf
        
        # Stability ratio: inverse of coefficient of variation of rolling means
        rolling_mean_std = rolling_mean.std()
        rolling_mean_mean = rolling_mean.mean()
        stability_ratio = abs(rolling_mean_mean) / rolling_mean_std if rolling_mean_std > 0 else np.inf
        
        stability_stats[param_name] = {
            "mean": overall_mean,
            "std": overall_std,
            "cv": cv,
            "stability_ratio": stability_ratio
        }
    
    return stability_stats