# Parameter Estimation Pipeline

## Overview

The Parameter Estimation Pipeline estimates behavioral parameters for ABM agents using processed economic data. It implements econometric estimation for Taylor rule parameters, firm behavioral coefficients, and household decision-making parameters that govern agent behavior in the macroeconomic simulation model.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/calibrators/parameters.py`
- **Class**: `ABMParameterEstimator`
- **Dependencies**: All Phase 1 pipelines (Data Harmonization + Industry Aggregation)
- **MATLAB Equivalent**: Parameter estimation scripts in calibration workflow
- **Phase**: Model Calibration

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `harmonized_data` | `Dict[Country, HarmonizedDataset]` | Data Harmonization | Yes | Consistent economic time series |
| `aggregated_industry_data` | `Dict[Country, AggregatedDataset]` | Industry Aggregation | Yes | NACE2 sectoral data |
| `estimation_config` | `EstimationConfig` | Configuration | Yes | Estimation methods and parameters |
| `target_countries` | `List[Country]` | Configuration | Yes | Countries for parameter estimation |

### Input Validation

- Verify sufficient time series length for econometric estimation
- Check data completeness for required economic variables
- Validate temporal alignment across different data sources
- Ensure industry data availability for sectoral parameter estimation

### Configuration Parameters

```yaml
calibration:
  taylor_rule_estimation: true
  hp_filter_lambda: 1600
  ar_estimation_lags: 4
  confidence_level: 0.95
  
parameter_estimation:
  method: "ols"
  robust_errors: true
  time_trend: false
  seasonal_dummies: false
  
validation:
  economic_bounds_checking: true
  parameter_significance_testing: true
  model_fit_requirements: 0.5
```

## Data Treatment

### Standard Treatment

The pipeline applies consistent econometric estimation procedures across all countries to ensure parameter comparability.

#### Processing Steps

1. **Data Preparation**
   - HP filter application for trend-cycle decomposition
   - Variable transformation (logs, differences, ratios)
   - Outlier detection and handling
   - Missing data treatment for estimation periods

2. **Taylor Rule Estimation**
   - Estimate central bank reaction function: `i_t = α + β_π π_t + β_y y_t + ρ i_{t-1} + ε_t`
   - Where i = interest rate, π = inflation, y = output gap
   - Apply instrumental variables for endogeneity issues
   - Validate Taylor principle (β_π > 1.0)

3. **Firm Parameter Estimation**
   - Price adjustment speed from sectoral price data
   - Investment sensitivity from capital formation and demand
   - Markup elasticity from industry-level margins

4. **Household Parameter Estimation**
   - Marginal propensity to consume from consumption and income
   - Wealth effect magnitude from consumption and asset data
   - Labor supply elasticity from employment and wage relationships

#### Standard Algorithms

```python
def estimate_parameters_standard(self, data: HarmonizedDataset, 
                               country: Country) -> ParameterResults:
    """Standard parameter estimation applied to all countries."""
    
    # Prepare data for estimation
    estimation_data = self.prepare_estimation_data(data)
    
    # Taylor rule estimation
    taylor_params = self.estimate_taylor_rule(
        estimation_data['interest_rate'],
        estimation_data['inflation'],
        estimation_data['output_gap']
    )
    
    # Firm parameters
    firm_params = self.estimate_firm_parameters(
        estimation_data['prices'],
        estimation_data['investment'],
        estimation_data['demand']
    )
    
    # Household parameters
    household_params = self.estimate_household_parameters(
        estimation_data['consumption'],
        estimation_data['income'],
        estimation_data['wealth']
    )
    
    return ParameterResults(
        taylor_rule=taylor_params,
        firm_parameters=firm_params,
        household_parameters=household_params
    )
```

### Country-Specific Treatment

Parameter estimation applies uniform econometric methods across all countries without country-specific modifications.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| None | N/A | Standard estimation for all | Econometric consistency required |

#### Implementation Details

The parameter estimation maintains methodological consistency by applying identical estimation procedures to all countries:

```python
def estimate_all_countries(self, data_dict: Dict[Country, HarmonizedDataset]) -> Dict[Country, ParameterResults]:
    """Apply standard parameter estimation to all countries."""
    
    results = {}
    for country, country_data in data_dict.items():
        # Same estimation methodology for all countries
        results[country] = self.estimate_parameters_standard(
            country_data, country
        )
        
        # Apply universal validation
        self.validate_parameter_estimates(results[country], country)
    
    return results
```

**Methodological Consistency**:
- **Same Estimation Period**: Use identical time windows across countries
- **Same Model Specifications**: Apply consistent econometric models
- **Same Validation Criteria**: Universal parameter bounds and significance tests
- **Same Uncertainty Quantification**: Consistent confidence interval calculation

### Error Handling

- **Insufficient Data**: Skip estimation and use default parameters with warnings
- **Estimation Failures**: Try alternative methods or shorter sample periods
- **Parameter Bounds Violations**: Apply constraints and flag violations
- **Significance Issues**: Report low-significance parameters with warnings

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `estimated_parameters` | `Dict[Country, ParameterSet]` | Structured dict | Estimated behavioral parameters by country |
| `estimation_statistics` | `Dict[Country, EstimationStats]` | Structured dict | Goodness-of-fit and significance tests |
| `confidence_intervals` | `Dict[Country, ConfidenceIntervals]` | Structured dict | Parameter uncertainty quantification |
| `validation_results` | `Dict[Country, ValidationResults]` | Structured dict | Economic theory compliance checks |

### Output Validation

- **Economic Theory Compliance**: Verify parameters satisfy economic restrictions
- **Statistical Significance**: Check parameter significance at specified confidence levels
- **Parameter Bounds**: Ensure estimates within reasonable economic ranges
- **Model Fit**: Validate estimation quality and goodness-of-fit measures

### Metadata

Information included with outputs:
- **Estimation Methods**: Econometric techniques used for each parameter
- **Sample Periods**: Data periods used for estimation by country
- **Model Specifications**: Detailed specification of estimated equations
- **Validation Flags**: Economic theory and statistical compliance indicators
- **Confidence Levels**: Uncertainty quantification and significance testing results

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n×m×t²) where n=countries, m=parameters, t=time periods
- **CPU Intensive**: Econometric estimation and optimization routines
- **Memory Usage**: Moderate - estimation datasets and results storage

### Scalability

- **Country Scaling**: Linear increase in processing time
- **Parameter Scaling**: Linear increase with number of estimated parameters
- **Time Period Scaling**: Quadratic increase due to matrix operations in estimation
- **Processing Time**: ~10-30 seconds per country depending on model complexity

## Integration Points

### Upstream Integration

Consumes outputs from all Phase 1 pipelines:
- **Harmonized Economic Data**: Consistent time series for econometric analysis
- **Industry-Aggregated Data**: Sectoral information for firm parameter estimation
- **Exchange Rate Data**: For open economy parameter estimation
- **Quality Metadata**: Data quality flags affecting estimation reliability

### Downstream Integration

Provides estimated parameters to subsequent pipelines:
- **Initial Conditions Pipeline**: Parameters for agent population setup
- **Model Validation**: Parameters for economic theory compliance checking
- **Export System**: Estimated parameters in multiple output formats
- **Simulation Models**: Behavioral parameters for ABM execution

## Validation and Testing

### Unit Tests

```python
def test_taylor_rule_estimation():
    """Test Taylor rule parameter estimation and validation."""
    
def test_firm_parameter_estimation():
    """Test firm behavioral parameter estimation."""
    
def test_household_parameter_estimation():
    """Test household behavioral parameter estimation."""
    
def test_parameter_bounds_validation():
    """Test economic bounds checking for estimated parameters."""
    
def test_estimation_with_missing_data():
    """Test parameter estimation robustness to missing data."""
```

### Integration Tests

- **End-to-End Parameter Estimation**: Complete workflow from data to parameters
- **Cross-Country Consistency**: Verify reasonable parameter variation across countries
- **MATLAB Comparison**: Validate against original MATLAB parameter estimates
- **Economic Validation**: Check estimated parameters against economic literature

## Configuration Examples

### Basic Configuration

```yaml
calibration:
  taylor_rule_estimation: true
  hp_filter_lambda: 1600
  
parameter_estimation:
  method: "ols"
  robust_errors: true
```

### Advanced Configuration

```yaml
calibration:
  taylor_rule_estimation: true
  hp_filter_lambda: 1600
  ar_estimation_lags: 4
  confidence_level: 0.95
  
parameter_estimation:
  method: "2sls"  # Two-stage least squares for endogeneity
  robust_errors: true
  time_trend: false
  seasonal_dummies: true
  instrument_set: ["lagged_variables", "external_instruments"]
  
estimation_bounds:
  taylor_inflation_response: [1.01, 3.0]
  taylor_output_response: [0.0, 2.0]
  taylor_smoothing: [0.0, 0.99]
  mpc_bounds: [0.1, 0.9]
  
validation:
  economic_bounds_checking: true
  parameter_significance_testing: true
  model_fit_requirements: 0.5
  cross_equation_restrictions: true
```

### Country-Specific Configuration

```yaml
# No country-specific estimation configuration
# All countries use identical econometric methodology

parameter_estimation:
  uniform_methodology: true
  consistent_sample_periods: true
  standardized_validation: true
  
estimation_robustness:
  alternative_specifications: true
  sensitivity_analysis: true
  bootstrap_confidence_intervals: true
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Insufficient time series data | Estimation failures or unreliable results | Extend sample period or use panel data methods |
| Parameter bound violations | Estimates outside economic ranges | Apply constrained estimation or review data quality |
| Low statistical significance | High p-values for key parameters | Check model specification, increase sample size |
| Taylor principle violations | Inflation response < 1.0 | Review monetary policy data, consider regime changes |
| Estimation convergence failures | Optimization errors in parameter search | Try alternative starting values or estimation methods |

### Debug Information

- **Estimation Logs**: Detailed econometric estimation output and diagnostics
- **Parameter Statistics**: Estimates, standard errors, t-statistics, p-values
- **Model Diagnostics**: R-squared, residual tests, specification diagnostics
- **Validation Results**: Economic theory compliance and bound checking
- **Performance Metrics**: Estimation time and convergence information

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Data Harmonization Pipeline](data-harmonization.md) - Input data preparation
- [Industry Aggregation Pipeline](industry-aggregation.md) - Sectoral data preparation
- [Initial Conditions Pipeline](initial-conditions.md) - Next pipeline using parameters
- [Model Calibration](../calibration.md) - Detailed calibration methodology