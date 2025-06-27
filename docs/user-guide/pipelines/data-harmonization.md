# Data Harmonization Pipeline

## Overview

The Data Harmonization Pipeline ensures temporal consistency and data quality across economic time series. It handles missing data interpolation, calculates deflators for nominal/real variable pairs, applies country-specific adjustments, and aligns quarterly and annual data frequencies.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/processors/harmonizer.py`
- **Class**: `DataHarmonizer`
- **Dependencies**: Currency Conversion Pipeline
- **MATLAB Equivalent**: Data harmonization logic in `b_calibration_data.m`
- **Phase**: Data Processing

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `converted_data` | `Dict[Country, ConvertedDataset]` | Currency Conversion | Yes | USD-converted economic time series |
| `timeframe` | `TimeFrame` | Configuration | Yes | Target time period for harmonization |
| `variables` | `List[str]` | Configuration | Yes | Variables requiring harmonization |
| `frequency_preference` | `str` | Configuration | No | Preferred data frequency (quarterly/annual) |

### Input Validation

- Verify time series completeness and temporal coverage
- Check for required nominal/real variable pairs for deflator calculation
- Validate data types and numeric consistency
- Ensure sufficient data points for interpolation operations

### Configuration Parameters

```yaml
processing:
  interpolation_method: "linear"
  frequency_preference: "quarterly"
  
harmonization:
  deflator_calculation: true
  missing_data_threshold: 0.8
  outlier_detection: true
  seasonal_adjustment: false
  
country_adjustments:
  enable_adjustments: true
  validation_required: true
```

## Data Treatment

### Standard Treatment

The pipeline applies consistent harmonization procedures to ensure data quality and temporal alignment across all countries.

#### Processing Steps

1. **Missing Data Interpolation**
   - Linear interpolation for gaps within time series
   - Nearest neighbor for edge cases
   - Quality flags for interpolated values
   - Validation of interpolation results

2. **Deflator Calculation**
   - Calculate deflators as nominal/real ratios: `deflator = nominal_var / real_var`
   - Apply to GDP, consumption, investment, trade variables
   - Smooth deflator series to remove noise
   - Validate deflator economic reasonableness

3. **Frequency Alignment**
   - Convert annual data to quarterly using interpolation
   - Aggregate quarterly data to annual when needed
   - Maintain temporal consistency across variables
   - Handle mixed-frequency datasets

4. **Data Quality Validation**
   - Outlier detection and flagging
   - Consistency checks across related variables
   - Temporal smoothness validation
   - Economic relationship verification

#### Standard Algorithms

```python
def harmonize_time_series_standard(self, data: pd.DataFrame, 
                                 country: Country) -> pd.DataFrame:
    """Standard harmonization applied to all countries."""
    
    # Step 1: Handle missing data
    data = self.interpolate_missing_data(data, method="linear")
    
    # Step 2: Calculate deflators
    if self.config.harmonization.deflator_calculation:
        data = self.calculate_deflators(data)
    
    # Step 3: Frequency alignment
    data = self.align_frequencies(data, target="quarterly")
    
    # Step 4: Quality validation
    data = self.validate_data_quality(data)
    
    return data
```

### Country-Specific Treatment

Special adjustments for countries with unique data characteristics or reporting methodologies.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| MEX | Price base adjustment | Change price base from "L" to "Q" | Different Mexican accounting standards |
| USA | Data reconstruction | Government consumption from growth rates | Historical reporting discontinuities |

#### Implementation Details

```python
def apply_country_specific_harmonization(self, data: pd.DataFrame, 
                                       country: Country) -> pd.DataFrame:
    """Apply country-specific harmonization adjustments."""
    
    if country == Country.MEX:
        # Mexico: Price base adjustment for GDP components
        data = self.apply_mexico_price_base_adjustment(data)
        self.metadata.adjustments_applied["MEX_price_base"] = True
        
    elif country == Country.USA:
        # USA: Government consumption reconstruction
        data = self.apply_usa_government_consumption_fix(data)
        self.metadata.adjustments_applied["USA_gov_consumption"] = True
    
    return data

def apply_mexico_price_base_adjustment(self, data: pd.DataFrame) -> pd.DataFrame:
    """Apply Mexico's price base change from L to Q."""
    gdp_variables = ['real_gdp', 'real_consumption', 'real_investment', 
                     'real_exports', 'real_imports']
    
    for var in gdp_variables:
        if var in data.columns:
            # Apply price base correction factor
            data[var] = data[var] * self.get_mexico_price_base_factor()
    
    return data

def apply_usa_government_consumption_fix(self, data: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct USA government consumption from growth rates."""
    if 'government_consumption_growth' in data.columns:
        # Backward calculation from growth rates
        data['government_consumption'] = self.reconstruct_from_growth_rates(
            data['government_consumption_growth'],
            method="backward_calculation"
        )
    
    return data
```

### Error Handling

- **Interpolation Failures**: Fall back to nearest neighbor or forward fill
- **Deflator Anomalies**: Flag unrealistic deflator values and use smoothed series
- **Missing Variable Pairs**: Skip deflator calculation and log warnings
- **Frequency Conflicts**: Prioritize higher frequency data and document decisions

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `harmonized_data` | `Dict[Country, HarmonizedDataset]` | Structured dict | Consistent time series data |
| `deflators` | `Dict[Country, pd.DataFrame]` | DataFrame | Calculated price deflators |
| `harmonization_metadata` | `HarmonizationMetadata` | Structured object | Processing methods and adjustments |
| `quality_flags` | `Dict[Country, QualityFlags]` | Structured dict | Data quality indicators |

### Output Validation

- **Temporal Consistency**: Verify smooth time series transitions
- **Economic Reasonableness**: Check deflator values and growth rates
- **Data Completeness**: Ensure all required variables present
- **Cross-Variable Consistency**: Validate relationships between related variables

### Metadata

Information included with outputs:
- **Interpolation Methods**: Techniques used for missing data
- **Country Adjustments**: Specific modifications applied by country
- **Quality Flags**: Indicators for interpolated, adjusted, or flagged data
- **Deflator Statistics**: Summary statistics for calculated deflators
- **Processing Timestamps**: When harmonization was performed

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n×m×t×log(t)) where n=countries, m=variables, t=time periods
- **Memory Usage**: Moderate - working datasets and intermediate calculations
- **CPU Intensive**: Interpolation and deflator calculations

### Scalability

- **Country Scaling**: Linear increase in processing time
- **Variable Scaling**: Linear increase with number of variables
- **Time Period Scaling**: Log-linear due to interpolation algorithms
- **Processing Time**: ~2-5 seconds per country for typical datasets

## Integration Points

### Upstream Integration

Consumes outputs from Currency Conversion Pipeline:
- **USD-Converted Data**: Monetary values in common currency
- **Time Series**: Economic variables requiring harmonization
- **Metadata**: Currency conversion quality flags
- **Country Mappings**: For applying country-specific adjustments

### Downstream Integration

Provides quality-assured data to subsequent pipelines:
- **Parameter Estimation**: Consistent time series for econometric analysis
- **Industry Aggregation**: Harmonized data for ICIO processing
- **Model Calibration**: Quality-validated datasets
- **Validation Framework**: Metadata for quality assessment

## Validation and Testing

### Unit Tests

```python
def test_missing_data_interpolation():
    """Test linear interpolation for missing values."""
    
def test_deflator_calculation():
    """Test nominal/real deflator computation."""
    
def test_mexico_price_base_adjustment():
    """Test Mexico's special price base handling."""
    
def test_usa_government_consumption_reconstruction():
    """Test USA's government consumption fixing logic."""
    
def test_frequency_alignment():
    """Test quarterly/annual data alignment."""
```

### Integration Tests

- **End-to-End Harmonization**: Complete workflow validation
- **Cross-Country Consistency**: Verify consistent processing across countries
- **MATLAB Comparison**: Validate against original MATLAB harmonization
- **Economic Validation**: Check economic relationship preservation

## Configuration Examples

### Basic Configuration

```yaml
processing:
  interpolation_method: "linear"
  frequency_preference: "quarterly"
  
harmonization:
  deflator_calculation: true
  missing_data_threshold: 0.8
```

### Advanced Configuration

```yaml
processing:
  interpolation_method: "linear"
  frequency_preference: "quarterly"
  
harmonization:
  deflator_calculation: true
  missing_data_threshold: 0.8
  outlier_detection: true
  outlier_threshold: 3.0
  seasonal_adjustment: false
  smoothing_window: 4
  
country_adjustments:
  enable_adjustments: true
  validation_required: true
  
validation:
  economic_consistency_checks: true
  deflator_bounds: [0.1, 10.0]
  growth_rate_bounds: [-0.5, 0.5]
```

### Country-Specific Configuration

```yaml
harmonization:
  country_adjustments:
    MEX:
      price_base_adjustment: true
      price_base_factor: 1.02
      affected_variables: ["real_gdp", "real_consumption", "real_investment"]
    USA:
      government_consumption_reconstruction: true
      reconstruction_method: "backward_calculation"
      validation_against_levels: true
      
  adjustment_validation:
    cross_check_sources: true
    economic_theory_validation: true
    matlab_compatibility_check: true
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Interpolation failures | `ValueError` in missing data handling | Adjust interpolation method, check data availability |
| Unrealistic deflators | Deflator values outside economic bounds | Review nominal/real data quality, apply smoothing |
| Mexico adjustment errors | Incorrect GDP values after processing | Verify price base factor, check variable mapping |
| USA data reconstruction issues | Government consumption discontinuities | Validate growth rate data, adjust reconstruction method |
| Frequency alignment problems | Temporal mismatches in output | Check input data frequencies, adjust alignment logic |

### Debug Information

- **Harmonization Logs**: Detailed processing steps and adjustments
- **Quality Metrics**: Data completeness and consistency statistics
- **Adjustment Tracking**: Country-specific modifications applied
- **Performance Monitoring**: Processing time and memory usage by country
- **Validation Results**: Economic consistency and reasonableness checks

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Country-Specific Processing](../country-specific-processing.md) - Detailed adjustment explanations
- [Currency Conversion Pipeline](currency-conversion.md) - Previous pipeline in workflow
- [Industry Aggregation Pipeline](industry-aggregation.md) - Parallel pipeline in workflow
- [Parameter Estimation Pipeline](parameter-estimation.md) - Next pipeline in workflow