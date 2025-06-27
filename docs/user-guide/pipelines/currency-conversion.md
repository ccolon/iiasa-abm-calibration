# Currency Conversion Pipeline

## Overview

The Currency Conversion Pipeline converts all monetary values from local currencies to a common target currency (USD) to enable cross-country economic comparisons. It implements complex EUR-based conversion logic and handles special cases for countries with limited historical exchange rate data.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/processors/currency.py`
- **Class**: `CurrencyConverter`
- **Dependencies**: Raw Data Extraction Pipeline
- **MATLAB Equivalent**: Currency conversion logic in `b_calibration_data.m`
- **Phase**: Data Processing

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `economic_data` | `Dict[Country, RawDataset]` | Raw Data Extraction | Yes | Economic time series with mixed currencies |
| `exchange_rates` | `Dict[str, pd.DataFrame]` | Raw Data Extraction | Yes | EUR-based exchange rate tables |
| `target_currency` | `str` | Configuration | Yes | Target currency for conversion (default: USD) |
| `currency_mappings` | `Dict[str, str]` | Configuration | Yes | Country to currency code mappings |

### Input Validation

- Verify exchange rate data availability for all required currencies
- Check economic data contains monetary variables requiring conversion
- Validate currency code mappings completeness
- Ensure overlapping time periods between economic data and exchange rates

### Configuration Parameters

```yaml
processing:
  target_currency: "USD"
  interpolation_method: "linear"
  
currency_conversion:
  special_currencies: ["ILS", "KRW", "MXN"]
  interpolation_fallback: "nearest"
  extrapolation_method: "nearest"
  validation_tolerance: 0.001
```

## Data Treatment

### Standard Treatment

The pipeline applies EUR-based conversion logic uniformly across most countries using Eurostat exchange rate data.

#### Processing Steps

1. **Currency Mapping**
   - Map OECD country codes to currency codes
   - Identify monetary variables requiring conversion
   - Validate data availability for conversion period

2. **EUR-Based Conversion Logic**
   - For non-EUR countries: `LC_to_USD = (LC_to_EUR) / (USD_to_EUR)`
   - Where LC = Local Currency, calculated as inverse relationship
   - Extract USD_to_EUR rate from Eurostat bilateral tables

3. **Rate Application**
   - Apply quarterly rates to quarterly data
   - Apply annual rates to annual data
   - Maintain temporal alignment between data and rates

#### Standard Algorithms

```python
def convert_currency_standard(self, data: pd.DataFrame, 
                            from_currency: str, 
                            to_currency: str = "USD") -> pd.DataFrame:
    """Standard EUR-based currency conversion."""
    
    if from_currency == "EUR":
        # Direct USD conversion for EUR countries
        eur_to_usd = self.get_eur_to_usd_rate(data.index)
        return data * eur_to_usd
    else:
        # Complex conversion: LC → EUR → USD
        lc_to_eur = self.get_exchange_rate(from_currency, "EUR", data.index)
        eur_to_usd = self.get_eur_to_usd_rate(data.index)
        return data * lc_to_eur / eur_to_usd
```

### Country-Specific Treatment

Special handling for countries with data limitations or unique currency characteristics.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| MEX | Limited historical data | Enhanced interpolation + extrapolation | Eurostat MXN data gaps before 2000 |
| ISR | Limited historical data | Nearest neighbor extrapolation | ILS not in early Eurostat databases |
| KOR | Limited historical data | Nearest neighbor extrapolation | KRW limited coverage in historical data |
| EUR Countries (19) | Direct calculation | USD = 1/USD_to_EUR rate | No separate conversion needed |

#### Implementation Details

```python
def apply_country_specific_conversion(self, country: Country, 
                                    data: pd.DataFrame) -> pd.DataFrame:
    """Apply country-specific currency conversion logic."""
    
    currency = self.get_country_currency(country)
    
    if country in [Country.MEX, Country.ISR, Country.KOR]:
        # Special handling for limited data currencies
        exchange_rates = self.get_exchange_rate_with_interpolation(
            currency, "USD", data.index,
            method="enhanced_interpolation"
        )
    elif currency == "EUR":
        # Direct EUR to USD conversion
        exchange_rates = self.get_direct_eur_usd_rate(data.index)
    else:
        # Standard EUR-based conversion
        exchange_rates = self.get_standard_conversion_rate(
            currency, "USD", data.index
        )
    
    return data * exchange_rates
```

**Enhanced Interpolation for Special Cases**:
- **Linear interpolation** for available data periods
- **Nearest neighbor extrapolation** for missing historical periods
- **Quality flags** for interpolated vs observed data
- **Validation** against alternative data sources where available

### Error Handling

- **Missing Exchange Rates**: Apply interpolation or use alternative sources
- **Data Type Mismatches**: Convert and validate numeric data
- **Temporal Misalignment**: Align exchange rates with data frequencies
- **Conversion Failures**: Log errors and continue with unconverted data

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `converted_data` | `Dict[Country, ConvertedDataset]` | Structured dict | Economic data in target currency |
| `conversion_rates` | `Dict[str, pd.DataFrame]` | DataFrame | Applied exchange rates by currency |
| `conversion_metadata` | `ConversionMetadata` | Structured object | Conversion methods and quality flags |
| `quality_flags` | `Dict[str, List[str]]` | Dict of lists | Data quality indicators by currency |

### Output Validation

- **Conversion Completeness**: Verify all monetary variables converted
- **Rate Reasonableness**: Check for extreme exchange rate movements
- **Temporal Consistency**: Ensure smooth time series transitions
- **Cross-Validation**: Compare with alternative exchange rate sources

### Metadata

Information included with outputs:
- **Conversion Methods**: Standard vs enhanced interpolation used
- **Data Quality Flags**: Interpolated vs observed exchange rates
- **Coverage Statistics**: Percentage of interpolated data points
- **Rate Sources**: Eurostat table sources and vintage information
- **Validation Results**: Cross-validation against alternative sources

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n×m×t) where n=countries, m=variables, t=time periods
- **Memory Usage**: Moderate - exchange rate tables and economic data
- **I/O Requirements**: Database queries for exchange rate tables

### Scalability

- **Country Scaling**: Linear increase with number of countries
- **Time Period Scaling**: Linear increase in memory and processing
- **Currency Complexity**: Additional overhead for special currency handling
- **Processing Time**: ~1-2 seconds per country for typical timeframes

## Integration Points

### Upstream Integration

Consumes outputs from Raw Data Extraction Pipeline:
- **Economic Data**: Raw time series requiring currency conversion
- **Exchange Rate Tables**: EUR-based bilateral rates from Eurostat
- **Country Mappings**: OECD country codes to currency codes
- **Timeframe Alignment**: Matching data and exchange rate periods

### Downstream Integration

Provides standardized USD data to subsequent pipelines:
- **Data Harmonization**: USD-converted economic time series
- **Industry Aggregation**: Monetary ICIO data in common currency
- **Parameter Estimation**: Consistent monetary units for estimation
- **Quality Assurance**: Conversion metadata for validation

## Validation and Testing

### Unit Tests

```python
def test_eur_to_usd_conversion():
    """Test direct EUR to USD conversion logic."""
    
def test_complex_currency_conversion():
    """Test LC → EUR → USD conversion chain."""
    
def test_interpolation_for_special_currencies():
    """Test enhanced interpolation for MXN, ILS, KRW."""
    
def test_eurozone_optimization():
    """Test direct calculation for EUR countries."""
```

### Integration Tests

- **Cross-Currency Validation**: Compare conversion results across methods
- **MATLAB Compatibility**: Verify identical conversion results
- **Data Consistency**: Validate converted data reasonableness
- **Performance Benchmarks**: Measure conversion speed and memory usage

## Configuration Examples

### Basic Configuration

```yaml
processing:
  target_currency: "USD"
  
currency_conversion:
  interpolation_method: "linear"
  validation_tolerance: 0.001
```

### Advanced Configuration

```yaml
processing:
  target_currency: "USD"
  
currency_conversion:
  special_currencies: ["ILS", "KRW", "MXN"]
  interpolation_method: "linear"
  interpolation_fallback: "nearest"
  extrapolation_method: "nearest"
  validation_tolerance: 0.001
  cross_validation: true
  alternative_sources: ["ECB", "FRED"]
  
performance:
  parallel_countries: true
  cache_exchange_rates: true
  memory_efficient: true
```

### Country-Specific Configuration

```yaml
currency_conversion:
  country_overrides:
    MEX:
      interpolation_method: "enhanced"
      extrapolation_method: "nearest"
      validation_sources: ["BANXICO"]
    ISR:
      interpolation_method: "nearest"
      extrapolation_method: "nearest"
    KOR:
      interpolation_method: "nearest"
      extrapolation_method: "nearest"
  
  eurozone_optimization: true
  direct_eur_calculation: true
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Missing exchange rate data | `KeyError` for currency codes | Check Eurostat table completeness, enable interpolation |
| Extreme conversion rates | Unrealistic converted values | Validate exchange rate data, check for data entry errors |
| Interpolation failures | `ValueError` in rate calculation | Increase interpolation tolerance, use fallback methods |
| EUR conversion errors | Incorrect USD rates for Eurozone | Verify USD_to_EUR rate extraction from bilateral tables |
| Memory issues with large datasets | `MemoryError` during processing | Enable memory-efficient mode, process countries in batches |

### Debug Information

- **Conversion Logs**: Detailed currency conversion steps and methods
- **Rate Validation**: Exchange rate reasonableness checks and warnings
- **Quality Metrics**: Percentage of interpolated vs observed data
- **Performance Monitoring**: Processing time and memory usage by country
- **Cross-Validation Results**: Comparison with alternative rate sources

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Country-Specific Processing](../country-specific-processing.md) - Special handling details
- [Data Harmonization Pipeline](data-harmonization.md) - Next pipeline in workflow
- [Data Sources](../data-sources.md) - Eurostat exchange rate tables