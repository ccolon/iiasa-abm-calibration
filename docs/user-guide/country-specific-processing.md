# Country-Specific Data Processing

This document details the ad-hoc data processing tasks and special handling requirements for specific countries in the calibration system. These adjustments ensure data quality and consistency across different national statistical reporting methodologies.

## Overview

While the system processes most countries using standardized methods, several countries require special handling due to:

- **Data availability limitations** - Missing or incomplete historical data
- **Methodological differences** - Different national accounting standards
- **Currency system changes** - Euro adoption and exchange rate complexities
- **Statistical reporting variations** - Country-specific data collection methods

## Countries Requiring Special Processing

### 🇲🇽 Mexico (MEX) - Most Complex Processing

Mexico requires the most extensive special handling due to multiple data quality issues.

#### Price Base Adjustments

**Issue**: Mexican GDP data requires different price base methodology than other countries.

**Location**: `src/macro_abm_calibration/processors/harmonizer.py:35-49`

**Processing**:
```python
# Mexico-specific price base change
if country == Country.MEX:
    # Change from "L" (constant prices) to "Q" (quarterly reference)
    gdp_params = {
        "price_base": "Q",  # Instead of standard "L"
        "measure": "C"      # Constant prices
    }
```

**Affected Variables**:
- `real_gdp` - Real Gross Domestic Product
- `real_consumption` - Real private consumption
- `real_investment` - Real gross fixed capital formation
- `real_exports` - Real exports of goods and services
- `real_imports` - Real imports of goods and services

**MATLAB Origin**: Replicates special handling from original MATLAB code where Mexico required different price base calculations for economic consistency.

#### Exchange Rate Data Gaps

**Issue**: Limited historical Mexican Peso (MXN) exchange rate data in Eurostat database.

**Location**: `src/macro_abm_calibration/data_sources/eurostat.py:405-409`

**Processing**:
```python
# Special interpolation for MXN
if currency_code == "MXN":
    # Apply nearest neighbor extrapolation for missing historical data
    rates = extrapolate_missing_data(rates, method="nearest")
```

**Method**: Uses nearest neighbor interpolation to fill data gaps, replicating MATLAB's `interp1` with 'nearest' and 'extrap' options.

### 🇺🇸 United States (USA) - Growth Rate Reconstruction

**Issue**: Historical inconsistencies in US government consumption data reporting.

**Location**: `src/macro_abm_calibration/processors/harmonizer.py:43-48`

**Processing**:
```python
# USA-specific government consumption reconstruction
if country == Country.USA:
    adjustments = {
        "government_consumption": {
            "method": "backward_calculation",
            "source": "growth_rates"
        }
    }
```

**Method**: Reconstructs time series data from growth rates using backward calculation to ensure historical consistency.

**Reason**: The US has reporting methodology changes over time that create discontinuities in level data, but growth rates remain consistent.

### 🇮🇱 Israel (ISR) - Exchange Rate Interpolation

**Issue**: Limited historical Israeli Shekel (ILS) exchange rate data availability.

**Location**: `src/macro_abm_calibration/processors/currency.py:234-281`

**Processing**:
```python
# Special handling for ILS
SPECIAL_CURRENCIES = ["ILS", "KRW", "MXN"]

if currency_code in SPECIAL_CURRENCIES:
    # Enhanced interpolation for limited data
    exchange_rates = apply_enhanced_interpolation(exchange_rates)
```

**Method**: Creates full time series using available data points and applies nearest neighbor extrapolation for missing periods.

### 🇰🇷 South Korea (KOR) - Exchange Rate Interpolation

**Issue**: Similar to Israel, limited historical Korean Won (KRW) exchange rate data.

**Location**: Same as Israel - `src/macro_abm_calibration/processors/currency.py:234-281`

**Processing**: Identical to Israel's handling with enhanced interpolation for data gaps.

**Historical Context**: Both Israel and South Korea have limited representation in early Eurostat exchange rate databases due to their later integration into international statistical reporting systems.

### 🇪🇺 Eurozone Countries - EUR Currency Logic

**Countries Affected**: 19 Eurozone members
- Austria (AUT), Belgium (BEL), Estonia (EST), Finland (FIN), France (FRA)
- Germany (DEU), Greece (GRC), Ireland (IRL), Italy (ITA), Latvia (LVA)
- Lithuania (LTU), Luxembourg (LUX), Netherlands (NLD), Portugal (PRT)
- Slovakia (SVK), Slovenia (SVN), Spain (ESP)

**Issue**: No separate currency conversion needed since EUR is the base currency for Eurostat data.

**Location**: `src/macro_abm_calibration/processors/currency.py:195-201`

**Processing**:
```python
# Eurozone countries - direct USD conversion
if country_currency == "EUR":
    # USD rate = 1 / (USD to EUR rate)
    usd_rate = 1.0 / eur_to_usd_rate
    return usd_rate
```

**Method**: Direct calculation using inverse of USD/EUR rate rather than database lookup.

**Efficiency**: Avoids unnecessary database queries for countries that use EUR as their base currency.

## Currency Code Mappings

All OECD countries require mapping from country codes to currency codes for exchange rate processing.

**Location**: `src/macro_abm_calibration/data_sources/eurostat.py:39-47`

```python
CURRENCY_MAPPINGS = {
    # North America
    "USA": "USD", "CAN": "CAD", "MEX": "MXN",
    
    # Europe (Non-EUR)
    "CZE": "CZK", "DNK": "DKK", "HUN": "HUF", "NOR": "NOK",
    "POL": "PLN", "SWE": "SEK", "GBR": "GBP",
    
    # Europe (EUR)
    "AUT": "EUR", "BEL": "EUR", "EST": "EUR", "FIN": "EUR",
    "FRA": "EUR", "DEU": "EUR", "GRC": "EUR", "IRL": "EUR",
    "ITA": "EUR", "LVA": "EUR", "LTU": "EUR", "LUX": "EUR",
    "NLD": "EUR", "PRT": "EUR", "SVK": "EUR", "SVN": "EUR",
    "ESP": "EUR",
    
    # Asia-Pacific
    "AUS": "AUD", "JPN": "JPY", "KOR": "KRW", "NZL": "NZD"
}
```

## Implementation Details

### Configuration System

Country-specific adjustments are managed through the configuration system:

```python
# config.yaml
processing:
  country_adjustments:
    MEX:
      price_base_override: "Q"
      exchange_rate_interpolation: true
    USA:
      government_consumption_method: "backward_calculation"
    ISR:
      exchange_rate_interpolation: true
    KOR:
      exchange_rate_interpolation: true
```

### Code Organization

**Base Processing**: `src/macro_abm_calibration/processors/base.py`
- Defines standard processing methods
- Provides hooks for country-specific overrides

**Country Adjustments**: `src/macro_abm_calibration/processors/utils.py:280-325`
- Centralizes country-specific adjustment logic
- Maintains COUNTRY_ADJUSTMENTS dictionary

**Validation**: `src/macro_abm_calibration/processors/validation.py`
- Includes country-specific validation rules
- Ensures adjustments don't compromise data quality

### Testing

Country-specific processing is thoroughly tested:

```python
# tests/unit/test_processors.py
def test_mexico_price_base_adjustment():
    """Test Mexico's special price base handling."""
    processor = HarmonizerProcessor(config)
    result = processor.process_country_data(Country.MEX, raw_data)
    
    assert result.gdp_params["price_base"] == "Q"
    assert result.data_quality.adjustments_applied["price_base"] == True

def test_usa_government_consumption():
    """Test USA's government consumption reconstruction."""
    processor = HarmonizerProcessor(config)
    result = processor.process_country_data(Country.USA, raw_data)
    
    assert result.processing_method["government_consumption"] == "backward_calculation"
```

## Historical Context

### MATLAB Heritage

These special processing requirements were identified and implemented in the original MATLAB codebase. The Python implementation preserves these adjustments to ensure:

1. **Compatibility** with existing research workflows
2. **Consistency** with published academic results
3. **Accuracy** of cross-country economic comparisons

### Evolution of Adjustments

**Phase 1 (MATLAB)**: Ad-hoc code modifications scattered throughout scripts
**Phase 2 (Python)**: Centralized, configurable country adjustment system
**Phase 3 (Future)**: Automated detection of required adjustments based on data quality metrics

## Data Quality Impact

### Before Adjustments
- Mexico: GDP data inconsistencies due to price base mismatches
- USA: Government consumption time series breaks
- Israel/Korea: Missing exchange rate data gaps
- Eurozone: Unnecessary currency conversion overhead

### After Adjustments
- ✅ Consistent GDP price base methodology across countries
- ✅ Smooth government consumption time series for USA
- ✅ Complete exchange rate coverage for all countries
- ✅ Optimized processing for Eurozone countries

## Validation and Quality Assurance

### Automated Checks

Each country-specific adjustment includes validation:

```python
def validate_country_adjustments(country: Country, processed_data: ProcessedData) -> ValidationResult:
    """Validate that country-specific adjustments were applied correctly."""
    
    if country == Country.MEX:
        # Validate price base adjustment
        assert processed_data.gdp_params["price_base"] == "Q"
        
    if country == Country.USA:
        # Validate government consumption continuity
        assert check_time_series_continuity(processed_data.government_consumption)
        
    # Additional country-specific validations...
```

### Cross-Validation

Country-specific adjustments are cross-validated against:
- Historical economic data from national statistical offices
- International databases (World Bank, IMF)
- Academic literature on country-specific data issues

## Future Considerations

### Potential Additional Countries

Countries that may require future special handling:
- **Turkey (TUR)**: Currency volatility and methodology changes
- **Chile (CHL)**: Different industry classification periods
- **Colombia (COL)**: Limited historical OECD data coverage

### Automation Opportunities

Future enhancements could include:
- **Automatic Detection**: ML-based identification of data quality issues
- **Dynamic Adjustments**: Context-aware processing based on data characteristics
- **Validation Frameworks**: Automated testing against multiple data sources

## Usage Guidelines

### For Researchers

When using country-specific data:

1. **Review Adjustments**: Understand what adjustments were applied
2. **Document Usage**: Reference specific processing in publications
3. **Validate Results**: Cross-check against alternative data sources
4. **Report Limitations**: Acknowledge processing adjustments in methodology

### For Developers

When adding new countries:

1. **Assess Data Quality**: Identify potential special processing needs
2. **Implement Centrally**: Add adjustments to `COUNTRY_ADJUSTMENTS` dictionary
3. **Test Thoroughly**: Include country-specific tests
4. **Document Changes**: Update this documentation

## Summary Table

| Country | Primary Issue | Processing Type | Impact | Validation Method |
|---------|---------------|-----------------|---------|-------------------|
| **Mexico (MEX)** | Price base + Exchange rates | Price adjustment + Interpolation | High | Economic consistency checks |
| **USA** | Government data breaks | Time series reconstruction | Medium | Continuity validation |
| **Israel (ISR)** | Exchange rate gaps | Interpolation | Low | Data coverage checks |
| **South Korea (KOR)** | Exchange rate gaps | Interpolation | Low | Data coverage checks |
| **Eurozone (19)** | Currency efficiency | Direct calculation | Performance | Rate accuracy validation |

## Related Documentation

- [Processing Pipeline](processing.md) - Overall data processing workflow
- [Data Sources](data-sources.md) - Raw data collection and quality
- [Configuration](../getting-started/configuration.md) - Setting up country adjustments
- [Validation](validation.md) - Quality assurance framework