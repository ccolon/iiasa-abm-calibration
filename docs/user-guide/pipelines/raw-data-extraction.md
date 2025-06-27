# Raw Data Extraction Pipeline

## Overview

The Raw Data Extraction Pipeline serves as the entry point for the entire calibration workflow. It extracts macroeconomic data from multiple sources including OECD databases, Eurostat exchange rate tables, and ICIO input-output matrices. This pipeline replicates the functionality of the original `a_data.m` MATLAB script.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/processors/pipeline.py`
- **Method**: `CalibrationPipeline._step_1_extract_raw_data()`
- **Dependencies**: None (entry point)
- **MATLAB Equivalent**: `a_data.m`
- **Phase**: Data Processing

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `config` | `CalibrationConfig` | Configuration System | Yes | Database paths and extraction parameters |
| `countries` | `List[Country]` | User specification | Yes | Target countries for calibration |
| `timeframe` | `TimeFrame` | User specification | Yes | Start and end years for data extraction |

### Input Validation

- Database file existence and accessibility checks
- Country code validation against available data
- Timeframe validation against data coverage
- Configuration parameter completeness verification

### Configuration Parameters

```yaml
database:
  oecd_path: "./Dataset/Database_OECD_Eurostat/OECD_CSV_SQL_Files/OECD_Data.sqlite"
  icio_path: "./Dataset/ICIOS_OECD_1995-2020_SML/oecd_ICIOs_SML_double.mat"
  cache_ttl: 3600

data_sources:
  oecd:
    timeout: 30
    max_retries: 3
  eurostat:
    timeout: 30
    max_retries: 3
  icio:
    preload: true
```

## Data Treatment

### Standard Treatment

The pipeline extracts economic data using standardized queries and parameters applied uniformly across all countries.

#### Processing Steps

1. **OECD Data Extraction**
   - GDP data (quarterly and annual, nominal and real)
   - Private consumption expenditure
   - Government consumption expenditure
   - Gross fixed capital formation (investment)
   - Exports and imports of goods and services
   - Unemployment rates by sector
   - Short-term interest rates

2. **Eurostat Exchange Rate Extraction**
   - Quarterly EUR-based exchange rates
   - Annual EUR-based exchange rates
   - Currency code mappings for all OECD countries

3. **ICIO Matrix Extraction**
   - Input-output tables with 44 ISIC Rev4 industries
   - Bilateral trade flows by industry
   - Final demand matrices

#### Standard Algorithms

```python
def extract_gdp_data(self, countries: List[Country], timeframe: TimeFrame):
    """Standard GDP data extraction for all countries."""
    return self.oecd_source.get_gdp_data(
        countries=countries,
        timeframe=timeframe,
        price_base="L",  # Standard: constant prices
        measure="C"      # Standard: seasonally adjusted
    )
```

### Country-Specific Treatment

The raw data extraction pipeline applies standard queries to all countries without country-specific modifications. Country-specific adjustments are handled in downstream pipelines.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| None | N/A | Standard extraction for all | Raw data extraction is uniform |

#### Implementation Details

Raw data extraction uses identical queries and parameters for all countries. The pipeline maintains consistency by applying the same extraction logic universally, leaving country-specific adjustments to specialized downstream processors.

```python
# Standard extraction applied to all countries
for country in countries:
    country_data = {
        'gdp': self.extract_gdp_data([country], timeframe),
        'consumption': self.extract_consumption_data([country], timeframe),
        'investment': self.extract_investment_data([country], timeframe),
        # ... same pattern for all countries
    }
```

### Error Handling

- **Database Connection Failures**: Retry logic with exponential backoff
- **Missing Data**: Log warnings and continue with available data
- **Query Timeouts**: Configurable timeout with graceful degradation
- **File Access Errors**: Clear error messages with suggested fixes

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `raw_datasets` | `Dict[Country, RawDataset]` | Structured dict | Economic time series per country |
| `exchange_rates` | `Dict[str, pd.DataFrame]` | DataFrame | EUR-based exchange rates by currency |
| `icio_matrices` | `Dict[str, np.ndarray]` | NumPy arrays | Input-output matrices by year |
| `metadata` | `ExtractionMetadata` | Structured object | Extraction timestamps and quality flags |

### Output Validation

- **Data Completeness**: Verify all requested variables extracted
- **Time Coverage**: Ensure data spans requested timeframe
- **Data Types**: Validate numeric data and proper formatting
- **Missing Value Patterns**: Document data gaps by country/variable

### Metadata

Information included with outputs:
- **Extraction Timestamp**: When data was retrieved
- **Data Vintage**: Source database timestamps
- **Coverage Statistics**: Available data percentages by country
- **Quality Flags**: Warnings about missing or questionable data
- **Source Versions**: Database and file versions used

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n×m×t) where n=countries, m=variables, t=time periods
- **I/O Bound**: Primary bottleneck is database query execution
- **Memory Usage**: Linear with data volume, peak during ICIO loading

### Scalability

- **Country Scaling**: Linear increase in processing time
- **Time Period Scaling**: Linear increase in memory and processing
- **Variable Scaling**: Minimal impact due to efficient SQL queries
- **ICIO File Size**: 2GB+ files require substantial memory

## Integration Points

### Upstream Integration

As the entry point pipeline:
- **Configuration Dependency**: Requires valid database paths
- **Data Source Availability**: Needs accessible OECD/Eurostat/ICIO files
- **No Data Dependencies**: Does not consume outputs from other pipelines

### Downstream Integration

Provides foundational data to all subsequent pipelines:
- **Currency Conversion**: Raw economic data + exchange rates
- **Industry Aggregation**: ICIO matrices and country data
- **Data Harmonization**: Complete economic time series
- **Standardized Format**: Consistent data structures for all consumers

## Validation and Testing

### Unit Tests

```python
def test_gdp_data_extraction():
    """Test GDP data extraction completeness and format."""
    
def test_exchange_rate_coverage():
    """Test exchange rate availability for all currencies."""
    
def test_icio_matrix_loading():
    """Test ICIO matrix structure and dimensions."""
```

### Integration Tests

- **End-to-End Extraction**: Complete workflow from config to outputs
- **Data Quality Validation**: Cross-check against known benchmarks
- **MATLAB Comparison**: Verify identical data extraction vs original scripts

## Configuration Examples

### Basic Configuration

```yaml
database:
  oecd_path: "./data/OECD_Data.sqlite"
  icio_path: "./data/oecd_ICIOs_SML_double.mat"

data_sources:
  oecd:
    timeout: 30
  eurostat:
    timeout: 30
  icio:
    preload: false
```

### Advanced Configuration

```yaml
database:
  oecd_path: "./data/OECD_Data.sqlite"
  icio_path: "./data/oecd_ICIOs_SML_double.mat"
  cache_ttl: 7200

data_sources:
  oecd:
    timeout: 60
    max_retries: 5
    chunk_size: 1000
  eurostat:
    timeout: 45
    max_retries: 3
  icio:
    preload: true
    memory_efficient: true
    
extraction:
  parallel_countries: true
  batch_size: 5
  validate_outputs: true
```

### Country-Specific Configuration

```yaml
# No country-specific configuration needed for raw data extraction
# All countries use identical extraction parameters
extraction:
  standard_parameters:
    price_base: "L"
    measure: "C"
    frequency: "quarterly"
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Database connection timeout | `ConnectionError` exceptions | Increase timeout, check network connectivity |
| OECD file not found | `FileNotFoundError` on SQLite path | Verify database file path and permissions |
| ICIO file loading failure | Memory errors or corrupted data | Check available RAM, validate .mat file integrity |
| Missing exchange rate data | Empty DataFrames for currencies | Verify Eurostat tables exist in database |
| Incomplete time series | Sparse data or gaps | Check data availability for requested timeframe |

### Debug Information

- **Logging Output**: Detailed progress and error information
- **Performance Metrics**: Query execution times and memory usage
- **Data Quality Reports**: Missing data patterns and coverage statistics
- **Validation Results**: Data type and completeness verification

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Data Sources](../data-sources.md) - Detailed data source documentation
- [Configuration](../../getting-started/configuration.md) - System configuration options
- [Currency Conversion Pipeline](currency-conversion.md) - Next pipeline in workflow