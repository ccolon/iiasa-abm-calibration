# Quick Start

This guide walks you through a basic calibration workflow.

## Basic Example

```python
from macro_abm_calibration import (
    CalibrationConfig,
    DataSourceManager,
    ProcessingPipeline,
    ParameterCalibrator
)
from macro_abm_calibration.models import Country, TimeFrame

# 1. Setup configuration
config = CalibrationConfig()

# 2. Initialize data sources
data_manager = DataSourceManager(config)

# 3. Define calibration scope
countries = [Country.DEU, Country.FRA, Country.ITA]
timeframe = TimeFrame(start_year=2010, end_year=2020)

# 4. Setup processing pipeline
pipeline = ProcessingPipeline(config)

# 5. Process data for each country
for country in countries:
    print(f"Processing {country.name}...")
    
    # Extract raw data
    raw_data = data_manager.extract_country_data(country, timeframe)
    
    # Process through pipeline
    processed_data = pipeline.process(raw_data)
    
    # Calibrate model parameters
    calibrator = ParameterCalibrator(config)
    results = calibrator.calibrate(processed_data)
    
    # Export results
    results.export_matlab(f"{country.name.lower()}_calibration.mat")
    
    print(f"✅ {country.name} calibration complete")
```

## Configuration Options

### Environment Variables

```bash
# Set database paths
export OECD_DB_PATH="./data/OECD_Data.sqlite"
export ICIO_DATA_PATH="./data/oecd_ICIOs_SML_double.mat"

# Configure processing
export PROCESSING_CURRENCY="USD"
export PROCESSING_INDUSTRY_AGGREGATION="NACE2_10"
```

### Configuration File

Create `config.yaml`:

```yaml
database:
  oecd_path: "./Dataset/Database_OECD_Eurostat/OECD_CSV_SQL_Files/OECD_Data.sqlite"
  icio_path: "./Dataset/ICIOS_OECD_1995-2020_SML/oecd_ICIOs_SML_double.mat"

processing:
  target_currency: "USD"
  industry_aggregation: "NACE2_10"
  
calibration:
  taylor_rule_estimation: true
  hp_filter_lambda: 1600
```

## Data Sources

### OECD Data

```python
from macro_abm_calibration.data_sources import OECDDataSource

oecd = OECDDataSource(config)

# Get GDP data
gdp_data = oecd.get_gdp_data(
    countries=[Country.DEU], 
    timeframe=timeframe,
    price_base="Current"
)
```

### Eurostat Exchange Rates

```python
from macro_abm_calibration.data_sources import EurostatDataSource

eurostat = EurostatDataSource(config)

# Get EUR/USD exchange rates
rates = eurostat.get_exchange_rates(
    base_currency="EUR",
    target_currencies=["USD"],
    timeframe=timeframe
)
```

## Next Steps

- [Configuration Details](configuration.md)
- [Data Sources Guide](../user-guide/data-sources.md)
- [Processing Pipeline](../user-guide/processing.md)
- [Complete Example](../examples/complete-workflow.md)