# Configuration

The package uses Pydantic for configuration management with support for environment variables, configuration files, and programmatic setup.

## Configuration Structure

The main configuration is hierarchical:

```python
from macro_abm_calibration import CalibrationConfig

config = CalibrationConfig()
print(config.model_dump())
```

### Database Configuration

```python
config.database.oecd_path          # Path to OECD SQLite database
config.database.icio_path          # Path to ICIO MATLAB file
config.database.cache_ttl          # Cache time-to-live (seconds)
```

### Data Source Configuration

```python
config.data_sources.oecd.timeout      # Request timeout
config.data_sources.eurostat.timeout  # Request timeout
config.data_sources.icio.preload      # Preload data on init
```

### Processing Configuration

```python
config.processing.target_currency         # Target currency (USD)
config.processing.industry_aggregation    # Industry mapping
config.processing.interpolation_method    # Missing data handling
```

## Environment Variables

All configuration can be overridden with environment variables:

```bash
# Database paths
export OECD_DB_PATH="/path/to/OECD_Data.sqlite"
export ICIO_DATA_PATH="/path/to/oecd_ICIOs_SML_double.mat"

# Processing options
export PROCESSING_TARGET_CURRENCY="USD"
export PROCESSING_INDUSTRY_AGGREGATION="NACE2_10"
export PROCESSING_INTERPOLATION_METHOD="linear"

# Caching
export DATABASE_CACHE_TTL=3600

# Logging
export LOGGING_LEVEL="INFO"
export LOGGING_FORMAT="detailed"
```

## Configuration Files

### YAML Configuration

Create `config.yaml`:

```yaml
database:
  oecd_path: "./data/OECD_Data.sqlite"
  icio_path: "./data/oecd_ICIOs_SML_double.mat"
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

processing:
  target_currency: "USD"
  industry_aggregation: "NACE2_10"
  interpolation_method: "linear"
  
calibration:
  taylor_rule_estimation: true
  hp_filter_lambda: 1600
  ar_estimation_lags: 4

output:
  base_path: "./output"
  formats: ["matlab", "json", "excel"]
  compression: true

logging:
  level: "INFO"
  format: "detailed"
  file_path: "./logs/calibration.log"
```

Load with:

```python
config = CalibrationConfig.from_yaml("config.yaml")
```

### JSON Configuration

```json
{
  "database": {
    "oecd_path": "./data/OECD_Data.sqlite",
    "icio_path": "./data/oecd_ICIOs_SML_double.mat"
  },
  "processing": {
    "target_currency": "USD",
    "industry_aggregation": "NACE2_10"
  }
}
```

Load with:

```python
config = CalibrationConfig.from_json("config.json")
```

## Environment File (.env)

Create `.env` file:

```bash
# Database Configuration
OECD_DB_PATH="./Dataset/Database_OECD_Eurostat/OECD_CSV_SQL_Files/OECD_Data.sqlite"
ICIO_DATA_PATH="./Dataset/ICIOS_OECD_1995-2020_SML/oecd_ICIOs_SML_double.mat"

# Processing
PROCESSING_TARGET_CURRENCY="USD"
PROCESSING_INDUSTRY_AGGREGATION="NACE2_10"

# Output
OUTPUT_BASE_PATH="./results"
OUTPUT_FORMATS="matlab,json,excel"

# Logging  
LOGGING_LEVEL="INFO"
LOGGING_FILE_PATH="./logs/calibration.log"
```

The package automatically loads `.env` files.

## Programmatic Configuration

```python
from macro_abm_calibration.config import (
    CalibrationConfig,
    DatabaseConfig,
    ProcessingConfig
)

# Custom database config
db_config = DatabaseConfig(
    oecd_path="/custom/path/OECD_Data.sqlite",
    icio_path="/custom/path/icio_data.mat",
    cache_ttl=7200
)

# Custom processing config
proc_config = ProcessingConfig(
    target_currency="EUR",
    industry_aggregation="ISIC_REV4",
    interpolation_method="cubic"
)

# Combined configuration
config = CalibrationConfig(
    database=db_config,
    processing=proc_config
)
```

## Validation

Configuration is automatically validated:

```python
try:
    config = CalibrationConfig(
        database={"oecd_path": "invalid/path.db"}
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Configuration Priority

Configuration is loaded in this order (highest to lowest priority):

1. **Programmatic** - Direct parameter setting
2. **Environment Variables** - OS environment variables
3. **Configuration Files** - YAML/JSON files
4. **Defaults** - Built-in default values

## Next Steps

- [Data Sources Guide](../user-guide/data-sources.md)
- [Quick Start Example](quickstart.md)