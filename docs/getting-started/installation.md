# Installation

## Requirements

- Python 3.9 or higher
- Required data files (see Data Requirements section)

## Install from Source

```bash
# Clone the repository
git clone https://github.com/username/macro-abm-calibration.git
cd macro-abm-calibration

# Install in development mode
pip install -e ".[dev]"

# Install with documentation dependencies
pip install -e ".[docs]"

# Install all optional dependencies
pip install -e ".[dev,docs,performance]"
```

## Data Requirements

The package requires specific data files to function:

### 1. OECD Database
- **File**: `./Dataset/Database_OECD_Eurostat/OECD_CSV_SQL_Files/OECD_Data.sqlite`
- **Description**: SQLite database containing OECD and Eurostat data
- **Source**: Processed OECD CSV files

### 2. ICIO Data
- **File**: `./Dataset/ICIOS_OECD_1995-2020_SML/oecd_ICIOs_SML_double.mat`
- **Description**: MATLAB file with input-output tables
- **Source**: OECD Inter-Country Input-Output (ICIO) tables

### 3. Eurostat Exchange Rates
- **Tables**: Included in OECD database
  - `estat_ert_bil_eur_q_filtered_en` (quarterly rates)
  - `estat_ert_bil_eur_a_filtered_en` (annual rates)

## Verification

Verify your installation:

```python
from macro_abm_calibration import CalibrationConfig, DataSourceManager

# Test configuration loading
config = CalibrationConfig()
print(f"Configuration loaded: {config.database.oecd_path}")

# Test data source manager
manager = DataSourceManager(config)
print("Installation successful!")
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration Setup](configuration.md)