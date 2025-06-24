# Macroeconomic Agent-Based Model Calibration

A Python package for calibrating agent-based macroeconomic models using OECD data sources, including national accounts, input-output tables, and financial statistics.

## Overview

This project converts and extends MATLAB code for processing economic data from multiple sources to initialize agent-based macroeconomic models. The system handles:

- **OECD National Accounts**: GDP, consumption, investment, trade data
- **Input-Output Tables**: Inter-industry flows and bilateral trade
- **Financial Statistics**: Interest rates, exchange rates, debt levels
- **Labor Statistics**: Employment and unemployment data

## Features

- 🌍 **Multi-Country Support**: 31 OECD countries + Rest of World
- 🏭 **Industry Detail**: 18 NACE2 sectors with full I-O linkages  
- 📊 **Data Integration**: Seamless combination of quarterly and annual data
- 💱 **Currency Conversion**: Automatic conversion to USD using historical rates
- ✅ **Data Validation**: Comprehensive quality checks and consistency validation
- 🔧 **Flexible Configuration**: Easily customize countries, industries, and time periods
- 📈 **Performance**: Optimized processing with parallel computation support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/username/macro-abm-calibration.git
cd macro-abm-calibration

# Install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from macro_abm_calibration import CalibrationConfig, CalibrationPipeline

# Create configuration
config = CalibrationConfig(
    countries=["USA", "DEU", "JPN"],
    industries=["A", "C", "G", "K"],  # Agriculture, Manufacturing, Trade, Finance
    estimation_period=TimeFrame(start_year=2000, end_year=2020),
    calibration_period=TimeFrame(start_year=2015, end_year=2017)
)

# Run calibration pipeline
pipeline = CalibrationPipeline(config)
results = pipeline.run_full_pipeline()

# Access calibrated data
gdp_data = results.gdp_by_country
io_tables = results.input_output_tables
model_params = results.model_parameters
```

### Configuration

The system supports flexible configuration through YAML files:

```yaml
# config.yaml
countries: ["USA", "DEU", "FRA", "JPN"]
industries: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]

estimation_period:
  start_year: 1996
  end_year: 2024
  frequency: "Q"

calibration_period:
  start_year: 2010
  end_year: 2017
  frequency: "A"

database:
  sqlite_path: "./data/OECD_Data.sqlite"

processing:
  base_currency: "USD"
  missing_data_strategy: "interpolate"
  parallel_processing: true
```

Load configuration:

```python
config = CalibrationConfig.from_file("config.yaml")
```

## Data Requirements

### Required Data Sources

1. **OECD SQLite Database** (`OECD_Data.sqlite`)
   - Download from OECD.Stat or construct from CSV files
   - Contains national accounts, financial markets, and labor force data

2. **OECD Input-Output Tables** (`oecd_ICIOs_SML_double.mat`)
   - Download from [OECD ICIO Database](https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm)
   - Provides industry-level production and trade linkages

3. **Eurostat Exchange Rates** (included in OECD database)
   - Bilateral exchange rates to EUR and USD
   - Quarterly and annual frequencies

### Directory Structure

```
data/
├── Database_OECD_Eurostat/
│   └── OECD_CSV_SQL_Files/
│       └── OECD_Data.sqlite
├── ICIOS_OECD_1995-2020_SML/
│   └── oecd_ICIOs_SML_double.mat
└── processed/
    ├── country_data/
    ├── industry_data/
    └── calibration_results/
```

## Architecture

### Core Components

- **Data Sources** (`data_sources/`): Connectors for OECD, Eurostat, and ICIO data
- **Processors** (`processors/`): Data transformation and aggregation modules
- **Calibrators** (`calibrators/`): Model parameter estimation and initialization
- **Utils** (`utils/`): Logging, validation, and helper functions

### Processing Pipeline

1. **Data Extraction**: Load data from SQLite and MATLAB files
2. **Currency Conversion**: Convert all series to USD using historical rates
3. **Industry Aggregation**: Map ISIC Rev4 to NACE2 classification
4. **Data Validation**: Check consistency and completeness
5. **Model Calibration**: Estimate parameters and set initial conditions

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=macro_abm_calibration --cov-report=html
```

### Code Quality

The project uses:
- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

## Configuration Reference

### Countries

Supported OECD countries (31 total):
- **Americas**: USA, CAN, MEX, CHL, COL, CRI
- **Europe**: DEU, FRA, GBR, ITA, ESP, NLD, BEL, AUT, CHE, SWE, DNK, NOR, FIN, IRL, GRC, PRT, POL, CZE, SVK, SVN, HUN, EST, LVA, LTU, ISL, TUR
- **Asia-Pacific**: JPN, KOR, AUS, NZL, ISR

### Industries (NACE2)

- **A**: Agriculture, forestry and fishing
- **B**: Mining and quarrying  
- **C**: Manufacturing
- **D**: Electricity, gas, steam and air conditioning supply
- **E**: Water supply; sewerage, waste management
- **F**: Construction
- **G**: Wholesale and retail trade
- **H**: Transportation and storage
- **I**: Accommodation and food service activities
- **J**: Information and communication
- **K**: Financial and insurance activities
- **L**: Real estate activities
- **M**: Professional, scientific and technical activities
- **N**: Administrative and support service activities
- **O**: Public administration and defence
- **P**: Education
- **Q**: Human health and social work activities
- **R**: Arts, entertainment and recreation
- **S**: Other service activities

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this software in your research, please cite:

```bibtex
@software{macro_abm_calibration,
  title={Macroeconomic Agent-Based Model Calibration},
  author={Research Team},
  year={2024},
  url={https://github.com/username/macro-abm-calibration}
}
```

## Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues](https://github.com/username/macro-abm-calibration/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/macro-abm-calibration/discussions)

## Roadmap

- [ ] Add support for additional data sources (IMF, World Bank)
- [ ] Implement real-time data updates
- [ ] Add web dashboard for monitoring calibration results
- [ ] Extend to non-OECD countries
- [ ] Integration with popular ABM frameworks