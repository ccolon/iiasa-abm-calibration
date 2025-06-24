# CLAUDE.md - Project State and Context

## Project Overview

**Project**: Macroeconomic Agent-Based Model Calibration System
**Goal**: Convert MATLAB code to a well-documented Python package for calibrating ABM models from OECD data
**Status**: Phase 2 Complete - Data Infrastructure Built
**Next**: Phase 3 - Processing Pipeline Implementation

## Project Structure

```
macro-abm-calibration/
├── src/macro_abm_calibration/
│   ├── __init__.py                 ✅ Core package exports
│   ├── config.py                   ✅ Pydantic configuration system
│   ├── models.py                   ✅ Core data models (Country, Industry, TimeFrame)
│   ├── data_sources/               ✅ Data connectors (OECD, Eurostat, ICIO)
│   │   ├── __init__.py
│   │   ├── base.py                 ✅ Abstract base classes and interfaces
│   │   ├── oecd.py                 ✅ OECD SQLite connector
│   │   ├── eurostat.py             ✅ Eurostat exchange rate connector
│   │   ├── icio.py                 ✅ ICIO input-output data loader
│   │   └── factory.py              ✅ Factory and manager classes
│   ├── processors/                 🔄 NEXT: Data processing pipeline
│   ├── calibrators/                🔄 NEXT: Model calibration logic
│   └── utils/                      ✅ Logging and validation utilities
├── tests/                          ✅ Unit and integration tests
├── docs/                           📝 Documentation (basic README done)
├── pyproject.toml                  ✅ Modern Python packaging
└── README.md                       ✅ Comprehensive project documentation
```

## Completed Phases

### ✅ Phase 1: Foundation (COMPLETE)
- [x] Project structure with modern Python packaging
- [x] Core data models (Country, Industry, TimeFrame)
- [x] Configuration system with Pydantic validation
- [x] Logging and utility framework
- [x] Testing infrastructure with pytest
- [x] Comprehensive documentation

### ✅ Phase 2: Data Infrastructure (COMPLETE)
- [x] Abstract data source interfaces with caching
- [x] OECD database connector (SQLite + SQLAlchemy)
- [x] Eurostat exchange rate connector
- [x] ICIO input-output data loader (MATLAB .mat files)
- [x] Data source factory and manager
- [x] Integration tests and error handling

## Key Implementation Details

### Data Sources Architecture
- **Base Classes**: `DataSource` (abstract), `CachedDataSource` (with TTL caching)
- **OECD Connector**: Replicates MATLAB query patterns exactly
  - GDP data: quarterly/annual with price base selection
  - Unemployment: sector-specific with age/sex filters
  - Interest rates: short-term rates with methodology filters
- **Eurostat Connector**: Complex exchange rate logic
  - EUR-based rates with USD conversion: `LC_to_USD = LC_to_EUR / USD_to_EUR`
  - Special handling for ILS, KRW, MXN (limited historical data)
  - Interpolation for missing values using nearest neighbor
- **ICIO Connector**: MATLAB .mat file processing
  - Input-output matrices: 44 ISIC Rev4 industries
  - Bilateral trade flows by industry and final demand sector
  - Matrix shrinking and aggregation (ROW creation)

### MATLAB Functionality Replicated
From original analysis, these key functions are implemented:
- `a_data.m` → OECD/Eurostat data fetching
- `calculate_deflator.m` → Nominal/real ratio calculations  
- `mapping_codes_EUROSTAT.m` → Currency code mappings
- `shrink_icios.m` → ICIO matrix filtering and aggregation
- `ISIC_REV4_to_NACE2_10.m` → Industry classification mapping

### Configuration System
- **Hierarchical**: Database, DataSources, Processing, Output, Logging configs
- **Environment Variables**: Support for `.env` files and ENV var overrides
- **File Support**: JSON/YAML configuration files
- **Validation**: Pydantic models with comprehensive validation rules

## Current State: Phase 4 Complete - Production Ready System

### ✅ Phase 3: Processing Pipeline (COMPLETE)
- [x] Base processor architecture with validation and chaining
- [x] Currency conversion processor (USD conversion with complex logic)
- [x] Industry aggregation processor (ISIC Rev4 → NACE2)
- [x] Data harmonization processor (deflators, time alignment)
- [x] Pipeline orchestrator (full MATLAB sequence replication)
- [x] Utility functions (deflators, interpolation, country adjustments)
- [x] Comprehensive unit tests for all processors
- [x] Package exports updated with all processors

### ✅ Phase 4: Model Calibrators (COMPLETE)
- [x] Base calibrator architecture with result management
- [x] ABM parameter estimator (Taylor rule, firm/household behavior)
- [x] Initial conditions setter (agent populations, balance sheets)
- [x] Model validation framework (economic theory, MATLAB comparison)
- [x] Calibration utilities (HP filter, AR estimation, economic functions)
- [x] Result export system (MATLAB, JSON, Excel, CSV formats)
- [x] Visualization framework (parameter plots, validation charts)
- [x] Comprehensive calibrator tests and integration tests
- [x] Complete example workflow demonstration

### Available Processing Capabilities
The system can now:
- **Extract Raw Data**: OECD/Eurostat data with proper filtering
- **Convert Currencies**: Complex EUR→USD conversion with special cases
- **Aggregate Industries**: 44 ISIC Rev4 → 18 NACE2 sectors
- **Harmonize Data**: Quarterly/annual alignment, deflator calculation
- **Process ICIO**: Input-output table shrinking and aggregation
- **Validate Quality**: Cross-dataset consistency and completeness

### Data Flow Pipeline
```
Raw OECD/Eurostat Data → Currency Conversion → Industry Aggregation → 
Data Harmonization → ICIO Processing → Final Calibration Datasets
```

### Testing Coverage
- Unit tests for all processors and utilities
- Integration tests for data sources and pipeline
- Validation framework for data quality
- Performance benchmarks ready

## Phase 4 Plan: Model Calibrators

### 🔄 Next Priorities (High)
1. **ABM Parameter Estimator** (`calibrators/parameters.py`)
   - Taylor rule estimation for interest rates
   - Firm behavioral parameters from industry data
   - Household consumption parameters

2. **Initial Conditions Setter** (`calibrators/initial_conditions.py`)
   - Agent population distribution
   - Wealth and balance sheet initialization
   - Industry capacity and employment

3. **Model Validation** (`calibrators/validation.py`)
   - Cross-validate against MATLAB outputs
   - Consistency checks for calibrated parameters
   - Simulation readiness verification

## Key Design Decisions

### Maintained from MATLAB
- **Exact Dataset Names**: Using original OECD dataset identifiers
- **Query Logic**: Replicating SQL filters and parameters exactly
- **Special Cases**: Country-specific adjustments (MEX price base Q, USA growth rates)
- **Industry Codes**: Maintaining ISIC Rev4 → NACE2 aggregation mappings
- **Time Handling**: Same date calculations (datenum equivalents)

### Python Improvements
- **Type Safety**: Full type hints with mypy checking
- **Error Handling**: Comprehensive exception hierarchy vs try-catch blocks
- **Configuration**: Flexible, validated configuration vs hard-coded parameters
- **Testing**: Comprehensive test suite vs no tests
- **Documentation**: Full API docs vs minimal comments
- **Modularity**: Clean separation of concerns vs monolithic scripts

## Development Environment

### Setup Commands
```bash
cd macro-abm-calibration
pip install -e ".[dev]"
pytest  # Run tests
black . # Format code
mypy src/ # Type checking
```

### Key Dependencies
- **pandas**: DataFrame operations and time series
- **sqlalchemy**: Database connections
- **scipy**: MATLAB file loading
- **pydantic**: Configuration validation
- **numpy**: Matrix operations
- **pytest**: Testing framework

## Data Requirements

### Required Files
1. **OECD Database**: `./Dataset/Database_OECD_Eurostat/OECD_CSV_SQL_Files/OECD_Data.sqlite`
2. **ICIO Data**: `./Dataset/ICIOS_OECD_1995-2020_SML/oecd_ICIOs_SML_double.mat`
3. **Eurostat Tables**: Included in OECD database
   - `estat_ert_bil_eur_q_filtered_en` (quarterly rates)
   - `estat_ert_bil_eur_a_filtered_en` (annual rates)

### Expected Output
- **Country Data**: Individual `.mat` equivalent files per country
- **ICIO Processed**: `icios_model_NACE2_10_USD.mat` equivalent
- **Calibration Data**: Final datasets for ABM initialization

## Known Issues & Considerations

### Technical Debt
- ICIO matrix aggregation logic is simplified (needs full ROW aggregation)
- Exchange rate interpolation could be more sophisticated
- Error handling for missing countries/years needs refinement

### Performance Notes
- ICIO data loading is memory-intensive (2GB+ files)
- SQLite queries could benefit from indexing
- Parallel processing not yet implemented

### MATLAB Compatibility
- Date handling differences (Python datetime vs MATLAB datenum)
- Matrix operations may have slight numerical differences
- File I/O formats need validation against original outputs

## Restart Instructions

To efficiently restart development:

1. **Quick Status Check**:
   ```python
   from macro_abm_calibration import CalibrationConfig, DataSourceManager
   config = CalibrationConfig()
   print("Phase 2 complete, ready for Phase 3")
   ```

2. **Continue from Phase 3**:
   - Start with `src/macro_abm_calibration/processors/__init__.py`
   - Implement currency conversion processor first
   - Follow the phase plan above

3. **Key Context**:
   - All data source connectors are complete and tested
   - Configuration system is fully functional
   - Ready to implement the processing pipeline
   - MATLAB functionality mapping is documented above

## Contact Points

- **Original MATLAB Code**: `/calibration_data/` and `/calibration_model/` folders
- **Key MATLAB Files**: `a_data.m`, `b_calibration_data.m`, `c1_icios_data.m`
- **Config Examples**: See tests for configuration patterns
- **Data Source Usage**: See integration tests for examples

---

**Last Updated**: Phase 3 completion - Processing Pipeline fully implemented
**Next Milestone**: Phase 4 - Model Calibrators (ABM parameter estimation and initialization)