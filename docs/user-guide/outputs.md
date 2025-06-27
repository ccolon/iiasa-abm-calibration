# Calibration Outputs

The calibration system generates comprehensive outputs in multiple formats to support different workflows and analysis needs. This page documents all output types, formats, and data structures.

## Overview

The system produces three main categories of outputs:

1. **Calibrated Parameters** - Estimated model parameters for ABM initialization
2. **Initial Conditions** - Agent populations and balance sheet configurations  
3. **Validation Results** - Quality assurance and compliance verification

All outputs support multiple export formats and maintain MATLAB compatibility.

## Export Formats

### Supported Formats

| Format | Extension | Use Case | Features |
|--------|-----------|----------|----------|
| MATLAB | `.mat` | MATLAB compatibility | Original workflow support |
| JSON | `.json` | Cross-platform, human-readable | API integration, debugging |
| Excel | `.xlsx` | Business analysis | Multi-sheet, pivot tables |
| CSV | `.csv` | Data analysis tools | Flat file, tabular data |
| Pickle | `.pkl` | Python-native | Full data type preservation |
| Markdown | `.md` | Reports | Validation summaries |
| PNG | `.png` | Visualizations | Charts and plots |

### Export API

```python
from macro_abm_calibration.calibrators import ParameterCalibrator

calibrator = ParameterCalibrator(config)
results = calibrator.calibrate(data)

# Export specific formats
results.export_matlab("parameters.mat")
results.export_json("parameters.json")
results.export_excel("parameters.xlsx")

# Export all formats
results.export_all(formats=["matlab", "json", "excel", "csv"])
```

## Output Categories

### 1. Calibrated Parameters

**Purpose**: Estimated behavioral parameters for ABM agents

**File Naming**: `calibrated_parameters_{YYYYMMDD_HHMMSS}.{ext}`

#### Data Structure

=== "JSON Format"
    ```json
    {
      "USA": {
        "taylor_rule": {
          "taylor_inflation_response": 1.5,
          "taylor_output_response": 0.5,
          "taylor_smoothing": 0.8,
          "taylor_neutral_rate": 2.0
        },
        "firm_parameters": {
          "price_adjustment_speed": 0.3,
          "investment_sensitivity": 1.0,
          "markup_elasticity": 2.0
        },
        "household_parameters": {
          "marginal_propensity_consume": 0.7,
          "wealth_effect": 0.05,
          "labor_supply_elasticity": 0.5
        }
      },
      "DEU": {
        // ... similar structure for Germany
      }
    }
    ```

=== "Excel Structure"
    **Sheets**:
    - `Summary` - All parameters overview
    - `Taylor_Rule` - Monetary policy parameters
    - `Firm_Parameters` - Firm behavioral parameters  
    - `Household_Parameters` - Consumer behavioral parameters
    - `Pivot_Tables` - Cross-country comparisons

=== "MATLAB Structure"
    ```matlab
    % Organized by country, then parameter category
    parameters.USA.taylor_rule.taylor_inflation_response
    parameters.USA.firm_parameters.price_adjustment_speed
    parameters.USA.household_parameters.marginal_propensity_consume
    ```

#### Parameter Categories

**Taylor Rule Parameters**
- `taylor_inflation_response` - Central bank response to inflation (typically > 1.0)
- `taylor_output_response` - Central bank response to output gap
- `taylor_smoothing` - Interest rate smoothing parameter
- `taylor_neutral_rate` - Long-run neutral interest rate

**Firm Parameters**
- `price_adjustment_speed` - Speed of price adjustments (0-1)
- `investment_sensitivity` - Investment response to demand
- `markup_elasticity` - Price markup elasticity

**Household Parameters**
- `marginal_propensity_consume` - Consumption out of income
- `wealth_effect` - Consumption response to wealth changes
- `labor_supply_elasticity` - Labor supply responsiveness

### 2. Initial Conditions

**Purpose**: Agent population and balance sheet initialization

**File Naming**: `initial_conditions_{YYYYMMDD_HHMMSS}.{ext}`

#### Data Structure

=== "JSON Format"
    ```json
    {
      "USA": {
        "agent_populations": {
          "household_population": {
            "count": 10000,
            "distribution_params": {
              "wealth_distribution": "pareto",
              "wealth_alpha": 1.16
            }
          },
          "firm_population": {
            "count": 1000,
            "size_distribution": "pareto", 
            "size_alpha": 1.06
          }
        },
        "balance_sheets": {
          "households": {
            "count": 10000,
            "total_wealth": 87500000000000,
            "wealth_distribution": [/* array of individual wealth values */],
            "income_distribution": [/* array of individual income values */]
          },
          "firms": {
            "count": 1000,
            "total_assets": 50000000000000,
            "size_distribution": [/* array of firm sizes */]
          }
        },
        "market_conditions": {
          "price_level": 1.0,
          "wage_level": 1.0,
          "interest_rate": 0.02,
          "unemployment_rate": 0.05
        }
      }
    }
    ```

#### Key Components

**Agent Populations**
- **Household Count**: Typically 10,000+ agents per country
- **Firm Count**: Typically 1,000+ agents per country
- **Distribution Parameters**: Pareto distribution parameters for realistic heterogeneity

**Balance Sheets**
- **Wealth Distributions**: Individual agent wealth levels
- **Income Distributions**: Individual agent income levels  
- **Size Distributions**: Firm size heterogeneity
- **Aggregate Totals**: Economy-wide totals for validation

**Market Conditions**
- **Price Levels**: Initial price indices
- **Wage Levels**: Initial wage levels
- **Interest Rates**: Initial policy rates
- **Unemployment**: Initial unemployment rates

### 3. Validation Results

**Purpose**: Quality assurance and model compliance verification

**File Naming**: `validation_results_{YYYYMMDD_HHMMSS}.{ext}`

#### Data Structure

=== "JSON Format"
    ```json
    {
      "summary": {
        "total_checks": 25,
        "passed_checks": 22,
        "failed_checks": 3,
        "warnings": 2,
        "errors": 1,
        "critical_issues": 0,
        "overall_passed": true
      },
      "checks": [
        {
          "name": "taylor_principle_USA",
          "level": "error",
          "passed": true,
          "message": "Taylor principle: inflation response (1.50) should be > 1.0",
          "details": {
            "country": "USA",
            "parameter": "taylor_inflation_response",
            "value": 1.5,
            "threshold": 1.0
          }
        },
        {
          "name": "wealth_distribution_DEU",
          "level": "warning", 
          "passed": false,
          "message": "Wealth inequality (Gini=0.85) exceeds typical range",
          "details": {
            "country": "DEU",
            "gini_coefficient": 0.85,
            "expected_range": [0.6, 0.8]
          }
        }
      ]
    }
    ```

=== "Markdown Report"
    ```markdown
    # Calibration Validation Report
    
    **Generated**: 2024-12-25 10:30:00
    **Overall Status**: ✅ PASSED (22/25 checks)
    
    ## Summary
    - Total Checks: 25
    - Passed: 22 ✅
    - Warnings: 2 ⚠️  
    - Errors: 1 ❌
    - Critical: 0 🔴
    
    ## Failed Checks
    
    ### ⚠️ wealth_distribution_DEU
    **Issue**: Wealth inequality exceeds typical range
    **Details**: Gini coefficient (0.85) above expected range [0.6, 0.8]
    **Recommendation**: Review wealth initialization parameters
    
    ## Country-Specific Results
    
    ### USA (8/8 passed)
    - ✅ Taylor principle compliance
    - ✅ Parameter bounds validation
    - ✅ Balance sheet consistency
    
    ### DEU (7/8 passed)  
    - ✅ Taylor principle compliance
    - ⚠️ Wealth distribution warning
    - ✅ Parameter bounds validation
    ```

#### Validation Categories

**Economic Theory Compliance**
- Taylor principle (inflation response > 1.0)
- Parameter economic bounds
- Behavioral consistency checks

**Data Quality Checks**
- Missing value detection
- Outlier identification
- Distribution shape validation

**Cross-Validation**
- MATLAB compatibility verification
- Historical data consistency
- Inter-country comparisons

## Visualization Outputs

### Parameter Visualizations

**File**: `parameter_comparison_{timestamp}.png`

- Bar charts comparing parameters across countries
- Separate panels for different parameter categories
- Distribution histograms for parameter uncertainty

### Validation Visualizations

**File**: `validation_results_{timestamp}.png`

- Pie charts showing pass/fail ratios
- Issue severity breakdowns by country
- Category-wise validation summaries

### Comprehensive Reports

**File**: `calibration_report_{timestamp}.png`

- Multi-panel dashboard combining all results
- Professional layout with metadata
- Publication-ready formatting

## File Organization

### Directory Structure

```
calibration_results/
├── parameters/
│   ├── calibrated_parameters_20241225_103000.mat
│   ├── calibrated_parameters_20241225_103000.xlsx
│   ├── calibrated_parameters_20241225_103000.json
│   └── calibrated_parameters_20241225_103000.csv
├── initial_conditions/
│   ├── initial_conditions_20241225_103000.json
│   ├── initial_conditions_20241225_103000.pkl
│   └── initial_conditions_20241225_103000.mat
├── validation/
│   ├── validation_results_20241225_103000.json
│   ├── validation_results_20241225_103000.xlsx
│   └── validation_report_20241225_103000.md
├── visualizations/
│   ├── parameter_comparison_20241225_103000.png
│   ├── validation_results_20241225_103000.png
│   └── calibration_report_20241225_103000.png
└── complete_results/
    └── complete_calibration_results_20241225_103000.json
```

### Naming Conventions

**Timestamp Format**: `YYYYMMDD_HHMMSS`
- Year (4 digits) + Month (2 digits) + Day (2 digits)
- Hour (24h format, 2 digits) + Minute (2 digits) + Second (2 digits)

**File Prefixes**:
- `calibrated_parameters_` - Parameter estimation results
- `initial_conditions_` - Agent initialization data
- `validation_results_` - Quality assurance results
- `complete_calibration_results_` - Combined output file

## MATLAB Compatibility

### Data Structure Mapping

The system ensures seamless MATLAB integration:

**Parameter Organization**
```matlab
% Hierarchical structure: country -> category -> parameter
params = load('calibrated_parameters_timestamp.mat');
usa_taylor_response = params.USA.taylor_rule.taylor_inflation_response;
```

**Array Conversion**
- NumPy arrays → MATLAB double arrays
- Python lists → MATLAB cell arrays
- Pandas DataFrames → MATLAB tables/structures

**Variable Naming**
- Follows original MATLAB conventions
- Underscore notation for multi-word variables
- Country codes as structure field names

### Loading in MATLAB

```matlab
% Load parameter file
param_data = load('calibrated_parameters_20241225_103000.mat');

% Access country-specific parameters
usa_params = param_data.USA;
taylor_response = usa_params.taylor_rule.taylor_inflation_response;

% Load initial conditions
init_data = load('initial_conditions_20241225_103000.mat');
household_wealth = init_data.USA.balance_sheets.households.wealth_distribution;
```

## Configuration Options

### Export Settings

```python
from macro_abm_calibration.config import CalibrationConfig

config = CalibrationConfig()

# Configure output settings
config.output.base_path = "./results"
config.output.formats = ["matlab", "json", "excel"]
config.output.compression = True
config.output.include_metadata = True
config.output.validation_level = "strict"
```

### Custom Export Paths

```python
# Custom file naming
results.export_matlab(
    filepath="custom_params.mat",
    include_timestamp=False
)

# Custom directory structure
results.export_all(
    base_path="./country_specific/USA/",
    formats=["json", "excel"]
)
```

## Integration Examples

### Python Workflow

```python
from macro_abm_calibration import CalibrationPipeline

# Run full calibration
pipeline = CalibrationPipeline(config)
results = pipeline.run(countries=["USA", "DEU", "FRA"])

# Export comprehensive results
results.export_all()

# Access specific outputs
params = results.parameters
conditions = results.initial_conditions
validation = results.validation
```

### MATLAB Integration

```matlab
% Load Python-generated calibration
calibration = load('complete_calibration_results.mat');

% Initialize ABM with calibrated parameters
abm = initializeABM(calibration.USA.parameters);
abm = setInitialConditions(abm, calibration.USA.initial_conditions);

% Run simulation
results = runSimulation(abm, time_periods=1000);
```

## Best Practices

### File Management
- Use timestamp-based naming for version control
- Organize outputs by calibration run or experiment
- Archive old results before new calibration runs
- Maintain separate directories for different model versions

### Quality Assurance
- Always review validation results before using parameters
- Cross-validate outputs against historical data
- Compare results across different calibration runs
- Document any manual adjustments or overrides

### Performance Considerations
- Large agent populations generate substantial output files
- Consider compression for archival storage
- Use appropriate precision for numerical outputs
- Balance detail level with file size constraints

## Next Steps

- [Processing Pipeline](processing.md) - Understanding data flow to outputs
- [Model Calibration](calibration.md) - How outputs are generated
- [Examples](../examples/complete-workflow.md) - Complete workflow examples