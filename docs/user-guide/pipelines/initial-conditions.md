# Initial Conditions Setup Pipeline

## Overview

The Initial Conditions Setup Pipeline creates the starting conditions for ABM simulations by initializing agent populations, distributing wealth and assets, and setting market conditions. It uses estimated behavioral parameters and processed economic data to create realistic agent heterogeneity and economic starting states.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/calibrators/initial_conditions.py`
- **Class**: `InitialConditionsSetter`
- **Dependencies**: All Phase 1 pipelines + Parameter Estimation Pipeline
- **MATLAB Equivalent**: Agent initialization scripts in ABM setup
- **Phase**: Model Calibration

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `estimated_parameters` | `Dict[Country, ParameterSet]` | Parameter Estimation | Yes | Behavioral parameters for agent setup |
| `harmonized_data` | `Dict[Country, HarmonizedDataset]` | Data Harmonization | Yes | Economic data for scaling and calibration |
| `industry_data` | `Dict[Country, AggregatedDataset]` | Industry Aggregation | Yes | Sectoral data for industry-specific agents |
| `scaling_config` | `ScalingConfig` | Configuration | Yes | Agent population and scaling parameters |

### Input Validation

- Verify parameter estimates available for all target countries
- Check economic data contains required variables for agent initialization
- Validate industry data completeness for sectoral agent distribution
- Ensure scaling parameters are economically reasonable

### Configuration Parameters

```yaml
initial_conditions:
  household_population: 10000
  firm_population: 1000
  wealth_distribution: "pareto"
  wealth_alpha: 1.16
  
agent_setup:
  heterogeneity_level: "high"
  industry_specialization: true
  regional_variation: false
  
scaling:
  gdp_scaling_factor: 1.0e-9
  population_scaling: 1.0e-6
  preserve_ratios: true
```

## Data Treatment

### Standard Treatment

The pipeline applies consistent agent initialization procedures to create comparable starting conditions across all countries.

#### Processing Steps

1. **Agent Population Setup**
   - Create household agents with heterogeneous characteristics
   - Initialize firm agents with industry-specific properties
   - Set up financial sector and government agents
   - Apply population scaling based on economic size

2. **Wealth Distribution**
   - Generate Pareto-distributed household wealth
   - Calibrate distribution parameters to match aggregate wealth
   - Initialize firm size distribution using economic data
   - Set bank and financial institution asset levels

3. **Balance Sheet Initialization**
   - Distribute assets and liabilities across agents
   - Ensure aggregate consistency with national accounts
   - Initialize industry-specific capital stocks
   - Set up inter-agent financial relationships

4. **Market Conditions Setup**
   - Initialize price levels and wage rates
   - Set employment and unemployment distributions
   - Establish interest rates and financial conditions
   - Configure market clearing mechanisms

#### Standard Algorithms

```python
def setup_initial_conditions_standard(self, country: Country, 
                                    parameters: ParameterSet,
                                    economic_data: HarmonizedDataset) -> InitialConditions:
    """Standard initial conditions setup for all countries."""
    
    # Step 1: Agent population setup
    households = self.create_household_population(
        count=self.config.household_population,
        parameters=parameters.household_parameters,
        economic_data=economic_data
    )
    
    firms = self.create_firm_population(
        count=self.config.firm_population,
        parameters=parameters.firm_parameters,
        industry_data=economic_data.industry_data
    )
    
    # Step 2: Wealth distribution
    household_wealth = self.generate_wealth_distribution(
        population=households,
        total_wealth=economic_data.aggregate_wealth,
        distribution="pareto",
        alpha=self.config.wealth_alpha
    )
    
    # Step 3: Balance sheet initialization
    balance_sheets = self.initialize_balance_sheets(
        households=households,
        firms=firms,
        wealth_distribution=household_wealth,
        economic_data=economic_data
    )
    
    # Step 4: Market conditions
    market_conditions = self.setup_market_conditions(
        parameters=parameters,
        economic_data=economic_data
    )
    
    return InitialConditions(
        agent_populations={
            "households": households,
            "firms": firms
        },
        balance_sheets=balance_sheets,
        market_conditions=market_conditions
    )
```

### Country-Specific Treatment

Initial conditions setup applies uniform methodology across all countries without country-specific modifications.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| None | N/A | Standard setup for all | Cross-country comparability required |

#### Implementation Details

The initial conditions pipeline maintains consistency by applying identical setup procedures to all countries:

```python
def setup_all_countries(self, parameters_dict: Dict[Country, ParameterSet],
                       data_dict: Dict[Country, HarmonizedDataset]) -> Dict[Country, InitialConditions]:
    """Apply standard initial conditions setup to all countries."""
    
    results = {}
    for country in parameters_dict.keys():
        # Same initialization methodology for all countries
        results[country] = self.setup_initial_conditions_standard(
            country=country,
            parameters=parameters_dict[country],
            economic_data=data_dict[country]
        )
        
        # Apply universal validation
        self.validate_initial_conditions(results[country], country)
    
    return results
```

**Consistency Principles**:
- **Same Agent Types**: Identical agent categories across countries
- **Same Distributions**: Consistent wealth and size distribution shapes
- **Same Scaling**: Proportional scaling based on economic size
- **Same Validation**: Universal consistency checks and bounds

### Error Handling

- **Parameter Availability**: Use default parameters if estimates unavailable
- **Data Insufficiency**: Scale from available partial data with warnings
- **Distribution Failures**: Fall back to uniform distributions if generation fails
- **Balance Sheet Inconsistencies**: Apply normalization to ensure accounting consistency

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `agent_populations` | `Dict[Country, AgentPopulations]` | Structured dict | Agent populations by type and country |
| `balance_sheets` | `Dict[Country, BalanceSheets]` | Structured dict | Initial wealth and asset distributions |
| `market_conditions` | `Dict[Country, MarketConditions]` | Structured dict | Initial market states and parameters |
| `initialization_metadata` | `Dict[Country, InitMetadata]` | Structured dict | Setup methods and validation results |

### Output Validation

- **Population Consistency**: Verify agent counts match configuration
- **Wealth Aggregation**: Check individual wealth sums to aggregate totals
- **Balance Sheet Balance**: Ensure assets equal liabilities plus equity
- **Market Equilibrium**: Validate initial market clearing conditions

### Metadata

Information included with outputs:
- **Population Statistics**: Agent counts and distribution parameters
- **Wealth Distribution Parameters**: Pareto alpha, Gini coefficients
- **Scaling Factors**: Applied scaling for population and economic variables
- **Validation Results**: Consistency checks and economic reasonableness tests
- **Initialization Methods**: Techniques used for different agent types

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n×h + n×f) where n=countries, h=households, f=firms
- **Memory Usage**: High - large agent populations and balance sheet data
- **Random Generation**: Computationally intensive distribution sampling

### Scalability

- **Agent Population Scaling**: Linear increase with number of agents
- **Country Scaling**: Linear increase with number of countries
- **Memory Requirements**: Quadratic growth with agent population size
- **Processing Time**: ~30-60 seconds per country for standard populations

## Integration Points

### Upstream Integration

Consumes outputs from all previous pipelines:
- **Estimated Parameters**: Behavioral parameters for agent characteristics
- **Economic Data**: Aggregate values for scaling and calibration
- **Industry Data**: Sectoral information for firm specialization
- **Configuration**: Population sizes and distribution parameters

### Downstream Integration

Provides initialization data for simulation models:
- **ABM Simulators**: Complete agent populations and starting conditions
- **Validation Framework**: Initial conditions for consistency checking
- **Export System**: Agent populations in simulation-ready formats
- **Analysis Tools**: Starting conditions for sensitivity analysis

## Validation and Testing

### Unit Tests

```python
def test_household_population_creation():
    """Test household agent population generation."""
    
def test_firm_population_creation():
    """Test firm agent population with industry specialization."""
    
def test_wealth_distribution_generation():
    """Test Pareto wealth distribution generation and calibration."""
    
def test_balance_sheet_consistency():
    """Test balance sheet initialization and accounting consistency."""
    
def test_market_conditions_setup():
    """Test initial market condition parameter setting."""
```

### Integration Tests

- **End-to-End Initialization**: Complete workflow from parameters to initial conditions
- **Cross-Country Consistency**: Verify comparable initialization across countries
- **Economic Validation**: Check initial conditions against empirical distributions
- **Simulation Readiness**: Validate compatibility with ABM simulation models

## Configuration Examples

### Basic Configuration

```yaml
initial_conditions:
  household_population: 10000
  firm_population: 1000
  wealth_distribution: "pareto"
  wealth_alpha: 1.16
```

### Advanced Configuration

```yaml
initial_conditions:
  household_population: 25000
  firm_population: 2500
  bank_population: 10
  government_agents: 1
  
wealth_distribution:
  type: "pareto"
  wealth_alpha: 1.16
  income_alpha: 1.5
  wealth_floor: 1000
  
firm_distribution:
  type: "pareto"
  size_alpha: 1.06
  industry_specialization: true
  productivity_heterogeneity: 0.2
  
agent_setup:
  heterogeneity_level: "high"
  industry_specialization: true
  regional_variation: false
  behavioral_types: 3
  
scaling:
  gdp_scaling_factor: 1.0e-9
  population_scaling: 1.0e-6
  preserve_ratios: true
  normalize_totals: true
```

### Country-Specific Configuration

```yaml
# No country-specific initial conditions configuration
# All countries use identical initialization methodology

initial_conditions:
  uniform_methodology: true
  consistent_populations: true
  standardized_distributions: true
  
validation:
  cross_country_comparisons: true
  economic_reasonableness_checks: true
  distribution_shape_validation: true
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Memory errors with large populations | `MemoryError` during agent creation | Reduce population sizes or enable memory-efficient mode |
| Wealth distribution failures | Unrealistic wealth distributions | Adjust Pareto alpha parameters, check data quality |
| Balance sheet inconsistencies | Assets don't equal liabilities | Review aggregation logic, apply normalization |
| Market condition errors | Unrealistic initial prices/wages | Validate input economic data, adjust scaling factors |
| Agent population mismatches | Wrong agent counts after initialization | Check population configuration, validate creation logic |

### Debug Information

- **Initialization Logs**: Detailed agent creation and distribution setup steps
- **Population Statistics**: Agent counts, distribution parameters, summary statistics
- **Balance Sheet Summaries**: Aggregate asset/liability totals and consistency checks
- **Market Condition Validation**: Initial price and wage level reasonableness
- **Performance Metrics**: Memory usage and processing time by initialization step

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Parameter Estimation Pipeline](parameter-estimation.md) - Input parameter source
- [Model Calibration](../calibration.md) - Detailed calibration methodology
- [Calibration Outputs](../outputs.md) - Output formats and structures
- [Configuration](../../getting-started/configuration.md) - Agent population setup options