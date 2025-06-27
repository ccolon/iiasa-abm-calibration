# Industry Aggregation Pipeline

## Overview

The Industry Aggregation Pipeline transforms industry classifications from the detailed 44-sector ISIC Rev4 system to the aggregated 18-sector NACE2 classification. It processes ICIO input-output matrices, country-industry datasets, and multi-dimensional arrays while maintaining economic consistency and enabling sectoral modeling.

## Location and Dependencies

- **Module**: `src/macro_abm_calibration/processors/industry.py`
- **Class**: `IndustryAggregator`
- **Dependencies**: Raw Data Extraction Pipeline + Data Harmonization Pipeline
- **MATLAB Equivalent**: Industry aggregation logic in `c1_icios_data.m`
- **Phase**: Data Processing

## Inputs

### Input Data Structures

| Input | Type | Source | Required | Description |
|-------|------|--------|----------|-------------|
| `icio_matrices` | `Dict[str, np.ndarray]` | Raw Data Extraction | Yes | Input-output matrices by year (44×44 industries) |
| `country_data` | `Dict[Country, HarmonizedDataset]` | Data Harmonization | Yes | Country-specific economic data by industry |
| `aggregation_mapping` | `Dict[str, str]` | Configuration | Yes | ISIC Rev4 to NACE2 sector mappings |
| `target_countries` | `List[Country]` | Configuration | Yes | Countries to include in aggregated matrices |

### Input Validation

- Verify ICIO matrix dimensions (44×44×countries×years)
- Check industry code completeness in mapping configuration
- Validate country data contains industry-specific variables
- Ensure temporal alignment between ICIO and country data

### Configuration Parameters

```yaml
processing:
  industry_aggregation: "NACE2_10"
  target_industries: 18
  
industry_mapping:
  source_classification: "ISIC_REV4"
  target_classification: "NACE2"
  aggregation_method: "sum"
  
icio_processing:
  include_row_aggregation: true
  matrix_validation: true
  memory_efficient: true
```

## Data Treatment

### Standard Treatment

The pipeline applies consistent aggregation methods to transform detailed industry data into broader sectoral classifications.

#### Processing Steps

1. **Industry Mapping Application**
   - Apply ISIC Rev4 → NACE2 aggregation mapping
   - Handle multiple aggregation cases:
     - **Case 1**: 44 × years vectors (industry time series)
     - **Case 2**: 44 × 44 × years matrices (input-output tables)
     - **Case 3**: 44 × countries × years arrays (multi-country industry data)

2. **Matrix Aggregation**
   - Sum corresponding rows and columns for input-output matrices
   - Preserve economic relationships in aggregated sectors
   - Maintain matrix balance and consistency

3. **ICIO Matrix Processing**
   - Load bilateral trade flow matrices
   - Aggregate 44 industries to 18 NACE2 sectors
   - Create Rest of World (ROW) aggregation for non-target countries
   - Validate matrix properties (balance, non-negativity)

4. **Data Consistency Validation**
   - Verify aggregated totals match original sums
   - Check economic relationships preservation
   - Validate matrix mathematical properties

#### Standard Algorithms

```python
def aggregate_industry_data_standard(self, data: np.ndarray, 
                                   aggregation_case: str) -> np.ndarray:
    """Standard industry aggregation for all data types."""
    
    mapping_matrix = self.get_aggregation_matrix()
    
    if aggregation_case == "vector_time_series":
        # Case 1: 44 × years → 18 × years
        return mapping_matrix @ data
        
    elif aggregation_case == "io_matrices":
        # Case 2: 44 × 44 × years → 18 × 18 × years
        return self.aggregate_io_matrices(data, mapping_matrix)
        
    elif aggregation_case == "multi_country":
        # Case 3: 44 × countries × years → 18 × countries × years
        return self.aggregate_multi_dimensional(data, mapping_matrix)
```

### Country-Specific Treatment

Industry aggregation applies uniform methodology across all countries without country-specific exceptions.

#### Country Exceptions

| Country | Exception Type | Processing Modification | Reason |
|---------|----------------|------------------------|--------|
| None | N/A | Standard aggregation for all | Industry mapping is universal |

#### Implementation Details

The industry aggregation pipeline maintains consistency by applying identical aggregation methods to all countries:

```python
def process_all_countries(self, data_dict: Dict[Country, Any]) -> Dict[Country, Any]:
    """Apply standard aggregation to all countries uniformly."""
    
    results = {}
    for country, country_data in data_dict.items():
        # Same aggregation logic for all countries
        results[country] = self.aggregate_industry_data_standard(
            country_data, 
            aggregation_case=self.detect_data_structure(country_data)
        )
    
    return results
```

**ICIO Special Processing**:
- **ROW Aggregation**: Countries not in target list aggregated into "Rest of World"
- **Matrix Shrinking**: Reduce full global matrices to target country subset
- **Balance Preservation**: Ensure aggregated matrices maintain economic balance

### Error Handling

- **Dimension Mismatches**: Validate input dimensions match expected structure
- **Missing Industries**: Handle incomplete industry coverage gracefully
- **Matrix Singularity**: Check for and resolve matrix mathematical issues
- **Aggregation Errors**: Validate aggregated totals and detect inconsistencies

## Outputs

### Output Data Structures

| Output | Type | Format | Description |
|--------|------|--------|-------------|
| `aggregated_icio` | `Dict[str, np.ndarray]` | NumPy arrays | 18×18 NACE2 input-output matrices |
| `aggregated_country_data` | `Dict[Country, AggregatedDataset]` | Structured dict | Country data with 18 NACE2 sectors |
| `aggregation_metadata` | `AggregationMetadata` | Structured object | Mapping details and validation results |
| `sector_mappings` | `Dict[str, List[str]]` | Dict of lists | ISIC codes grouped by NACE2 sector |

### Output Validation

- **Aggregation Consistency**: Verify totals preserved through aggregation
- **Matrix Properties**: Check positive semi-definiteness and balance
- **Sector Completeness**: Ensure all target sectors represented
- **Economic Relationships**: Validate preserved input-output relationships

### Metadata

Information included with outputs:
- **Aggregation Methods**: Summation vs weighted average techniques used
- **Sector Mappings**: Complete ISIC Rev4 to NACE2 correspondence
- **Validation Results**: Matrix balance and consistency checks
- **ROW Composition**: Countries included in Rest of World aggregation
- **Processing Statistics**: Aggregation ratios and coverage metrics

## Performance Characteristics

### Computational Complexity

- **Time Complexity**: O(n²×c×t) where n=industries, c=countries, t=time periods
- **Memory Usage**: High - large multi-dimensional matrices
- **Matrix Operations**: Computationally intensive aggregation operations

### Scalability

- **Industry Scaling**: Quadratic increase with source industry count
- **Country Scaling**: Linear increase with number of countries
- **Time Period Scaling**: Linear increase with temporal coverage
- **Processing Time**: ~5-10 seconds per year of ICIO data

## Integration Points

### Upstream Integration

Consumes outputs from multiple pipelines:
- **Raw ICIO Data**: 44×44 industry matrices from Raw Data Extraction
- **Harmonized Country Data**: Industry-specific economic indicators
- **Configuration**: Industry mapping and aggregation specifications
- **Country List**: Target countries for matrix processing

### Downstream Integration

Provides aggregated data to calibration pipelines:
- **Parameter Estimation**: Sectoral data for industry-specific parameters
- **Initial Conditions**: Industry distributions for agent initialization
- **Model Validation**: Aggregated data for consistency checks
- **Export System**: NACE2-based outputs for downstream analysis

## Validation and Testing

### Unit Tests

```python
def test_vector_aggregation():
    """Test 44×years to 18×years vector aggregation."""
    
def test_matrix_aggregation():
    """Test 44×44 to 18×18 matrix aggregation."""
    
def test_multi_dimensional_aggregation():
    """Test 44×countries×years aggregation."""
    
def test_icio_matrix_processing():
    """Test complete ICIO matrix aggregation workflow."""
    
def test_aggregation_consistency():
    """Test that aggregated totals match original sums."""
```

### Integration Tests

- **End-to-End ICIO Processing**: Complete workflow from raw to aggregated matrices
- **Cross-Dataset Consistency**: Verify alignment between ICIO and country data
- **MATLAB Comparison**: Validate against original MATLAB aggregation results
- **Economic Validation**: Check preservation of input-output relationships

## Configuration Examples

### Basic Configuration

```yaml
processing:
  industry_aggregation: "NACE2_10"
  
industry_mapping:
  source_classification: "ISIC_REV4"
  target_classification: "NACE2"
  aggregation_method: "sum"
```

### Advanced Configuration

```yaml
processing:
  industry_aggregation: "NACE2_10"
  target_industries: 18
  
industry_mapping:
  source_classification: "ISIC_REV4"
  target_classification: "NACE2"
  aggregation_method: "sum"
  preserve_ratios: true
  
icio_processing:
  include_row_aggregation: true
  row_countries: ["ROW"]
  matrix_validation: true
  balance_tolerance: 0.001
  memory_efficient: true
  parallel_processing: true
  
validation:
  check_matrix_properties: true
  validate_economic_relationships: true
  cross_validate_totals: true
```

### Country-Specific Configuration

```yaml
# No country-specific configuration needed for industry aggregation
# All countries use identical ISIC Rev4 → NACE2 mapping

industry_mapping:
  uniform_mapping: true
  apply_to_all_countries: true
  
icio_processing:
  target_countries: ["USA", "DEU", "FRA", "ITA", "ESP", "GBR", "JPN"]
  row_aggregation:
    method: "sum"
    include_all_others: true
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Matrix dimension errors | `ValueError` in aggregation operations | Verify input matrix dimensions match expected 44×44 structure |
| Aggregation inconsistencies | Totals don't match after aggregation | Check mapping completeness, validate aggregation method |
| Memory errors with large ICIO | `MemoryError` during matrix processing | Enable memory-efficient mode, process years sequentially |
| Missing industry codes | `KeyError` for ISIC codes | Verify industry mapping completeness, update configuration |
| Matrix balance violations | Economic inconsistencies in output | Check input data quality, adjust balance tolerance |

### Debug Information

- **Aggregation Logs**: Detailed mapping application and validation steps
- **Matrix Statistics**: Dimensions, totals, and balance metrics before/after
- **Memory Usage**: Peak memory consumption during processing
- **Performance Metrics**: Processing time by aggregation case and dataset size
- **Validation Results**: Economic consistency and mathematical property checks

## Related Documentation

- [Workflow Overview](../workflow-overview.md) - Overall system architecture
- [Data Harmonization Pipeline](data-harmonization.md) - Parallel pipeline in workflow
- [Parameter Estimation Pipeline](parameter-estimation.md) - Next pipeline using aggregated data
- [ICIO Data Source](../data-sources.md) - Input-output data specifications
- [Configuration](../../getting-started/configuration.md) - Industry mapping setup