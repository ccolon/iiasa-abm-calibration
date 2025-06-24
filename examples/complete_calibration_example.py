#!/usr/bin/env python3
"""
Complete ABM Calibration Example

This script demonstrates the full calibration workflow for a macroeconomic
agent-based model, from data processing to parameter estimation and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from macro_abm_calibration import (
    CalibrationConfig,
    DataSourceManager,
    CalibrationPipeline,
    ABMParameterEstimator,
    InitialConditionsSetter,
    ModelValidator
)
from macro_abm_calibration.calibrators.base import CalibrationManager
from macro_abm_calibration.calibrators.export import CalibrationExporter, CalibrationVisualizer


def create_sample_data() -> pd.DataFrame:
    """
    Create sample economic data for demonstration.
    
    In a real application, this would be replaced with actual OECD data
    loading using the data sources.
    """
    print("Creating sample economic data...")
    
    countries = ["USA", "GBR", "DEU", "FRA", "ITA"]
    dates = pd.date_range("2010", "2023", freq="Q")
    
    data_list = []
    
    for country in countries:
        # Country-specific base values
        country_multipliers = {
            "USA": 1.0,
            "GBR": 0.8,
            "DEU": 0.9,
            "FRA": 0.85,
            "ITA": 0.7
        }
        
        multiplier = country_multipliers[country]
        
        for i, date in enumerate(dates):
            # Create realistic economic time series with trends and cycles
            trend_factor = 1 + 0.02 * (i / len(dates))  # 2% annual growth
            cycle_factor = 1 + 0.05 * np.sin(2 * np.pi * i / 4)  # Business cycle
            noise = np.random.normal(0, 0.02)
            
            base_gdp = 1000 * multiplier * trend_factor * cycle_factor * (1 + noise)
            
            data_list.append({
                "REF_AREA": country,
                "TIME_PERIOD": date.strftime("%Y-Q%q"),
                "gdp": base_gdp,
                "consumption": base_gdp * 0.7 * (1 + np.random.normal(0, 0.01)),
                "investment": base_gdp * 0.2 * (1 + np.random.normal(0, 0.03)),
                "exports": base_gdp * 0.3 * (1 + np.random.normal(0, 0.05)),
                "imports": base_gdp * 0.25 * (1 + np.random.normal(0, 0.05)),
                "interest_rate": max(0.1, 2.0 + np.random.normal(0, 0.5)),
                "inflation_rate": max(-2.0, 2.0 + np.random.normal(0, 0.8)),
                "unemployment_rate": max(1.0, 5.0 + np.random.normal(0, 1.0))
            })
    
    df = pd.DataFrame(data_list)
    print(f"Created data for {len(countries)} countries, {len(dates)} quarters")
    return df


def setup_calibration_pipeline(sample_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Set up and run the data processing pipeline.
    """
    print("\nSetting up calibration pipeline...")
    
    # Create configuration
    config = CalibrationConfig(
        countries=["USA", "GBR", "DEU"],
        industries=["A", "C", "G", "K"],  # Subset for demo
        estimation_period_start_year=2010,
        estimation_period_end_year=2023,
        calibration_period_start_year=2020,
        calibration_period_end_year=2023
    )
    
    # Note: In a real application, you would use actual data sources
    # For this example, we'll work directly with the sample data
    print("✓ Configuration created")
    
    return {"processed_data": sample_data, "config": config}


def run_parameter_estimation(processed_data: pd.DataFrame, config: CalibrationConfig) -> Dict[str, Any]:
    """
    Run ABM parameter estimation.
    """
    print("\nRunning parameter estimation...")
    
    estimator = ABMParameterEstimator()
    
    parameters = {
        "countries": config.countries,
        "parameter_groups": ["taylor_rule", "firm_behavior", "household_behavior"],
        "estimation_method": "ols"
    }
    
    result = estimator.calibrate(processed_data, parameters)
    
    if result.is_success:
        estimated_params = result.data["estimated_parameters"]
        print(f"✓ Successfully estimated parameters for {len(estimated_params)} countries")
        
        # Display sample results
        for country, params in estimated_params.items():
            print(f"  {country}:")
            for param_name, param_value in list(params.items())[:3]:  # Show first 3
                print(f"    {param_name}: {param_value:.3f}")
            if len(params) > 3:
                print(f"    ... and {len(params) - 3} more parameters")
    else:
        print(f"✗ Parameter estimation failed: {result.errors}")
        return {}
    
    return result.data


def run_initial_conditions_setup(processed_data: pd.DataFrame, config: CalibrationConfig) -> Dict[str, Any]:
    """
    Set up initial conditions for the ABM.
    """
    print("\nSetting up initial conditions...")
    
    setter = InitialConditionsSetter()
    
    parameters = {
        "countries": config.countries,
        "population_scaling": {"USA": 1.0, "GBR": 0.8, "DEU": 0.9}
    }
    
    result = setter.calibrate(processed_data, parameters)
    
    if result.is_success:
        initial_conditions = result.data["initial_conditions"]
        print(f"✓ Successfully set initial conditions for {len(initial_conditions)} countries")
        
        # Display sample results
        for country, conditions in initial_conditions.items():
            if "agent_populations" in conditions:
                populations = conditions["agent_populations"]
                total_agents = sum(pop.count for pop in populations.values())
                print(f"  {country}: {total_agents:,} total agents")
    else:
        print(f"✗ Initial conditions setup failed: {result.errors}")
        return {}
    
    return result.data


def run_model_validation(
    estimated_parameters: Dict[str, Any],
    initial_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate the calibrated model.
    """
    print("\nRunning model validation...")
    
    validator = ModelValidator()
    
    validation_data = {
        "estimated_parameters": estimated_parameters,
        "initial_conditions": initial_conditions
    }
    
    parameters = {
        "validation_level": "standard"
    }
    
    result = validator.calibrate(validation_data, parameters)
    
    if result.is_success:
        summary = result.data["validation_summary"]
        print(f"✓ Validation completed:")
        print(f"  Total checks: {summary.total_checks}")
        print(f"  Passed: {summary.passed_checks}")
        print(f"  Failed: {summary.failed_checks}")
        print(f"  Warnings: {summary.warnings}")
        print(f"  Overall result: {'PASSED' if summary.overall_passed else 'FAILED'}")
    else:
        print(f"✗ Validation failed: {result.errors}")
        return {}
    
    return result.data


def export_and_visualize_results(
    estimated_parameters: Dict[str, Any],
    initial_conditions: Dict[str, Any],
    validation_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Export and visualize calibration results.
    """
    print(f"\nExporting results to {output_dir}...")
    
    # Create exporters
    exporter = CalibrationExporter(output_dir)
    visualizer = CalibrationVisualizer(output_dir)
    
    # Export parameters
    if estimated_parameters:
        param_files = exporter.export_parameters(
            estimated_parameters["estimated_parameters"],
            format="all"
        )
        print(f"✓ Exported parameters to {len(param_files)} files")
    
    # Export initial conditions
    if initial_conditions:
        ic_files = exporter.export_initial_conditions(
            initial_conditions["initial_conditions"],
            format="json"
        )
        print(f"✓ Exported initial conditions to {len(ic_files)} files")
    
    # Create visualizations
    if estimated_parameters and validation_results:
        try:
            # Parameter comparison plot
            param_plot = visualizer.plot_parameter_comparison(
                estimated_parameters["estimated_parameters"]
            )
            print(f"✓ Created parameter comparison plot: {Path(param_plot).name}")
            
            # Validation results plot
            validation_plot = visualizer.plot_validation_results(
                validation_results["validation_summary"]
            )
            print(f"✓ Created validation results plot: {Path(validation_plot).name}")
            
            # Comprehensive report
            report_path = visualizer.create_calibration_report(
                estimated_parameters["estimated_parameters"],
                validation_results["validation_summary"],
                initial_conditions.get("initial_conditions")
            )
            print(f"✓ Created comprehensive report: {Path(report_path).name}")
            
        except Exception as e:
            print(f"⚠ Visualization failed (optional): {e}")


def main():
    """
    Main calibration workflow.
    """
    print("=" * 60)
    print("Macroeconomic ABM Calibration - Complete Example")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path("calibration_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Create sample data (replace with real data loading)
        sample_data = create_sample_data()
        
        # Step 2: Set up calibration pipeline
        pipeline_results = setup_calibration_pipeline(sample_data)
        
        if not pipeline_results:
            print("✗ Pipeline setup failed")
            return
        
        processed_data = pipeline_results["processed_data"]
        config = pipeline_results["config"]
        
        # Step 3: Estimate parameters
        estimated_parameters = run_parameter_estimation(processed_data, config)
        
        # Step 4: Set up initial conditions
        initial_conditions = run_initial_conditions_setup(processed_data, config)
        
        # Step 5: Validate model
        validation_results = {}
        if estimated_parameters and initial_conditions:
            validation_results = run_model_validation(
                estimated_parameters["estimated_parameters"],
                initial_conditions["initial_conditions"]
            )
        
        # Step 6: Export and visualize results
        export_and_visualize_results(
            estimated_parameters,
            initial_conditions,
            validation_results,
            output_dir
        )
        
        print("\n" + "=" * 60)
        print("✓ Calibration workflow completed successfully!")
        print(f"✓ Results saved to: {output_dir.absolute()}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Calibration workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()