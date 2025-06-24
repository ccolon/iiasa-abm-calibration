"""
Calibration result export and visualization.

This module provides functionality for exporting calibration results to various
formats (MATLAB, JSON, Excel) and creating visualizations of the calibrated
parameters and model validation results.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import savemat

from .base import CalibrationResult
from .validation import ValidationSummary


class CalibrationExporter:
    """
    Export calibration results to various formats.
    
    This class handles the export of calibration results including estimated
    parameters, initial conditions, and validation results to formats compatible
    with MATLAB, Excel, and Python analysis tools.
    """
    
    def __init__(self, output_directory: Union[str, Path]):
        """
        Initialize calibration exporter.
        
        Args:
            output_directory: Base directory for output files
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def export_parameters(
        self,
        parameters: Dict[str, Dict[str, float]],
        format: str = "all",
        filename_prefix: str = "calibrated_parameters"
    ) -> List[str]:
        """
        Export estimated parameters to files.
        
        Args:
            parameters: Dictionary of parameters by country
            format: Export format ('matlab', 'excel', 'json', 'csv', 'all')
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of created file paths
        """
        created_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["matlab", "all"]:
            # MATLAB format
            matlab_file = self.output_directory / f"{filename_prefix}_{timestamp}.mat"
            matlab_data = self._convert_parameters_for_matlab(parameters)
            savemat(matlab_file, matlab_data)
            created_files.append(str(matlab_file))
        
        if format in ["excel", "all"]:
            # Excel format
            excel_file = self.output_directory / f"{filename_prefix}_{timestamp}.xlsx"
            self._export_parameters_to_excel(parameters, excel_file)
            created_files.append(str(excel_file))
        
        if format in ["json", "all"]:
            # JSON format
            json_file = self.output_directory / f"{filename_prefix}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(parameters, f, indent=2, default=self._json_serializer)
            created_files.append(str(json_file))
        
        if format in ["csv", "all"]:
            # CSV format
            csv_file = self.output_directory / f"{filename_prefix}_{timestamp}.csv"
            df = self._parameters_to_dataframe(parameters)
            df.to_csv(csv_file, index=False)
            created_files.append(str(csv_file))
        
        return created_files
    
    def export_initial_conditions(
        self,
        initial_conditions: Dict[str, Dict[str, Any]],
        format: str = "all",
        filename_prefix: str = "initial_conditions"
    ) -> List[str]:
        """
        Export initial conditions to files.
        
        Args:
            initial_conditions: Dictionary of initial conditions by country
            format: Export format
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of created file paths
        """
        created_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["matlab", "all"]:
            # MATLAB format
            matlab_file = self.output_directory / f"{filename_prefix}_{timestamp}.mat"
            matlab_data = self._convert_initial_conditions_for_matlab(initial_conditions)
            savemat(matlab_file, matlab_data)
            created_files.append(str(matlab_file))
        
        if format in ["json", "all"]:
            # JSON format
            json_file = self.output_directory / f"{filename_prefix}_{timestamp}.json"
            json_data = self._convert_initial_conditions_for_json(initial_conditions)
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=self._json_serializer)
            created_files.append(str(json_file))
        
        if format in ["pickle", "all"]:
            # Pickle format (preserves full Python objects)
            pickle_file = self.output_directory / f"{filename_prefix}_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(initial_conditions, f)
            created_files.append(str(pickle_file))
        
        return created_files
    
    def export_validation_results(
        self,
        validation_summary: ValidationSummary,
        format: str = "all",
        filename_prefix: str = "validation_results"
    ) -> List[str]:
        """
        Export validation results to files.
        
        Args:
            validation_summary: Validation summary object
            format: Export format
            filename_prefix: Prefix for output filenames
            
        Returns:
            List of created file paths
        """
        created_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert validation summary to exportable format
        validation_data = {
            "summary": {
                "total_checks": validation_summary.total_checks,
                "passed_checks": validation_summary.passed_checks,
                "failed_checks": validation_summary.failed_checks,
                "warnings": validation_summary.warnings,
                "errors": validation_summary.errors,
                "critical_issues": validation_summary.critical_issues,
                "overall_passed": validation_summary.overall_passed
            },
            "checks": [
                {
                    "name": check.name,
                    "level": check.level.value,
                    "passed": check.passed,
                    "message": check.message,
                    "details": check.details
                }
                for check in validation_summary.checks
            ]
        }
        
        if format in ["json", "all"]:
            # JSON format
            json_file = self.output_directory / f"{filename_prefix}_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(validation_data, f, indent=2, default=self._json_serializer)
            created_files.append(str(json_file))
        
        if format in ["excel", "all"]:
            # Excel format
            excel_file = self.output_directory / f"{filename_prefix}_{timestamp}.xlsx"
            self._export_validation_to_excel(validation_data, excel_file)
            created_files.append(str(excel_file))
        
        return created_files
    
    def export_complete_results(
        self,
        calibration_results: Dict[str, CalibrationResult],
        filename_prefix: str = "complete_calibration_results"
    ) -> str:
        """
        Export complete calibration results to a comprehensive file.
        
        Args:
            calibration_results: Dictionary of all calibration results
            filename_prefix: Prefix for output filename
            
        Returns:
            Path to created file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_directory / f"{filename_prefix}_{timestamp}.json"
        
        # Compile all results
        complete_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "calibration_components": list(calibration_results.keys())
            },
            "results": {}
        }
        
        for component_name, result in calibration_results.items():
            complete_data["results"][component_name] = {
                "status": result.status.value,
                "data": result.data,
                "metadata": {
                    "calibrator_name": result.metadata.calibrator_name,
                    "operation_id": result.metadata.operation_id,
                    "timestamp": result.metadata.timestamp.isoformat(),
                    "parameters": result.metadata.parameters,
                    "computation_time": result.metadata.computation_time,
                    "validation_results": result.metadata.validation_results
                },
                "errors": result.errors,
                "warnings": result.warnings
            }
        
        with open(output_file, 'w') as f:
            json.dump(complete_data, f, indent=2, default=self._json_serializer)
        
        return str(output_file)
    
    def _convert_parameters_for_matlab(self, parameters: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Convert parameters to MATLAB-compatible format."""
        matlab_data = {}
        
        for country, params in parameters.items():
            country_data = {}
            
            # Group parameters by type
            taylor_params = {k: v for k, v in params.items() if k.startswith("taylor_")}
            firm_params = {k: v for k, v in params.items() if "firm" in k or "price" in k or "investment" in k or "markup" in k}
            household_params = {k: v for k, v in params.items() if "household" in k or "consumption" in k or "wealth" in k or "labor" in k}
            
            if taylor_params:
                country_data["taylor_rule"] = taylor_params
            if firm_params:
                country_data["firm_parameters"] = firm_params
            if household_params:
                country_data["household_parameters"] = household_params
            
            # Add any remaining parameters
            other_params = {k: v for k, v in params.items() 
                          if k not in taylor_params and k not in firm_params and k not in household_params}
            if other_params:
                country_data["other_parameters"] = other_params
            
            matlab_data[country] = country_data
        
        return matlab_data
    
    def _convert_initial_conditions_for_matlab(self, initial_conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert initial conditions to MATLAB-compatible format."""
        matlab_data = {}
        
        for country, conditions in initial_conditions.items():
            country_data = {}
            
            # Agent populations
            if "agent_populations" in conditions:
                populations = conditions["agent_populations"]
                for agent_type, population in populations.items():
                    agent_key = f"{agent_type.value}_population"
                    country_data[agent_key] = {
                        "count": population.count,
                        "distribution_params": population.distribution_params,
                        "initial_conditions": population.initial_conditions
                    }
            
            # Balance sheets
            if "balance_sheets" in conditions:
                for sheet_type, sheet_data in conditions["balance_sheets"].items():
                    sheet_key = f"{sheet_type}_balance_sheet"
                    # Convert numpy arrays to lists for MATLAB
                    converted_data = {}
                    for key, value in sheet_data.items():
                        if isinstance(value, np.ndarray):
                            converted_data[key] = value.tolist()
                        else:
                            converted_data[key] = value
                    country_data[sheet_key] = converted_data
            
            # Market conditions
            if "market_conditions" in conditions:
                market = conditions["market_conditions"]
                country_data["market_conditions"] = {
                    "price_level": market.price_level,
                    "wage_level": market.wage_level,
                    "interest_rate": market.interest_rate,
                    "exchange_rate": market.exchange_rate,
                    "unemployment_rate": market.unemployment_rate,
                    "capacity_utilization": market.capacity_utilization
                }
            
            matlab_data[country] = country_data
        
        return matlab_data
    
    def _convert_initial_conditions_for_json(self, initial_conditions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert initial conditions to JSON-compatible format."""
        def convert_value(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert_value(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        return convert_value(initial_conditions)
    
    def _export_parameters_to_excel(self, parameters: Dict[str, Dict[str, float]], output_file: Path) -> None:
        """Export parameters to Excel format."""
        df = self._parameters_to_dataframe(parameters)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Separate sheets by parameter type
            taylor_params = df[df['parameter'].str.startswith('taylor_')]
            if not taylor_params.empty:
                pivot_taylor = taylor_params.pivot(index='country', columns='parameter', values='value')
                pivot_taylor.to_excel(writer, sheet_name='Taylor_Rule')
            
            firm_params = df[df['parameter'].str.contains('firm|price|investment|markup')]
            if not firm_params.empty:
                pivot_firm = firm_params.pivot(index='country', columns='parameter', values='value')
                pivot_firm.to_excel(writer, sheet_name='Firm_Parameters')
            
            household_params = df[df['parameter'].str.contains('household|consumption|wealth|labor')]
            if not household_params.empty:
                pivot_household = household_params.pivot(index='country', columns='parameter', values='value')
                pivot_household.to_excel(writer, sheet_name='Household_Parameters')
    
    def _export_validation_to_excel(self, validation_data: Dict[str, Any], output_file: Path) -> None:
        """Export validation results to Excel format."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame([validation_data["summary"]])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed checks sheet
            checks_df = pd.DataFrame(validation_data["checks"])
            checks_df.to_excel(writer, sheet_name='Validation_Checks', index=False)
    
    def _parameters_to_dataframe(self, parameters: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert parameters dictionary to DataFrame."""
        rows = []
        for country, params in parameters.items():
            for param_name, param_value in params.items():
                rows.append({
                    "country": country,
                    "parameter": param_name,
                    "value": param_value
                })
        
        return pd.DataFrame(rows)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return str(obj)


class CalibrationVisualizer:
    """
    Create visualizations of calibration results.
    
    This class generates plots and charts to visualize estimated parameters,
    validation results, and model diagnostics.
    """
    
    def __init__(self, output_directory: Union[str, Path]):
        """
        Initialize calibration visualizer.
        
        Args:
            output_directory: Directory for saving plots
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_parameter_comparison(
        self,
        parameters: Dict[str, Dict[str, float]],
        reference_parameters: Optional[Dict[str, Dict[str, float]]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot comparison of parameters across countries.
        
        Args:
            parameters: Estimated parameters
            reference_parameters: Reference parameters for comparison
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        df = self._parameters_to_dataframe(parameters)
        
        # Create separate plots for different parameter types
        taylor_params = ["taylor_inflation_response", "taylor_output_response", "taylor_smoothing"]
        firm_params = ["price_adjustment_speed", "investment_sensitivity", "markup_elasticity"]
        household_params = ["marginal_propensity_consume", "wealth_effect", "labor_supply_elasticity"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Calibrated Parameters by Country', fontsize=16)
        
        # Taylor rule parameters
        taylor_df = df[df['parameter'].isin(taylor_params)]
        if not taylor_df.empty:
            taylor_pivot = taylor_df.pivot(index='country', columns='parameter', values='value')
            taylor_pivot.plot(kind='bar', ax=axes[0, 0], title='Taylor Rule Parameters')
            axes[0, 0].set_ylabel('Parameter Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Firm parameters
        firm_df = df[df['parameter'].isin(firm_params)]
        if not firm_df.empty:
            firm_pivot = firm_df.pivot(index='country', columns='parameter', values='value')
            firm_pivot.plot(kind='bar', ax=axes[0, 1], title='Firm Parameters')
            axes[0, 1].set_ylabel('Parameter Value')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Household parameters
        household_df = df[df['parameter'].isin(household_params)]
        if not household_df.empty:
            household_pivot = household_df.pivot(index='country', columns='parameter', values='value')
            household_pivot.plot(kind='bar', ax=axes[1, 0], title='Household Parameters')
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Parameter distribution
        param_counts = df['parameter'].value_counts()
        param_counts.plot(kind='bar', ax=axes[1, 1], title='Parameter Coverage')
        axes[1, 1].set_ylabel('Number of Countries')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_directory / f"parameter_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_validation_results(
        self,
        validation_summary: ValidationSummary,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot validation results summary.
        
        Args:
            validation_summary: Validation summary object
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Validation Results', fontsize=16)
        
        # Overall summary pie chart
        summary_data = [
            validation_summary.passed_checks,
            validation_summary.failed_checks
        ]
        labels = ['Passed', 'Failed']
        colors = ['green', 'red']
        
        axes[0, 0].pie(summary_data, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Check Results')
        
        # Issue severity breakdown
        severity_data = [
            validation_summary.warnings,
            validation_summary.errors,
            validation_summary.critical_issues
        ]
        severity_labels = ['Warnings', 'Errors', 'Critical']
        severity_colors = ['orange', 'red', 'darkred']
        
        if sum(severity_data) > 0:
            axes[0, 1].pie(severity_data, labels=severity_labels, colors=severity_colors, autopct='%1.1f%%')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Issues Found', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Issue Severity')
        
        # Check results by category
        check_categories = {}
        for check in validation_summary.checks:
            category = check.name.split('_')[0]
            if category not in check_categories:
                check_categories[category] = {'passed': 0, 'failed': 0}
            
            if check.passed:
                check_categories[category]['passed'] += 1
            else:
                check_categories[category]['failed'] += 1
        
        if check_categories:
            categories = list(check_categories.keys())
            passed_counts = [check_categories[cat]['passed'] for cat in categories]
            failed_counts = [check_categories[cat]['failed'] for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, passed_counts, width, label='Passed', color='green', alpha=0.7)
            axes[1, 0].bar(x + width/2, failed_counts, width, label='Failed', color='red', alpha=0.7)
            
            axes[1, 0].set_xlabel('Check Category')
            axes[1, 0].set_ylabel('Number of Checks')
            axes[1, 0].set_title('Checks by Category')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories, rotation=45)
            axes[1, 0].legend()
        
        # Validation summary statistics
        stats_text = f"""
        Total Checks: {validation_summary.total_checks}
        Passed: {validation_summary.passed_checks}
        Failed: {validation_summary.failed_checks}
        
        Warnings: {validation_summary.warnings}
        Errors: {validation_summary.errors}
        Critical: {validation_summary.critical_issues}
        
        Overall: {'PASSED' if validation_summary.overall_passed else 'FAILED'}
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_directory / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_parameter_distributions(
        self,
        parameters: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Plot distributions of parameters across countries.
        
        Args:
            parameters: Estimated parameters
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        df = self._parameters_to_dataframe(parameters)
        
        # Get unique parameters
        unique_params = df['parameter'].unique()
        n_params = len(unique_params)
        
        # Create subplot grid
        cols = 3
        rows = (n_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Parameter Distributions Across Countries', fontsize=16)
        
        for i, param in enumerate(unique_params):
            row = i // cols
            col = i % cols
            
            param_data = df[df['parameter'] == param]['value']
            
            if len(param_data) > 1:
                axes[row, col].hist(param_data, bins=min(10, len(param_data)), alpha=0.7, edgecolor='black')
                axes[row, col].axvline(param_data.mean(), color='red', linestyle='--', label=f'Mean: {param_data.mean():.3f}')
                axes[row, col].axvline(param_data.median(), color='green', linestyle='--', label=f'Median: {param_data.median():.3f}')
            else:
                axes[row, col].bar(['Value'], [param_data.iloc[0]], alpha=0.7)
            
            axes[row, col].set_title(param)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
        
        # Hide empty subplots
        for i in range(n_params, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_directory / f"parameter_distributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _parameters_to_dataframe(self, parameters: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert parameters dictionary to DataFrame."""
        rows = []
        for country, params in parameters.items():
            for param_name, param_value in params.items():
                rows.append({
                    "country": country,
                    "parameter": param_name,
                    "value": param_value
                })
        
        return pd.DataFrame(rows)
    
    def create_calibration_report(
        self,
        parameters: Dict[str, Dict[str, float]],
        validation_summary: ValidationSummary,
        initial_conditions: Optional[Dict[str, Dict[str, Any]]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive calibration report with multiple visualizations.
        
        Args:
            parameters: Estimated parameters
            validation_summary: Validation results
            initial_conditions: Initial conditions (optional)
            save_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        # Create individual plots
        param_plot = self.plot_parameter_comparison(parameters)
        validation_plot = self.plot_validation_results(validation_summary)
        distribution_plot = self.plot_parameter_distributions(parameters)
        
        # Create combined report figure
        fig = plt.figure(figsize=(20, 24))
        
        # Load and display individual plots
        from PIL import Image
        
        # Parameter comparison
        param_img = Image.open(param_plot)
        ax1 = plt.subplot(3, 1, 1)
        ax1.imshow(param_img)
        ax1.axis('off')
        ax1.set_title('Parameter Comparison Across Countries', fontsize=16, pad=20)
        
        # Validation results
        validation_img = Image.open(validation_plot)
        ax2 = plt.subplot(3, 1, 2)
        ax2.imshow(validation_img)
        ax2.axis('off')
        ax2.set_title('Model Validation Results', fontsize=16, pad=20)
        
        # Parameter distributions
        dist_img = Image.open(distribution_plot)
        ax3 = plt.subplot(3, 1, 3)
        ax3.imshow(dist_img)
        ax3.axis('off')
        ax3.set_title('Parameter Distributions', fontsize=16, pad=20)
        
        # Add overall title and metadata
        fig.suptitle('Macroeconomic ABM Calibration Report', fontsize=20, y=0.98)
        
        # Add timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(0.02, 0.02, timestamp_text, fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_directory / f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clean up individual plot files
        Path(param_plot).unlink()
        Path(validation_plot).unlink()
        Path(distribution_plot).unlink()
        
        return str(save_path)