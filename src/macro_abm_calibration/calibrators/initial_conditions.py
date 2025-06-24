"""
Initial conditions setter for ABM models.

This module implements the initialization of agent populations, balance sheets,
and market conditions for agent-based macroeconomic models, following the
patterns established in the original MATLAB code.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats

from .base import EconomicCalibrator, CalibrationResult, CalibrationStatus, CalibrationMetadata
from ..models import Country, Industry, OECD_COUNTRIES, NACE2_INDUSTRIES


class AgentType(Enum):
    """Types of agents in the ABM."""
    HOUSEHOLD = "household"
    FIRM = "firm"
    BANK = "bank"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"


@dataclass
class AgentPopulation:
    """Agent population specification."""
    agent_type: AgentType
    count: int
    distribution_params: Dict[str, Any] = field(default_factory=dict)
    initial_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Initial market conditions."""
    price_level: float = 1.0
    wage_level: float = 1.0
    interest_rate: float = 0.02
    exchange_rate: float = 1.0
    unemployment_rate: float = 0.05
    capacity_utilization: float = 0.8


class InitialConditionsSetter(EconomicCalibrator):
    """
    Initial conditions setter for ABM models.
    
    This class sets up the initial state of the agent-based model including:
    - Agent populations and their characteristics
    - Balance sheets for all agent types
    - Market conditions and prices
    - Industry-specific conditions
    """
    
    # Default population sizes (scaled by country GDP)
    DEFAULT_POPULATIONS = {
        AgentType.HOUSEHOLD: 10000,  # Base number of households
        AgentType.FIRM: 1000,        # Base number of firms
        AgentType.BANK: 50,          # Base number of banks
        AgentType.GOVERNMENT: 1,     # One government per country
        AgentType.CENTRAL_BANK: 1    # One central bank per country
    }
    
    # Wealth distribution parameters (Pareto distribution)
    WEALTH_DISTRIBUTION_PARAMS = {
        "household_wealth_alpha": 1.16,  # Pareto shape parameter
        "firm_size_alpha": 1.06,         # Firm size distribution
        "bank_size_alpha": 1.2           # Bank size distribution  
    }
    
    def __init__(self, name: Optional[str] = None):
        """Initialize initial conditions setter."""
        super().__init__(name or "InitialConditionsSetter")
        self.agent_populations: Dict[str, Dict[AgentType, AgentPopulation]] = {}
        self.market_conditions: Dict[str, MarketConditions] = {}
    
    def calibrate(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Set initial conditions for ABM models.
        
        Args:
            data: Economic data for calibration
            parameters: Initial conditions parameters including:
                - countries: List of countries
                - population_scaling: Population scaling factors
                - wealth_distribution: Wealth distribution parameters
                - market_conditions: Initial market conditions
                
        Returns:
            CalibrationResult with initial conditions
        """
        operation_id = self._generate_operation_id()
        
        # Validate inputs
        errors = self.validate_inputs(data, parameters)
        if errors:
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            return CalibrationResult(
                status=CalibrationStatus.FAILED,
                data={},
                metadata=metadata,
                errors=errors
            )
        
        # Extract parameters
        countries = parameters.get("countries", []) if parameters else []
        population_scaling = parameters.get("population_scaling", {}) if parameters else {}
        wealth_params = parameters.get("wealth_distribution", self.WEALTH_DISTRIBUTION_PARAMS) if parameters else self.WEALTH_DISTRIBUTION_PARAMS
        
        self.logger.info(f"Setting initial conditions for {len(countries)} countries")
        
        try:
            initial_conditions = {}
            
            # Process each country
            for country in countries:
                country_data = self._extract_country_data(data, country)
                
                if country_data.empty:
                    self.logger.warning(f"No data available for country {country}")
                    continue
                
                # Set up agent populations
                populations = self._setup_agent_populations(
                    country, country_data, population_scaling, wealth_params
                )
                
                # Set initial market conditions
                market_conds = self._setup_market_conditions(country, country_data)
                
                # Create industry-specific conditions
                industry_conds = self._setup_industry_conditions(country, country_data)
                
                # Create balance sheets
                balance_sheets = self._create_initial_balance_sheets(
                    country, populations, market_conds, country_data
                )
                
                initial_conditions[country] = {
                    "agent_populations": populations,
                    "market_conditions": market_conds,
                    "industry_conditions": industry_conds,
                    "balance_sheets": balance_sheets,
                    "metadata": {
                        "total_agents": sum(pop.count for pop in populations.values()),
                        "initialization_timestamp": operation_id
                    }
                }
                
                self.logger.info(f"Initialized {sum(pop.count for pop in populations.values())} agents for {country}")
            
            # Create metadata
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            result = CalibrationResult(
                status=CalibrationStatus.COMPLETED,
                data={"initial_conditions": initial_conditions},
                metadata=metadata
            )
            
            self.logger.info(f"Initial conditions set for {len(initial_conditions)} countries")
            return result
            
        except Exception as e:
            self.logger.error(f"Initial conditions setting failed: {e}")
            
            metadata = CalibrationMetadata(
                calibrator_name=self.name,
                operation_id=operation_id,
                parameters=parameters or {}
            )
            
            return CalibrationResult(
                status=CalibrationStatus.FAILED,
                data={},
                metadata=metadata,
                errors=[str(e)]
            )
    
    def _setup_agent_populations(
        self,
        country: str,
        data: pd.DataFrame,
        population_scaling: Dict[str, float],
        wealth_params: Dict[str, float]
    ) -> Dict[AgentType, AgentPopulation]:
        """Set up agent populations for a country."""
        populations = {}
        
        # Get GDP-based scaling factor
        gdp_scale = self._get_gdp_scaling_factor(data)
        
        for agent_type, base_count in self.DEFAULT_POPULATIONS.items():
            # Apply country-specific scaling
            country_scale = population_scaling.get(country, 1.0)
            
            # Calculate final population size
            if agent_type in [AgentType.GOVERNMENT, AgentType.CENTRAL_BANK]:
                # Always exactly 1 for these agents
                final_count = 1
            else:
                final_count = int(base_count * gdp_scale * country_scale)
                final_count = max(final_count, 10)  # Minimum population
            
            # Set up distribution parameters
            dist_params = self._get_distribution_parameters(agent_type, wealth_params)
            
            # Set up initial conditions
            initial_conds = self._get_agent_initial_conditions(agent_type, data)
            
            populations[agent_type] = AgentPopulation(
                agent_type=agent_type,
                count=final_count,
                distribution_params=dist_params,
                initial_conditions=initial_conds
            )
        
        return populations
    
    def _get_gdp_scaling_factor(self, data: pd.DataFrame) -> float:
        """Calculate GDP-based scaling factor."""
        if "gdp" not in data.columns or data["gdp"].empty:
            return 1.0
        
        # Use latest GDP value
        gdp = data["gdp"].dropna()
        if gdp.empty:
            return 1.0
        
        latest_gdp = gdp.iloc[-1]
        
        # Scale relative to US GDP (approximately $25 trillion)
        us_gdp_baseline = 25e12
        scaling_factor = np.sqrt(latest_gdp / us_gdp_baseline)  # Square root scaling
        
        # Bound the scaling factor
        return np.clip(scaling_factor, 0.1, 10.0)
    
    def _get_distribution_parameters(
        self,
        agent_type: AgentType,
        wealth_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get distribution parameters for agent type."""
        if agent_type == AgentType.HOUSEHOLD:
            return {
                "wealth_distribution": "pareto",
                "wealth_alpha": wealth_params.get("household_wealth_alpha", 1.16),
                "income_distribution": "lognormal",
                "income_sigma": 0.6
            }
        
        elif agent_type == AgentType.FIRM:
            return {
                "size_distribution": "pareto",
                "size_alpha": wealth_params.get("firm_size_alpha", 1.06),
                "productivity_distribution": "lognormal",
                "productivity_sigma": 0.3
            }
        
        elif agent_type == AgentType.BANK:
            return {
                "size_distribution": "pareto", 
                "size_alpha": wealth_params.get("bank_size_alpha", 1.2),
                "capital_ratio_mean": 0.12,
                "capital_ratio_std": 0.02
            }
        
        else:
            return {}
    
    def _get_agent_initial_conditions(
        self,
        agent_type: AgentType,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Get initial conditions for agent type."""
        conditions = {}
        
        if agent_type == AgentType.HOUSEHOLD:
            # Household initial conditions
            if "unemployment_rate" in data.columns:
                unemployment = data["unemployment_rate"].dropna()
                if not unemployment.empty:
                    conditions["employment_rate"] = 1.0 - unemployment.iloc[-1] / 100
            
            conditions.setdefault("employment_rate", 0.95)
            conditions["initial_consumption_level"] = 1.0
            conditions["initial_wealth_level"] = 1.0
        
        elif agent_type == AgentType.FIRM:
            # Firm initial conditions
            conditions["capacity_utilization"] = 0.8
            conditions["initial_price_level"] = 1.0
            conditions["initial_wage_level"] = 1.0
            
            # Set industry distribution
            conditions["industry_distribution"] = self._get_industry_distribution()
        
        elif agent_type == AgentType.BANK:
            # Bank initial conditions
            conditions["capital_adequacy_ratio"] = 0.12
            conditions["deposit_rate"] = 0.01
            conditions["lending_rate"] = 0.04
        
        elif agent_type == AgentType.GOVERNMENT:
            # Government initial conditions
            if "gdp" in data.columns:
                gdp = data["gdp"].dropna()
                if not gdp.empty:
                    conditions["debt_to_gdp_ratio"] = 0.6  # Default assumption
                    conditions["deficit_to_gdp_ratio"] = 0.03
        
        elif agent_type == AgentType.CENTRAL_BANK:
            # Central bank initial conditions
            if "interest_rate" in data.columns:
                rates = data["interest_rate"].dropna()
                if not rates.empty:
                    conditions["policy_rate"] = rates.iloc[-1] / 100
            
            conditions.setdefault("policy_rate", 0.02)
            conditions["inflation_target"] = 0.02
        
        return conditions
    
    def _get_industry_distribution(self) -> Dict[str, float]:
        """Get distribution of firms across industries."""
        # Simplified industry distribution based on typical developed economy
        total_industries = len(NACE2_INDUSTRIES)
        
        # Weight by typical industry shares
        industry_weights = {
            "A": 0.05,   # Agriculture
            "B": 0.02,   # Mining
            "C": 0.20,   # Manufacturing
            "D": 0.01,   # Utilities
            "E": 0.01,   # Water/waste
            "F": 0.08,   # Construction
            "G": 0.15,   # Wholesale/retail
            "H": 0.05,   # Transportation
            "I": 0.08,   # Accommodation
            "J": 0.05,   # Information
            "K": 0.08,   # Finance
            "L": 0.02,   # Real estate
            "M": 0.08,   # Professional
            "N": 0.03,   # Administrative
            "O": 0.05,   # Public admin
            "P": 0.06,   # Education
            "Q": 0.05,   # Health
            "R": 0.03    # Arts/entertainment
        }
        
        # Normalize to sum to 1
        total_weight = sum(industry_weights.values())
        return {k: v/total_weight for k, v in industry_weights.items()}
    
    def _setup_market_conditions(
        self,
        country: str,
        data: pd.DataFrame
    ) -> MarketConditions:
        """Set up initial market conditions."""
        conditions = MarketConditions()
        
        # Interest rate
        if "interest_rate" in data.columns:
            rates = data["interest_rate"].dropna()
            if not rates.empty:
                conditions.interest_rate = rates.iloc[-1] / 100
        
        # Unemployment rate
        if "unemployment_rate" in data.columns:
            unemployment = data["unemployment_rate"].dropna()
            if not unemployment.empty:
                conditions.unemployment_rate = unemployment.iloc[-1] / 100
        
        # Exchange rate (default to 1.0, would need bilateral rates for accuracy)
        conditions.exchange_rate = 1.0
        
        # Price and wage levels (normalized to 1.0)
        conditions.price_level = 1.0
        conditions.wage_level = 1.0
        
        # Capacity utilization
        conditions.capacity_utilization = 0.8
        
        return conditions
    
    def _setup_industry_conditions(
        self,
        country: str,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Set up industry-specific initial conditions."""
        industry_conditions = {}
        
        for industry in NACE2_INDUSTRIES:
            industry_conditions[industry.code] = {
                "capacity_utilization": 0.8,
                "employment_level": 1.0,
                "productivity_level": 1.0,
                "price_level": 1.0,
                "output_level": 1.0
            }
        
        return industry_conditions
    
    def _create_initial_balance_sheets(
        self,
        country: str,
        populations: Dict[AgentType, AgentPopulation],
        market_conditions: MarketConditions,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Create initial balance sheets for all agent types."""
        balance_sheets = {}
        
        # Get GDP for scaling
        gdp = self._get_latest_value(data, "gdp", 1e12)  # Default 1 trillion
        
        # Household balance sheets
        if AgentType.HOUSEHOLD in populations:
            household_pop = populations[AgentType.HOUSEHOLD]
            balance_sheets["households"] = self._create_household_balance_sheets(
                household_pop, gdp, market_conditions
            )
        
        # Firm balance sheets
        if AgentType.FIRM in populations:
            firm_pop = populations[AgentType.FIRM]
            balance_sheets["firms"] = self._create_firm_balance_sheets(
                firm_pop, gdp, market_conditions
            )
        
        # Bank balance sheets
        if AgentType.BANK in populations:
            bank_pop = populations[AgentType.BANK]
            balance_sheets["banks"] = self._create_bank_balance_sheets(
                bank_pop, gdp, market_conditions
            )
        
        # Government balance sheet
        if AgentType.GOVERNMENT in populations:
            balance_sheets["government"] = self._create_government_balance_sheet(
                gdp, market_conditions
            )
        
        # Central bank balance sheet
        if AgentType.CENTRAL_BANK in populations:
            balance_sheets["central_bank"] = self._create_central_bank_balance_sheet(
                gdp, market_conditions
            )
        
        return balance_sheets
    
    def _create_household_balance_sheets(
        self,
        population: AgentPopulation,
        gdp: float,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Create household balance sheets."""
        n_households = population.count
        alpha = population.distribution_params.get("wealth_alpha", 1.16)
        
        # Total household wealth (approximately 3-4x GDP)
        total_wealth = gdp * 3.5
        wealth_per_household = total_wealth / n_households
        
        # Generate Pareto-distributed wealth
        wealth_distribution = stats.pareto.rvs(alpha, size=n_households) * wealth_per_household
        
        # Generate income distribution (roughly 60% of GDP goes to households)
        total_income = gdp * 0.6
        income_per_household = total_income / n_households
        income_distribution = stats.lognorm.rvs(0.6, size=n_households) * income_per_household
        
        return {
            "count": n_households,
            "wealth_distribution": wealth_distribution,
            "income_distribution": income_distribution,
            "total_wealth": total_wealth,
            "total_income": total_income,
            "employment_rate": population.initial_conditions.get("employment_rate", 0.95)
        }
    
    def _create_firm_balance_sheets(
        self,
        population: AgentPopulation,
        gdp: float,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Create firm balance sheets."""
        n_firms = population.count
        alpha = population.distribution_params.get("size_alpha", 1.06)
        
        # Total firm assets (approximately 2x GDP)
        total_assets = gdp * 2.0
        assets_per_firm = total_assets / n_firms
        
        # Generate Pareto-distributed firm sizes
        size_distribution = stats.pareto.rvs(alpha, size=n_firms) * assets_per_firm
        
        # Industry distribution
        industry_dist = population.initial_conditions.get("industry_distribution", {})
        
        return {
            "count": n_firms,
            "size_distribution": size_distribution,
            "total_assets": total_assets,
            "industry_distribution": industry_dist,
            "capacity_utilization": market_conditions.capacity_utilization,
            "average_markup": 1.2
        }
    
    def _create_bank_balance_sheets(
        self,
        population: AgentPopulation,
        gdp: float,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Create bank balance sheets."""
        n_banks = population.count
        alpha = population.distribution_params.get("size_alpha", 1.2)
        
        # Total banking assets (approximately 1.5x GDP)
        total_assets = gdp * 1.5
        assets_per_bank = total_assets / n_banks
        
        # Generate bank size distribution
        size_distribution = stats.pareto.rvs(alpha, size=n_banks) * assets_per_bank
        
        # Capital ratios
        capital_ratio_mean = population.distribution_params.get("capital_ratio_mean", 0.12)
        capital_ratio_std = population.distribution_params.get("capital_ratio_std", 0.02)
        capital_ratios = np.random.normal(capital_ratio_mean, capital_ratio_std, n_banks)
        capital_ratios = np.clip(capital_ratios, 0.08, 0.20)  # Regulatory bounds
        
        return {
            "count": n_banks,
            "size_distribution": size_distribution,
            "total_assets": total_assets,
            "capital_ratios": capital_ratios,
            "deposit_rate": market_conditions.interest_rate - 0.01,
            "lending_rate": market_conditions.interest_rate + 0.02
        }
    
    def _create_government_balance_sheet(
        self,
        gdp: float,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Create government balance sheet."""
        return {
            "debt_to_gdp_ratio": 0.6,
            "deficit_to_gdp_ratio": 0.03,
            "total_debt": gdp * 0.6,
            "annual_deficit": gdp * 0.03,
            "tax_rate": 0.25,
            "spending_to_gdp_ratio": 0.4
        }
    
    def _create_central_bank_balance_sheet(
        self,
        gdp: float,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Create central bank balance sheet."""
        return {
            "policy_rate": market_conditions.interest_rate,
            "inflation_target": 0.02,
            "money_supply_to_gdp_ratio": 0.5,
            "foreign_reserves_to_gdp_ratio": 0.1,
            "total_money_supply": gdp * 0.5
        }
    
    def _get_latest_value(self, data: pd.DataFrame, column: str, default: float) -> float:
        """Get latest value from a time series column."""
        if column not in data.columns:
            return default
        
        series = data[column].dropna()
        if series.empty:
            return default
        
        return float(series.iloc[-1])
    
    def validate_initial_conditions(
        self,
        initial_conditions: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Validate initial conditions for economic consistency.
        
        Args:
            initial_conditions: Dictionary of initial conditions by country
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        for country, conditions in initial_conditions.items():
            # Check agent populations
            if "agent_populations" in conditions:
                populations = conditions["agent_populations"]
                
                # Check minimum populations
                for agent_type, population in populations.items():
                    if population.count < 1:
                        warnings.append(f"{country}: {agent_type.value} population is zero")
                
                # Check ratios
                if AgentType.HOUSEHOLD in populations and AgentType.FIRM in populations:
                    household_count = populations[AgentType.HOUSEHOLD].count
                    firm_count = populations[AgentType.FIRM].count
                    ratio = household_count / firm_count
                    
                    if ratio < 5 or ratio > 50:
                        warnings.append(
                            f"{country}: Household/firm ratio ({ratio:.1f}) is unusual"
                        )
            
            # Check balance sheet consistency
            if "balance_sheets" in conditions:
                balance_sheets = conditions["balance_sheets"]
                
                # Check wealth consistency
                if "households" in balance_sheets and "firms" in balance_sheets:
                    household_wealth = balance_sheets["households"].get("total_wealth", 0)
                    firm_assets = balance_sheets["firms"].get("total_assets", 0)
                    
                    if household_wealth + firm_assets == 0:
                        warnings.append(f"{country}: Total private wealth is zero")
        
        return warnings
    
    def export_initial_conditions(
        self,
        initial_conditions: Dict[str, Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export initial conditions to file.
        
        Args:
            initial_conditions: Initial conditions data
            output_path: Output file path
            format: Export format ('json', 'pickle', 'matlab')
        """
        import json
        import pickle
        from pathlib import Path
        
        output_path = Path(output_path)
        
        if format == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._convert_for_json(initial_conditions)
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(initial_conditions, f)
        
        elif format == "matlab":
            try:
                from scipy.io import savemat
                
                # Convert to MATLAB-compatible format
                matlab_data = self._convert_for_matlab(initial_conditions)
                savemat(output_path, matlab_data)
                
            except ImportError:
                self.logger.error("scipy is required for MATLAB export")
                raise
        
        self.logger.info(f"Initial conditions exported to {output_path}")
    
    def _convert_for_json(self, data: Any) -> Any:
        """Convert data for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._convert_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, AgentType):
            return data.value
        elif hasattr(data, '__dict__'):
            return self._convert_for_json(data.__dict__)
        else:
            return data
    
    def _convert_for_matlab(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert data for MATLAB export."""
        matlab_data = {}
        
        for country, conditions in data.items():
            country_data = {}
            
            # Convert agent populations
            if "agent_populations" in conditions:
                for agent_type, population in conditions["agent_populations"].items():
                    agent_key = f"{agent_type.value}_population"
                    country_data[agent_key] = {
                        "count": population.count,
                        "distribution_params": population.distribution_params,
                        "initial_conditions": population.initial_conditions
                    }
            
            # Convert balance sheets
            if "balance_sheets" in conditions:
                for sheet_type, sheet_data in conditions["balance_sheets"].items():
                    sheet_key = f"{sheet_type}_balance_sheet"
                    country_data[sheet_key] = sheet_data
            
            # Convert market conditions
            if "market_conditions" in conditions:
                market_conds = conditions["market_conditions"]
                country_data["market_conditions"] = {
                    "price_level": market_conds.price_level,
                    "wage_level": market_conds.wage_level,
                    "interest_rate": market_conds.interest_rate,
                    "exchange_rate": market_conds.exchange_rate,
                    "unemployment_rate": market_conds.unemployment_rate,
                    "capacity_utilization": market_conds.capacity_utilization
                }
            
            matlab_data[country] = country_data
        
        return matlab_data