"""
Unit tests for core data models.
"""

import pytest
from datetime import datetime

from macro_abm_calibration.models import (
    Country, Industry, TimeFrame, CurrencyCode, 
    IndustryClassification, FrequencyType
)


class TestCountry:
    """Test Country model."""
    
    def test_valid_country_creation(self):
        """Test creating a valid country."""
        country = Country(
            oecd_code="USA",
            eurostat_code="USD", 
            name="United States",
            currency=CurrencyCode.USD
        )
        
        assert country.oecd_code == "USA"
        assert country.eurostat_code == "USD"
        assert country.currency == CurrencyCode.USD
        assert country.is_oecd_member is True
        assert country.is_eurozone is False
    
    def test_eurozone_country(self):
        """Test creating a eurozone country."""
        country = Country(
            oecd_code="DEU",
            eurostat_code="EUR",
            name="Germany", 
            currency=CurrencyCode.EUR,
            is_eurozone=True
        )
        
        assert country.is_eurozone is True
        assert country.currency == CurrencyCode.EUR
    
    def test_invalid_oecd_code_length(self):
        """Test invalid OECD code length raises error."""
        with pytest.raises(ValueError, match="OECD code must be 3 characters"):
            Country(
                oecd_code="US",  # Too short
                eurostat_code="USD",
                name="United States",
                currency=CurrencyCode.USD
            )
    
    def test_eurozone_currency_mismatch(self):
        """Test eurozone country with non-EUR currency raises error.""" 
        with pytest.raises(ValueError, match="Eurozone country must use EUR"):
            Country(
                oecd_code="DEU",
                eurostat_code="EUR",
                name="Germany",
                currency=CurrencyCode.USD,  # Wrong currency
                is_eurozone=True
            )


class TestIndustry:
    """Test Industry model."""
    
    def test_valid_industry_creation(self):
        """Test creating a valid industry."""
        industry = Industry(
            code="C",
            name="Manufacturing",
            classification=IndustryClassification.NACE2,
            description="Manufacturing of goods"
        )
        
        assert industry.code == "C"
        assert industry.name == "Manufacturing"
        assert industry.classification == IndustryClassification.NACE2
        assert industry.description == "Manufacturing of goods"
    
    def test_empty_code_raises_error(self):
        """Test empty industry code raises error."""
        with pytest.raises(ValueError, match="Industry code cannot be empty"):
            Industry(
                code="",
                name="Empty",
                classification=IndustryClassification.NACE2
            )


class TestTimeFrame:
    """Test TimeFrame model."""
    
    def test_valid_timeframe_creation(self):
        """Test creating a valid timeframe."""
        tf = TimeFrame(
            start_year=2010,
            end_year=2020,
            frequency=FrequencyType.ANNUAL
        )
        
        assert tf.start_year == 2010
        assert tf.end_year == 2020
        assert tf.frequency == FrequencyType.ANNUAL
        assert tf.num_years == 11
        assert tf.num_periods == 11
    
    def test_quarterly_periods(self):
        """Test quarterly period calculation."""
        tf = TimeFrame(
            start_year=2010,
            end_year=2012, 
            frequency=FrequencyType.QUARTERLY
        )
        
        assert tf.num_years == 3
        assert tf.num_periods == 12  # 3 years * 4 quarters
    
    def test_monthly_periods(self):
        """Test monthly period calculation."""
        tf = TimeFrame(
            start_year=2010,
            end_year=2011,
            frequency=FrequencyType.MONTHLY
        )
        
        assert tf.num_years == 2
        assert tf.num_periods == 24  # 2 years * 12 months
    
    def test_end_before_start_raises_error(self):
        """Test end year before start year raises error."""
        with pytest.raises(ValueError, match="end_year must be >= start_year"):
            TimeFrame(
                start_year=2020,
                end_year=2010  # Invalid
            )
    
    def test_year_range_property(self):
        """Test years property returns correct range."""
        tf = TimeFrame(start_year=2010, end_year=2012)
        years = list(tf.years)
        assert years == [2010, 2011, 2012]
    
    def test_to_datetime_index_annual(self):
        """Test conversion to annual datetime index."""
        tf = TimeFrame(
            start_year=2010,
            end_year=2012,
            frequency=FrequencyType.ANNUAL
        )
        
        index = tf.to_datetime_index()
        assert len(index) == 3
        assert index.freq.name == "A-DEC"
    
    def test_to_datetime_index_quarterly(self):
        """Test conversion to quarterly datetime index."""
        tf = TimeFrame(
            start_year=2010,
            end_year=2010,
            frequency=FrequencyType.QUARTERLY
        )
        
        index = tf.to_datetime_index()
        assert len(index) == 4
        assert index.freq.name == "Q-DEC"