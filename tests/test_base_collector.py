"""
Unit tests for base collector
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.collectors.base_collector import BaseCollector


class MockCollector(BaseCollector):
    """Mock collector for testing"""
    
    def fetch(self, **kwargs):
        """Return mock data"""
        return pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='MS'),
            'value': range(10),
            'category': ['A'] * 10
        })
    
    def validate(self, df):
        """Simple validation"""
        return not df.empty


def test_collector_initialization():
    """Test collector initialization"""
    collector = MockCollector(
        collector_name='TestCollector',
        data_source='test',
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    
    assert collector.collector_name == 'TestCollector'
    assert collector.data_source == 'test'
    assert collector.start_date == '2020-01-01'
    assert collector.end_date == '2020-12-31'


def test_collect_workflow():
    """Test the collect workflow"""
    collector = MockCollector(
        collector_name='TestCollector',
        data_source='test'
    )
    
    df = collector.collect(save_data=False, validate_data=True)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert 'date' in df.columns
    assert 'value' in df.columns


def test_date_range():
    """Test date range retrieval"""
    collector = MockCollector(
        collector_name='TestCollector',
        data_source='test',
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    
    date_range = collector.get_date_range()
    assert date_range['start'] == '2020-01-01'
    assert date_range['end'] == '2020-12-31'


if __name__ == '__main__':
    pytest.main([__file__])
