"""
TODO Add description here
"""
import pytest
import pytrendy as pt
import pandas as pd
import numpy as np
from conftest import assert_segments_match


class TestNonDateCases:
    """Test cases where non-date indexes are used"""

    @pytest.mark.core
    def test_integer_index(self):
        """Test standard gradual trend but with no date index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Down', 'start': 24, 'end': 35},
            {'direction': 'Flat', 'start': 36, 'end': 39},
            {'direction': 'Up', 'start': 40, 'end': 72},
            {'direction': 'Flat', 'start': 73, 'end': 75},
            {'direction': 'Down', 'start': 76, 'end': 90},
            {'direction': 'Up', 'start': 91, 'end': 127},
            {'direction': 'Down', 'start': 128, 'end': 167},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ]

        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_float_index(self):
        """Test standard gradual trend but with float lookup."""
        df = pt.load_data('series_synthetic')
        df['float_lookup'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='float_lookup',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up',   'start': 0.005556, 'end': 0.127778},
            {'direction': 'Down', 'start': 0.133333, 'end': 0.194444},
            {'direction': 'Flat', 'start': 0.200000, 'end': 0.216667},
            {'direction': 'Up',   'start': 0.222222, 'end': 0.400000},
            {'direction': 'Flat', 'start': 0.405556, 'end': 0.416667},
            {'direction': 'Down', 'start': 0.422222, 'end': 0.500000},
            {'direction': 'Up',   'start': 0.505556, 'end': 0.705556},
            {'direction': 'Down', 'start': 0.711111, 'end': 0.927778},
            {'direction': 'Flat', 'start': 0.933333, 'end': 1.000000},
        ]

        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_string_index(self):
        """Test standard gradual trend but with string lookup."""
        df = pt.load_data('series_synthetic')
        df['string_lookup'] = [f"Step {i}" for i in range(len(df))]
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='string_lookup',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up',   'start': 'Step 1',   'end': 'Step 23'},
            {'direction': 'Down', 'start': 'Step 24',  'end': 'Step 35'},
            {'direction': 'Flat', 'start': 'Step 36',  'end': 'Step 39'},
            {'direction': 'Up',   'start': 'Step 40',  'end': 'Step 72'},
            {'direction': 'Flat', 'start': 'Step 73',  'end': 'Step 75'},
            {'direction': 'Down', 'start': 'Step 76',  'end': 'Step 90'},
            {'direction': 'Up',   'start': 'Step 91',  'end': 'Step 127'},
            {'direction': 'Down', 'start': 'Step 128', 'end': 'Step 167'},
            {'direction': 'Flat', 'start': 'Step 168', 'end': 'Step 180'},
        ]

        assert_segments_match(results.segments, expected_segments)