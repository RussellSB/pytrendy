"""
TODO Add description here
"""
import pytest
import pytrendy as pt
import pandas as pd
from conftest import assert_segments_match


class TestNonDateCases:
    """TODO Update Docstrings"""

    @pytest.mark.core
    def test_gradual_trends(self):
        """Test detection of gradual trends in synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
            {'direction': 'Down', 'start': '2025-01-25', 'end': '2025-02-05'},
            {'direction': 'Flat', 'start': '2025-02-06', 'end': '2025-02-09'},
            {'direction': 'Up', 'start': '2025-02-10', 'end': '2025-03-14'},
            {'direction': 'Flat', 'start': '2025-03-15', 'end': '2025-03-17'},
            {'direction': 'Down', 'start': '2025-03-18', 'end': '2025-04-01'},
            {'direction': 'Up', 'start': '2025-04-02', 'end': '2025-05-08'},
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-17'},
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-30'},
        ]

        assert_segments_match(results.segments, expected_segments)