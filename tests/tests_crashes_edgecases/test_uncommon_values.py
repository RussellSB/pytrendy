"""
Tests for edge case scenarios caused by unusual data in trend detection algorithm.

These tests verify that the trend detection algorithm handles situations where the data is doing something 
we wouldn't typically expect.

Reference: tests/tests_crashes_edgecases/data/TESTDATA.md
"""

import pandas as pd
import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestUncommonValues:
    """Test cases for Scenarios with Uncommon or Unusual Values"""

    def test_low_value_series(self):
        """Test that algorithm handles a low/normalised value series (data in range [0, 1]) reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/low_value_series.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='trend',
            plot=False,
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2000-01-02', 'end': '2000-01-14'},
            {'direction': 'Flat', 'start': '2000-01-15', 'end': '2000-01-18'},
            {'direction': 'Down', 'start': '2000-01-19', 'end': '2000-03-17'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)