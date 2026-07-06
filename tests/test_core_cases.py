"""
Tests for core logic of trend detection on synthetic data.

These tests verify that the trend detection algorithm produces consistent
results for gradual and abrupt trends, validating segment boundaries and
directions against expected behaviour.
"""

import pytest
import pytrendy as pt
import pandas as pd
from conftest import assert_segments_match


class TestCoreCases:
    """Test cases for core logic on synthetic data."""

    @pytest.mark.core
    def test_gradual_trends(self):
        """Test detection of gradual trends in synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False
        )
        
        # Expected segments based on current behaviour
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

    @pytest.mark.core
    def test_abrupt_trends_no_padding(self):
        """Test detection of abrupt trends without padding."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        # Expected segments based on current behaviour
        expected_segments = [
            {'direction': 'Flat', 'start': '2025-01-01', 'end': '2025-02-27'},
            {'direction': 'Up', 'start': '2025-02-28', 'end': '2025-03-01'},
            {'direction': 'Flat', 'start': '2025-03-02', 'end': '2025-05-01'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-05-05'},
            {'direction': 'Flat', 'start': '2025-05-06', 'end': '2025-06-30'},
        ]
        
        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_abrupt_trends_with_padding(self):
        """Test detection of abrupt trends with padding enabled."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )
        # Expected segments based on current behaviour
        expected_segments = [
            {'direction': 'Flat', 'start': '2025-01-01', 'end': '2025-02-27'},
            {'direction': 'Up', 'start': '2025-02-28', 'end': '2025-03-29'},
            {'direction': 'Flat', 'start': '2025-03-30', 'end': '2025-05-01'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-06-02'},
            {'direction': 'Flat', 'start': '2025-06-03', 'end': '2025-06-30'},
        ]
        
        assert_segments_match(results.segments, expected_segments)
