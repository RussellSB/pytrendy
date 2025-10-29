"""
Tests for original baseline test cases from the mock test file.

These tests verify that the original test cases produce the expected segments
with correct start dates, end dates, and directions.
"""

import pytest
import pytrendy as pt
import pandas as pd


class TestOriginalCases:
    """Test cases for original baseline synthetic data."""

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
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-29'},
        ]
        
        # Assert number of segments matches
        assert len(results.segments) == len(expected_segments), \
            f"Expected {len(expected_segments)} segments, got {len(results.segments)}"
        
        # Assert each segment matches expected values
        for i, (actual, expected) in enumerate(zip(results.segments, expected_segments)):
            # Normalize dates for comparison
            actual_start = pd.to_datetime(actual['start']).strftime('%Y-%m-%d')
            actual_end = pd.to_datetime(actual['end']).strftime('%Y-%m-%d')
            
            assert actual['direction'] == expected['direction'], \
                f"Segment {i+1}: Expected direction '{expected['direction']}', got '{actual['direction']}'"
            assert actual_start == expected['start'], \
                f"Segment {i+1}: Expected start '{expected['start']}', got '{actual_start}'"
            assert actual_end == expected['end'], \
                f"Segment {i+1}: Expected end '{expected['end']}', got '{actual_end}'"

    def test_abrupt_trends_no_padding(self):
        """Test detection of abrupt trends without padding."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Flat', 'start': '2025-01-01', 'end': '2025-02-27'},
            {'direction': 'Up', 'start': '2025-02-28', 'end': '2025-03-01'},
            {'direction': 'Flat', 'start': '2025-03-02', 'end': '2025-05-01'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-05-05'},
            {'direction': 'Flat', 'start': '2025-05-06', 'end': '2025-06-29'},
        ]
        
        # Assert number of segments matches
        assert len(results.segments) == len(expected_segments), \
            f"Expected {len(expected_segments)} segments, got {len(results.segments)}"
        
        # Assert each segment matches expected values
        for i, (actual, expected) in enumerate(zip(results.segments, expected_segments)):
            # Normalize dates for comparison
            actual_start = pd.to_datetime(actual['start']).strftime('%Y-%m-%d')
            actual_end = pd.to_datetime(actual['end']).strftime('%Y-%m-%d')
            
            assert actual['direction'] == expected['direction'], \
                f"Segment {i+1}: Expected direction '{expected['direction']}', got '{actual['direction']}'"
            assert actual_start == expected['start'], \
                f"Segment {i+1}: Expected start '{expected['start']}', got '{actual_start}'"
            assert actual_end == expected['end'], \
                f"Segment {i+1}: Expected end '{expected['end']}', got '{actual_end}'"

    def test_abrupt_trends_with_padding(self):
        """Test detection of abrupt trends with padding enabled."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Flat', 'start': '2025-01-01', 'end': '2025-02-27'},
            {'direction': 'Up', 'start': '2025-02-28', 'end': '2025-03-29'},
            {'direction': 'Flat', 'start': '2025-03-30', 'end': '2025-05-01'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-06-02'},
            {'direction': 'Flat', 'start': '2025-06-03', 'end': '2025-06-29'},
        ]
        
        # Assert number of segments matches
        assert len(results.segments) == len(expected_segments), \
            f"Expected {len(expected_segments)} segments, got {len(results.segments)}"
        
        # Assert each segment matches expected values
        for i, (actual, expected) in enumerate(zip(results.segments, expected_segments)):
            # Normalize dates for comparison
            actual_start = pd.to_datetime(actual['start']).strftime('%Y-%m-%d')
            actual_end = pd.to_datetime(actual['end']).strftime('%Y-%m-%d')
            
            assert actual['direction'] == expected['direction'], \
                f"Segment {i+1}: Expected direction '{expected['direction']}', got '{actual['direction']}'"
            assert actual_start == expected['start'], \
                f"Segment {i+1}: Expected start '{expected['start']}', got '{actual_start}'"
            assert actual_end == expected['end'], \
                f"Segment {i+1}: Expected end '{expected['end']}', got '{actual_end}'"
