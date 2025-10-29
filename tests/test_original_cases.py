"""
Tests for original baseline test cases.

This module tests PyTrendy's core functionality on the original synthetic dataset
with gradual and abrupt trends. These tests serve as baseline verification for
the trend detection algorithm.
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
        
        # Assert that segments are detected
        assert len(results.df) > 0, "No segments detected for gradual trends"
        
        # Check for presence of expected trend directions
        directions = results.df['direction'].value_counts()
        assert 'Up' in directions.index, "Expected upward trends in gradual data"
        assert 'Down' in directions.index, "Expected downward trends in gradual data"
        
        # Verify that gradual trends are classified correctly
        non_flat_segments = results.df[results.df['direction'].isin(['Up', 'Down'])]
        if len(non_flat_segments) > 0 and 'trend_class' in results.df.columns:
            # Check that some segments are classified as gradual
            gradual_count = (non_flat_segments['trend_class'] == 'gradual').sum()
            assert gradual_count > 0, "Expected some gradual trend classifications"
        
        # Verify segments have valid date ranges
        for idx, row in results.df.iterrows():
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])
            assert start <= end, f"Segment {idx}: start date should be <= end date"
            
        # Check that segments have positive days count
        assert all(results.df['days'] > 0), "All segments should have positive duration"

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
        
        # Assert that segments are detected
        assert len(results.df) > 0, "No segments detected for abrupt trends"
        
        # Check for presence of trends
        directions = results.df['direction'].unique()
        assert len(directions) > 0, "Expected trend directions to be detected"
        
        # Verify that we detect multiple segments (abrupt data has clear transitions)
        assert len(results.df) >= 3, "Expected at least 3 segments in abrupt data"
        
        # Check that segments cover reasonable time span
        all_starts = pd.to_datetime(results.df['start'])
        all_ends = pd.to_datetime(results.df['end'])
        
        time_span = (all_ends.max() - all_starts.min()).days
        assert time_span > 30, "Detected segments should span more than 30 days"
        
        # Verify segments don't overlap
        for i in range(len(results.df) - 1):
            current_end = pd.to_datetime(results.df.iloc[i]['end'])
            next_start = pd.to_datetime(results.df.iloc[i + 1]['start'])
            # Allow for same-day or next-day transitions
            assert (next_start - current_end).days >= 0, \
                f"Segments {i} and {i+1} should not overlap"

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
        
        # Assert that segments are detected
        assert len(results.df) > 0, "No segments detected for abrupt trends with padding"
        
        # Check for presence of various trend types
        directions = results.df['direction'].unique()
        assert len(directions) > 0, "Expected trend directions to be detected"
        
        # Verify all required columns are present
        expected_columns = ['direction', 'start', 'end', 'days', 'change_rank']
        for col in expected_columns:
            assert col in results.df.columns, f"Expected column '{col}' in results"
        
        # Check that start and end dates are valid
        assert all(pd.notna(results.df['start'])), "All segments should have start dates"
        assert all(pd.notna(results.df['end'])), "All segments should have end dates"
        
        # Verify that segments are ordered chronologically
        starts = pd.to_datetime(results.df['start'])
        for i in range(len(starts) - 1):
            assert starts.iloc[i] <= starts.iloc[i + 1], \
                "Segments should be ordered by start date"

    def test_segment_properties(self):
        """Test that detected segments have expected properties."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False
        )
        
        # Check that all segments have required properties
        for idx, row in results.df.iterrows():
            # Direction should be one of the expected values
            assert row['direction'] in ['Up', 'Down', 'Flat', 'Noise'], \
                f"Segment {idx} has invalid direction: {row['direction']}"
            
            # Days should be positive
            assert row['days'] > 0, f"Segment {idx} should have positive days"
            
            # Date range should be valid
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])
            days_diff = (end - start).days
            # Days might be calculated differently (inclusive/exclusive), allow some tolerance
            assert abs(row['days'] - days_diff) <= 2, \
                f"Segment {idx}: days mismatch ({row['days']} vs {days_diff})"
            
            # Change rank should exist (even if NaN for Flat segments)
            assert 'change_rank' in row.index, \
                f"Segment {idx} should have change_rank column"
            
            # SNR should be present
            assert 'SNR' in row.index, f"Segment {idx} should have SNR column"
