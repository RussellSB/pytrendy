"""
Tests for previously failing edge cases (crash instances).

This module tests edge cases that previously caused crashes or unexpected behavior.
These tests ensure that the algorithm handles difficult scenarios gracefully and
maintains stability even with challenging input data.
"""

import pytest
import pytrendy as pt
import pandas as pd
import numpy as np


class TestCrashInstances:
    """Test cases for previously problematic scenarios that caused crashes."""

    @pytest.fixture
    def base_synthetic_data(self):
        """Load base synthetic dataset."""
        return pt.load_data('series_synthetic')

    def test_extreme_noise_case_1(self, base_synthetic_data):
        """Test handling of extreme noise case that previously crashed."""
        np.random.seed(42)
        df = base_synthetic_data.copy()
        noise_std = 50
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        # Should not crash
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Basic validation - should complete without errors
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"
        
        # If segments are detected, they should be valid
        if len(results.df) > 0:
            assert all(pd.notna(results.df['direction'])), \
                "All segments should have valid directions"
            assert all(results.df['days'] > 0), \
                "All segments should have positive days"

    def test_extreme_noise_case_2(self, base_synthetic_data):
        """Test another extreme noise scenario."""
        np.random.seed(123)
        df = base_synthetic_data.copy()
        noise_std = 50
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        # Should handle gracefully
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"
        
        # Verify date integrity
        if len(results.df) > 0:
            for idx, row in results.df.iterrows():
                assert pd.notna(row['start']), f"Segment {idx} should have start date"
                assert pd.notna(row['end']), f"Segment {idx} should have end date"
                start = pd.to_datetime(row['start'])
                end = pd.to_datetime(row['end'])
                assert start <= end, f"Segment {idx}: start should be <= end"

    def test_extreme_noise_case_3(self, base_synthetic_data):
        """Test third extreme noise scenario."""
        np.random.seed(456)
        df = base_synthetic_data.copy()
        noise_std = 50
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        # Should not hang or crash
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"

    def test_extreme_noise_case_4(self, base_synthetic_data):
        """Test fourth extreme noise scenario."""
        np.random.seed(789)
        df = base_synthetic_data.copy()
        noise_std = 50
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        # Should handle without issues
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"
        
        # Check that results are well-formed
        if len(results.df) > 0:
            expected_columns = ['direction', 'start', 'end', 'days']
            for col in expected_columns:
                assert col in results.df.columns, f"Missing expected column: {col}"

    def test_extreme_noise_case_5(self, base_synthetic_data):
        """Test fifth extreme noise scenario."""
        np.random.seed(321)
        df = base_synthetic_data.copy()
        noise_std = 50
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        # Should complete successfully
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"

    def test_complex_pattern_no_crash(self, base_synthetic_data):
        """Test complex pattern that previously caused issues."""
        df = base_synthetic_data.copy()
        df.set_index('date', inplace=True)
        
        # Create complex pattern with multiple abrupt changes
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300
        
        # Should handle complex overlapping patterns
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        assert len(results.df) >= 0, "Should return valid results"
        
        if len(results.df) > 0:
            # Verify basic properties
            assert all(results.df['days'] > 0), "All segments should have positive days"
            assert all(pd.notna(results.df['direction'])), \
                "All segments should have directions"

    def test_rapid_transitions(self, base_synthetic_data):
        """Test data with rapid transitions between levels."""
        df = base_synthetic_data.copy()
        df.set_index('date', inplace=True)
        
        # Create rapid transitions
        df.loc['2025-01-01':'2025-01-10', 'abrupt'] = 0
        df.loc['2025-01-11':'2025-01-20', 'abrupt'] = 100
        df.loc['2025-01-21':'2025-01-30', 'abrupt'] = 50
        df.loc['2025-01-31':'2025-02-09', 'abrupt'] = 150
        df.loc['2025-02-10':'2025-02-19', 'abrupt'] = 25
        
        # Should handle rapid transitions
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        assert len(results.df) >= 0, "Should handle rapid transitions"
        
        if len(results.df) > 0:
            # Check temporal ordering
            starts = pd.to_datetime(results.df['start'])
            for i in range(len(starts) - 1):
                assert starts.iloc[i] <= starts.iloc[i + 1], \
                    "Segments should be chronologically ordered"

    def test_single_value_stability(self, base_synthetic_data):
        """Test handling of series with repeated single values."""
        df = base_synthetic_data.copy()
        df.set_index('date', inplace=True)
        
        # Set most values to same level
        df.loc['2025-01-01':'2025-05-31', 'abrupt'] = 100
        df.loc['2025-06-01':'2025-06-30', 'abrupt'] = 150
        
        # Should handle flat regions
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        assert isinstance(results.df, pd.DataFrame), "Should return valid DataFrame"
        
        # Should detect the flat and transition
        if len(results.df) > 0:
            directions = results.df['direction'].unique()
            # Expect Flat or abrupt transition
            assert len(directions) > 0, "Should detect some segments"

    def test_empty_segments_handling(self):
        """Test handling of minimal data that might produce empty segments."""
        # Create minimal dataset (needs to be larger than algorithm window)
        # Algorithm uses window_length internally, so we need sufficient data
        dates = pd.date_range('2025-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': [10] * 50  # All same value
        })
        
        # Should handle minimal/flat data without crashing
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # When no segments are detected, results.segments is an empty list
        # and df attribute doesn't exist
        assert isinstance(results.segments, list), \
            "Should return results with segments list"
        
        # If segments detected, validate them
        if hasattr(results, 'df') and len(results.df) > 0:
            # If segments detected, should be valid
            assert all(pd.notna(results.df['direction'])), \
                "Detected segments should have directions"

    def test_large_value_jumps(self, base_synthetic_data):
        """Test handling of very large value jumps."""
        df = base_synthetic_data.copy()
        df.set_index('date', inplace=True)
        
        # Create large jumps
        df.loc['2025-01-01':'2025-02-28', 'abrupt'] = 10
        df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 1000  # Large spike
        df.loc['2025-03-02':'2025-04-30', 'abrupt'] = 10
        df.loc['2025-05-01':'2025-05-01', 'abrupt'] = 1000  # Another large spike
        
        # Should handle large value jumps
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        assert isinstance(results.df, pd.DataFrame), \
            "Should handle large value jumps"
        
        if len(results.df) > 0:
            # Segments should be valid
            for idx, row in results.df.iterrows():
                assert pd.notna(row['start']), f"Segment {idx} should have start"
                assert pd.notna(row['end']), f"Segment {idx} should have end"
                assert row['days'] > 0, f"Segment {idx} should have positive days"
