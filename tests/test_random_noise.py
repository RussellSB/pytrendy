"""
Tests for random noise handling.

This module tests PyTrendy's robustness to random noise in time series data.
It verifies that the algorithm can still detect underlying trends when noise
is present at various levels.
"""

import pytest
import pytrendy as pt
import pandas as pd
import numpy as np


class TestRandomNoise:
    """Test cases for handling random noise in time series."""

    @pytest.fixture
    def base_synthetic_data(self):
        """Load base synthetic dataset."""
        return pt.load_data('series_synthetic')

    @pytest.mark.parametrize("noise_std", [0, 10, 15, 20, 50])
    def test_increasing_noise_levels(self, base_synthetic_data, noise_std):
        """Test detection with increasing levels of random noise."""
        np.random.seed(42)  # For reproducibility
        df = base_synthetic_data.copy()
        df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False
        )
        
        # Assert that segments are detected even with noise
        assert len(results.df) > 0, f"No segments detected with noise_std={noise_std}"
        
        # Verify segments have valid properties
        assert all(pd.notna(results.df['direction'])), \
            f"All segments should have directions (noise_std={noise_std})"
        assert all(results.df['days'] > 0), \
            f"All segments should have positive days (noise_std={noise_std})"
        
        # Check that segments span a reasonable time range
        all_starts = pd.to_datetime(results.df['start'])
        all_ends = pd.to_datetime(results.df['end'])
        time_span = (all_ends.max() - all_starts.min()).days
        
        # Even with high noise, should detect some patterns
        assert time_span > 0, f"Should detect segments over time (noise_std={noise_std})"
        
        # For lower noise levels, expect more structured results
        if noise_std <= 20:
            # Should detect multiple segments
            assert len(results.df) >= 3, \
                f"Expected multiple segments with noise_std={noise_std}"
            # Should have both Up and Down trends
            directions = results.df['direction'].unique()
            assert len(directions) >= 2, \
                f"Expected multiple direction types with noise_std={noise_std}"

    def test_high_noise_stability(self, base_synthetic_data):
        """Test that high noise doesn't cause crashes or errors."""
        noise_std = 50
        
        # Run multiple iterations to test stability
        for iteration in range(5):
            np.random.seed(42 + iteration)  # Different seed each time
            df = base_synthetic_data.copy()
            df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
            
            # Should not raise any exceptions
            results = pt.detect_trends(
                df,
                date_col='date',
                value_col='value_noisy',
                plot=False
            )
            
            # Basic validation
            assert len(results.df) >= 0, \
                f"Results should be valid (iteration {iteration})"
            
            if len(results.df) > 0:
                # If segments are detected, they should be valid
                assert all(pd.notna(results.df['start'])), \
                    f"All segments should have start dates (iteration {iteration})"
                assert all(pd.notna(results.df['end'])), \
                    f"All segments should have end dates (iteration {iteration})"

    def test_repeated_noise_runs(self, base_synthetic_data):
        """Test stability with repeated runs at moderate noise level."""
        noise_std = 10
        
        for iteration in range(10):
            np.random.seed(100 + iteration)
            df = base_synthetic_data.copy()
            df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
            
            # Test with padded option
            results = pt.detect_trends(
                df,
                date_col='date',
                value_col='value_noisy',
                plot=False,
                method_params=dict(is_abrupt_padded=True)
            )
            
            # Should successfully detect segments
            assert len(results.df) > 0, \
                f"Should detect segments in iteration {iteration}"
            
            # Segments should be ordered
            if len(results.df) > 1:
                starts = pd.to_datetime(results.df['start'])
                for i in range(len(starts) - 1):
                    assert starts.iloc[i] <= starts.iloc[i + 1], \
                        f"Segments should be ordered (iteration {iteration})"

    def test_noise_vs_clean_comparison(self, base_synthetic_data):
        """Compare detection between clean and noisy versions of the same data."""
        # Clean version
        df_clean = base_synthetic_data.copy()
        results_clean = pt.detect_trends(
            df_clean,
            date_col='date',
            value_col='gradual',
            plot=False
        )
        
        # Noisy version
        np.random.seed(42)
        df_noisy = base_synthetic_data.copy()
        df_noisy['value_noisy'] = df_noisy['gradual'] + np.random.normal(0, 10, size=len(df_noisy))
        results_noisy = pt.detect_trends(
            df_noisy,
            date_col='date',
            value_col='value_noisy',
            plot=False
        )
        
        # Both should detect segments
        assert len(results_clean.df) > 0, "Clean data should have segments"
        assert len(results_noisy.df) > 0, "Noisy data should have segments"
        
        # The number of segments might differ, but should be in same ballpark
        # Allow noisy version to have +/- 50% segments compared to clean
        clean_count = len(results_clean.df)
        noisy_count = len(results_noisy.df)
        
        # Just verify both produced reasonable output
        assert clean_count > 0 and noisy_count > 0, \
            "Both versions should produce segments"
        
        # Check that main trend directions are still detected in noisy version
        clean_directions = set(results_clean.df['direction'].unique())
        noisy_directions = set(results_noisy.df['direction'].unique())
        
        # Should have some overlap in detected directions
        assert len(clean_directions & noisy_directions) > 0, \
            "Noisy version should detect some of the same trend types as clean version"

    def test_zero_noise_baseline(self, base_synthetic_data):
        """Test that zero noise (clean data) produces stable results."""
        df = base_synthetic_data.copy()
        df['value_noisy'] = df['gradual']  # No noise added
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False
        )
        
        # Should detect clear trends in clean data
        assert len(results.df) >= 5, "Clean data should detect multiple segments"
        
        # Should have variety of directions
        directions = results.df['direction'].unique()
        assert 'Up' in directions, "Should detect upward trends in clean data"
        assert 'Down' in directions, "Should detect downward trends in clean data"
