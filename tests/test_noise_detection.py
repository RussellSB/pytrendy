"""
Tests for noise detection in trend detection algorithm.

These tests verify that the trend detection algorithm correctly identifies
noise segments based on the noise level in the data. When noise increases,
more and/or longer noise segments should be detected.
"""

import pytest
import numpy as np
import pandas as pd
import pytrendy as pt


class TestNoiseDetection:
    """Test cases for noise detection with different noise levels."""

    def test_increasing_noise_levels(self):
        """Test that increasing noise levels result in more/longer noise segments."""
        # Test with increasing noise levels using deterministic seed
        noise_levels = [0, 10, 20, 50]
        noise_segment_counts = []
        total_noise_lengths = []
        
        for noise_std in noise_levels:
            np.random.seed(42)  # Deterministic behavior
            df = pt.load_data('series_synthetic')
            df['value_noisy'] = df['gradual'] + np.random.normal(0, noise_std, size=len(df))
            results = pt.detect_trends(df, date_col='date', value_col='value_noisy', plot=False)
            
            # Count noise segments and total noise length
            noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
            noise_count = len(noise_segments)
            
            # Calculate total length of noise segments
            total_length = 0
            for seg in noise_segments:
                start = pd.to_datetime(seg['start'])
                end = pd.to_datetime(seg['end'])
                # Calculate days between start and end (inclusive)
                length = (end - start).days + 1
                total_length += length
            
            noise_segment_counts.append(noise_count)
            total_noise_lengths.append(total_length)
        
        # Verify that noise increases with noise level
        # At noise_std=0, there should be no noise segments
        assert noise_segment_counts[0] == 0, "No noise should be detected with zero noise"
        
        # At higher noise levels, we should see more noise detection
        # The total noise length should generally increase with noise level
        assert total_noise_lengths[3] > total_noise_lengths[1], \
            f"High noise (std=50) should have more total noise length than low noise (std=10). " \
            f"Got {total_noise_lengths[3]} vs {total_noise_lengths[1]}"

    @pytest.mark.core
    def test_high_noise_mostly_noise(self):
        """Test that high noise level (std=50) detects mostly noise segments."""
        np.random.seed(42)  # Deterministic behavior
        df = pt.load_data('series_synthetic')
        df['value_noisy'] = df['gradual'] + np.random.normal(0, 50, size=len(df))
        results = pt.detect_trends(df, date_col='date', value_col='value_noisy', plot=False)
        
        # Count noise segments and calculate total noise length
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        total_segments = len(results.segments)
        
        # Calculate total length of noise segments vs total data length
        total_noise_length = 0
        for seg in noise_segments:
            start = pd.to_datetime(seg['start'])
            end = pd.to_datetime(seg['end'])
            total_noise_length += (end - start).days + 1
        
        # Total data length (from first to last date)
        first_date = pd.to_datetime(results.segments[0]['start'])
        last_date = pd.to_datetime(results.segments[-1]['end'])
        total_data_length = (last_date - first_date).days + 1
        
        # With high noise (std=50), we expect a significant portion to be classified as noise
        # At least 50% of the data should be noise
        noise_percentage = (total_noise_length / total_data_length) * 100
        assert noise_percentage >= 50, \
            f"High noise (std=50) should detect at least 50% of data as noise. " \
            f"Got {noise_percentage:.1f}%"

    def test_gradual_noisy_20_column(self):
        """Test noise detection using the gradual-noisy-20 column from synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, date_col='date', value_col='gradual-noisy-20', plot=False)
        
        # Count noise segments
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        noise_count = len(noise_segments)
        
        # The gradual-noisy-20 column should have some noise segments
        assert noise_count > 0, "gradual-noisy-20 column should detect some noise segments"
        
        # Verify that we have a mix of segment types (not all noise)
        segment_types = set(seg['direction'] for seg in results.segments)
        assert len(segment_types) > 1, \
            "gradual-noisy-20 should detect multiple segment types, not just noise"
        assert 'Noise' in segment_types, \
            "gradual-noisy-20 should detect noise segments"

    def test_noise_segment_length_correlation(self):
        """Test that higher noise levels produce longer individual noise segments."""
        # Test with two different noise levels
        np.random.seed(123)  # Different seed for variety
        
        # Low noise
        df_low = pt.load_data('series_synthetic')
        df_low['value_noisy'] = df_low['gradual'] + np.random.normal(0, 10, size=len(df_low))
        results_low = pt.detect_trends(df_low, date_col='date', value_col='value_noisy', plot=False)
        
        noise_segments_low = [seg for seg in results_low.segments if seg['direction'] == 'Noise']
        avg_length_low = 0
        if noise_segments_low:
            total_length_low = sum((pd.to_datetime(seg['end']) - pd.to_datetime(seg['start'])).days + 1 for seg in noise_segments_low)
            avg_length_low = total_length_low / len(noise_segments_low)
        
        # High noise
        np.random.seed(123)  # Same seed for fair comparison
        df_high = pt.load_data('series_synthetic')
        df_high['value_noisy'] = df_high['gradual'] + np.random.normal(0, 50, size=len(df_high))
        results_high = pt.detect_trends(df_high, date_col='date', value_col='value_noisy', plot=False)
        
        noise_segments_high = [seg for seg in results_high.segments if seg['direction'] == 'Noise']
        avg_length_high = 0
        if noise_segments_high:
            total_length_high = sum((pd.to_datetime(seg['end']) - pd.to_datetime(seg['start'])).days + 1 for seg in noise_segments_high)
            avg_length_high = total_length_high / len(noise_segments_high)
        
        # With high noise, we expect longer noise segments on average or more total noise
        # Either average length should be longer OR there should be more noise segments
        assert (avg_length_high > avg_length_low) or (len(noise_segments_high) > len(noise_segments_low)), \
            f"High noise should produce longer or more noise segments. " \
            f"Low noise: {len(noise_segments_low)} segments, avg {avg_length_low:.1f} days. " \
            f"High noise: {len(noise_segments_high)} segments, avg {avg_length_high:.1f} days."
