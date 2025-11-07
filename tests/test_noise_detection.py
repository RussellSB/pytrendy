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

    @pytest.mark.core
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
        
        # Verify each higher noise level has more noise than the previous
        for i in range(1, len(noise_levels)):
            assert total_noise_lengths[i] >= total_noise_lengths[i-1], \
                f"Noise at std={noise_levels[i]} should be >= noise at std={noise_levels[i-1]}. " \
                f"Got {total_noise_lengths[i]} vs {total_noise_lengths[i-1]} days"

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
        # At least 70% of the data should be noise
        noise_percentage = (total_noise_length / total_data_length) * 100
        assert noise_percentage >= 70, \
            f"High noise (std=50) should detect at least 70% of data as noise. " \
            f"Got {noise_percentage:.1f}%"

    def test_gradual_noisy_20_column(self):
        """Test noise detection using the gradual-noisy-20 column from synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, date_col='date', value_col='gradual-noisy-20', plot=False)
        
        # Count different segment types
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        up_segments = [seg for seg in results.segments if seg['direction'] == 'Up']
        down_segments = [seg for seg in results.segments if seg['direction'] == 'Down']
        flat_segments = [seg for seg in results.segments if seg['direction'] == 'Flat']
        
        # The gradual-noisy-20 column should have some noise segments
        assert len(noise_segments) > 0, "gradual-noisy-20 column should detect some noise segments"
        
        # Verify that we have a mix of segment types (not all noise)
        segment_types = set(seg['direction'] for seg in results.segments)
        assert len(segment_types) > 1, \
            "gradual-noisy-20 should detect multiple segment types, not just noise"
        assert 'Noise' in segment_types, \
            "gradual-noisy-20 should detect noise segments"
        
        # Verify that actual trend segments (Up/Down) are detected despite noise
        trend_segments = up_segments + down_segments
        assert len(trend_segments) > 0, \
            "gradual-noisy-20 should detect at least some Up or Down trend segments"
        
        # Calculate noise percentage
        total_noise_length = sum((pd.to_datetime(seg['end']) - pd.to_datetime(seg['start'])).days + 1 
                                  for seg in noise_segments)
        first_date = pd.to_datetime(results.segments[0]['start'])
        last_date = pd.to_datetime(results.segments[-1]['end'])
        total_data_length = (last_date - first_date).days + 1
        noise_percentage = (total_noise_length / total_data_length) * 100
        
        # Noise should be significant but not overwhelming (between 30% and 70%)
        assert 30 <= noise_percentage <= 70, \
            f"gradual-noisy-20 noise percentage should be between 30% and 70%. Got {noise_percentage:.1f}%"
        
        # Verify we have a reasonable mix of flat segments too
        assert len(flat_segments) > 0, \
            "gradual-noisy-20 should detect some flat segments"
