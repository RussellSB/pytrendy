"""
Tests for noise spike scenarios with gradual trends.

These tests verify that the trend detection algorithm correctly handles
noise spikes (single-point or short-duration anomalies) in time series
with gradual trend changes. Each test focuses on the location and impact
of spikes on trend detection.

Reference: tests/test.py lines 133-210 (Graduals and Spikes section)

Note: This test file was extracted from exploratory tests in test.py to
provide formal test coverage for noise spike detection in gradual trends.
"""

import pytest
import pytrendy as pt
from conftest import assert_segments_match


class TestNoiseSpikesGradual:
    """Test cases for noise spike detection with gradual trends."""

    @pytest.mark.core
    def test_gradual_spike_single_mid_series(self):
        """
        Test gradual trends with one spike in the middle of the series.
        
        Reference: test.py spike test 0.1, lines 136-142
        Spike location: 2025-03-25 (value=200)
        
        This test verifies that a single spike in the middle of a gradual
        trend series is correctly identified as noise without disrupting
        the detection of the underlying gradual trends.
        """
        # spike test 0.1 - add a spike
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-03-25':'2025-03-25', 'gradual'] = 200  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected noise segments representing the spike
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-03-24', 'end': '2025-03-26'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_spike_single_later_series(self):
        """
        Test gradual trends with one spike later in the series.
        
        Reference: test.py spike test 1.1, lines 145-150
        Spike location: 2025-04-06 (value=200)
        
        This test verifies that a spike occurring later in the series
        is correctly identified and doesn't create noise artifacts on
        either side.
        """
        # spike test 1.1 - add a spike
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-04-06':'2025-04-06', 'gradual'] = 200  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected noise segments representing the spike
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-04-04', 'end': '2025-04-07'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_three_spikes_distributed(self):
        """
        Test gradual trends with three spikes distributed across the series.
        
        Reference: test.py spike test 1.2, lines 153-160
        Spike locations:
        - 2025-04-08 (value=200)
        - 2025-05-08 (value=200)
        - 2025-06-08 (value=200)
        
        This test verifies that multiple spikes distributed throughout
        the series are correctly identified as noise, and that the
        algorithm properly handles fill-in flats and doesn't displace
        trends on either side.
        """
        # spike test 1.2 - add 3 spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-04-08':'2025-04-08', 'gradual'] = 200  # spike
        df.loc['2025-05-08':'2025-05-08', 'gradual'] = 200  # spike
        df.loc['2025-06-08':'2025-06-08', 'gradual'] = 200  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected noise segments representing the three spikes
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-04-07', 'end': '2025-04-09'},
            {'direction': 'Noise', 'start': '2025-05-07', 'end': '2025-05-09'},
            {'direction': 'Noise', 'start': '2025-06-07', 'end': '2025-06-09'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_single_spike_higher_value(self):
        """
        Test gradual trends with one spike at a higher value.
        
        Reference: test.py spike test 1.3, lines 163-168
        Spike location: 2025-04-08 (value=250)
        
        This test verifies that a higher-value spike is detected precisely
        without creating white gaps or hang-ups during abrupt shave processing.
        """
        # spike test 1.3 - add a spike with higher value
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-04-08':'2025-04-08', 'gradual'] = 250  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected noise segments representing the spike
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-04-05', 'end': '2025-04-09'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_two_spikes_different_values(self):
        """
        Test gradual trends with two spikes at different values.
        
        Reference: test.py spike test 1.4, lines 171-178
        Spike locations:
        - 2025-04-09 (value=100)
        - 2025-05-06 (value=200)
        
        This test verifies that spikes with different values are detected
        properly and don't create white gaps or kill uptrends on the left.
        """
        # spike test 1.4 - add 2 spikes with different values
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 100  # spike
        df.loc['2025-05-06':'2025-05-06', 'gradual'] = 200  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected noise segments representing the two spikes
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-04-08', 'end': '2025-04-10'},
            {'direction': 'Noise', 'start': '2025-05-05', 'end': '2025-05-07'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_three_spikes_variant_a(self):
        """
        Test gradual trends with three spikes (variant A).
        
        Reference: test.py spike test 1.5, lines 181-188
        Spike locations:
        - 2025-02-17 (value=100)
        - 2025-04-09 (value=150)
        - 2025-06-03 (value=350)
        
        This test verifies that three spikes with varying values are
        detected correctly, especially ensuring the high-value spike
        (350) is detected as noise rather than being overcast with
        a red (down) trend.
        """
        # spike test 1.5 - add 3 spikes with varying values
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100  # spike
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150  # spike
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 350  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected noise segments representing the three spikes
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-02-16', 'end': '2025-02-18'},
            {'direction': 'Noise', 'start': '2025-04-08', 'end': '2025-04-10'},
            {'direction': 'Noise', 'start': '2025-06-02', 'end': '2025-06-04'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_three_spikes_variant_b(self):
        """
        Test gradual trends with three spikes (variant B).
        
        Reference: test.py spike test 1.6, lines 191-198
        Spike locations:
        - 2025-02-17 (value=100)
        - 2025-04-09 (value=150)
        - 2025-06-03 (value=320)
        
        This test verifies similar spike patterns to variant A but with
        a different value for the last spike (320 vs 350), ensuring
        precise detection at the far right without white gaps.
        """
        # spike test 1.6 - add 3 spikes with varying values (variant B)
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100  # spike
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150  # spike
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected noise segments representing the three spikes
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-02-16', 'end': '2025-02-18'},
            {'direction': 'Noise', 'start': '2025-04-08', 'end': '2025-04-10'},
            {'direction': 'Noise', 'start': '2025-06-02', 'end': '2025-06-04'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)

    @pytest.mark.core
    def test_gradual_four_spikes_distributed(self):
        """
        Test gradual trends with four spikes distributed across the series.
        
        Reference: test.py spike test 1.7, lines 202-210
        Spike locations:
        - 2025-02-28 (value=125)
        - 2025-04-09 (value=150)
        - 2025-05-08 (value=300)
        - 2025-06-03 (value=320)
        
        This test verifies that four spikes with varying values are
        correctly detected as noise without creating white gaps or
        noise artifacts on the right side.
        """
        # spike test 1.7 - add 4 spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-02-28':'2025-02-28', 'gradual'] = 125  # spike
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150  # spike
        df.loc['2025-05-08':'2025-05-08', 'gradual'] = 300  # spike
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Expected noise segments representing the four spikes
        expected_noise_segments = [
            {'direction': 'Noise', 'start': '2025-02-27', 'end': '2025-03-01'},
            {'direction': 'Noise', 'start': '2025-04-08', 'end': '2025-04-10'},
            {'direction': 'Noise', 'start': '2025-05-07', 'end': '2025-05-09'},
            {'direction': 'Noise', 'start': '2025-06-02', 'end': '2025-06-04'},
        ]
        
        # Filter for noise segments and validate
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert_segments_match(noise_segments, expected_noise_segments)
