"""
Tests for noise spike scenarios with abrupt trends.

These tests verify that the trend detection algorithm correctly handles
noise spikes (single-point or short-duration anomalies) in time series
with abrupt trend changes. Each test focuses on the location and impact
of spikes on trend detection.

Reference: tests/test.py lines 16-72 (Abrupts and Spikes section)
"""

import pytest
import pytrendy as pt


class TestNoiseSpikesAbrupt:
    """Test cases for noise spike detection with abrupt trends."""

    @pytest.mark.core
    def test_abrupt_base_no_spikes(self):
        """
        Test abrupt trends without spikes (base instance).
        
        Reference: test.py synth 1, lines 19-27
        
        This test establishes the baseline behavior for abrupt trend detection
        without any noise spikes. The data has multiple abrupt changes at
        different levels.
        """
        # synth 1 - base instance with no spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Verify results were generated
        assert results is not None
        assert hasattr(results, 'segments')
        assert len(results.segments) > 0
        
        # Count different segment types
        segment_types = [seg['direction'] for seg in results.segments]
        
        # Verify we have multiple segments (indicating trend detection)
        assert len(segment_types) >= 3, \
            "Should detect multiple segments for abrupt trends"

    @pytest.mark.core
    def test_abrupt_single_spike(self):
        """
        Test abrupt trends with one noise spike.
        
        Reference: test.py synth 2, lines 30-39
        Spike location: 2025-06-01 (value=300)
        
        This test verifies that a single spike at the end of the series
        is correctly identified and doesn't interfere with the detection
        of preceding abrupt trends.
        """
        # synth 2 - 1 spike
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # spike
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Verify results were generated
        assert results is not None
        assert hasattr(results, 'segments')
        assert len(results.segments) > 0
        
        # Check if noise segments are detected
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        
        # Verify that at least one noise segment is detected
        assert len(noise_segments) >= 1, \
            "Should detect at least one noise segment for the spike"

    @pytest.mark.core
    def test_abrupt_three_spikes(self):
        """
        Test abrupt trends with three noise spikes.
        
        Reference: test.py synth 3, lines 42-54
        Spike locations:
        - 2025-02-01 (value=500)
        - 2025-03-01 (value=500)
        - 2025-06-01 (value=300)
        
        This test verifies that multiple spikes at different locations
        are correctly identified as noise and don't interfere with
        detection of abrupt trends.
        """
        # synth 3 - 3 spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # spike
        df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # spike
        df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500  # spike
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Verify results were generated
        assert results is not None
        assert hasattr(results, 'segments')
        assert len(results.segments) > 0
        
        # Check if noise segments are detected
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        
        # Verify that multiple noise segments are detected
        assert len(noise_segments) >= 2, \
            "Should detect at least two noise segments for multiple spikes"

    @pytest.mark.core
    def test_abrupt_four_spikes(self):
        """
        Test abrupt trends with four noise spikes.
        
        Reference: test.py synth 4, lines 57-72
        Spike locations:
        - 2025-02-01 (value=500)
        - 2025-03-01 (value=500)
        - 2025-04-14 (value=500)
        - 2025-06-01 (value=300)
        
        This test verifies that the algorithm can handle multiple spikes
        distributed throughout the series, including spikes that occur
        near trend boundaries.
        """
        # synth 4 - 4 spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # spike
        df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # spike
        df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500  # spike
        df.loc['2025-04-14':'2025-04-14', 'abrupt'] = 500  # spike
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Verify results were generated
        assert results is not None
        assert hasattr(results, 'segments')
        assert len(results.segments) > 0
        
        # Check if noise segments are detected
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        
        # Verify that multiple noise segments are detected
        assert len(noise_segments) >= 3, \
            "Should detect at least three noise segments for four spikes"
        
        # Verify that non-noise segments still exist
        non_noise_segments = [seg for seg in results.segments if seg['direction'] != 'Noise']
        assert len(non_noise_segments) >= 2, \
            "Should still detect non-noise trend segments despite multiple spikes"
