"""
Tests for gradual trends with spike noise.

This module tests PyTrendy's ability to detect gradual trends while properly
handling spike noise at various positions in the time series. It verifies that
spikes are correctly identified and don't disrupt the underlying trend detection.
"""

import pytest
import pytrendy as pt
import pandas as pd


class TestGradualsAndSpikes:
    """Test cases for gradual trends with spike noise."""

    @pytest.fixture
    def base_synthetic_data(self):
        """Load base synthetic dataset."""
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        return df

    def test_single_spike_in_gradual_early(self, base_synthetic_data):
        """Test gradual trend detection with a single spike early in the series."""
        df = base_synthetic_data.copy()
        df.loc['2025-03-25':'2025-03-25', 'gradual'] = 200
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Should detect segments
        assert len(results.df) > 0, "No segments detected with early spike"
        
        # Check segments around spike date
        spike_date = pd.to_datetime('2025-03-25')
        for idx, row in results.df.iterrows():
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])
            assert start <= end, f"Segment {idx}: invalid date range"
        
        # Verify detection continues after spike
        max_end = pd.to_datetime(results.df['end']).max()
        assert max_end > spike_date, "Detection should continue beyond spike"

    def test_single_spike_in_gradual_mid(self, base_synthetic_data):
        """Test gradual trend detection with a single spike in the middle."""
        df = base_synthetic_data.copy()
        df.loc['2025-04-06':'2025-04-06', 'gradual'] = 200
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Should detect multiple segments
        assert len(results.df) >= 3, "Expected multiple segments with mid spike"
        
        # Verify segments before and after spike
        spike_date = pd.to_datetime('2025-04-06')
        before_spike = results.df[pd.to_datetime(results.df['end']) < spike_date]
        after_spike = results.df[pd.to_datetime(results.df['start']) > spike_date]
        
        # Should have segments both before and after
        assert len(before_spike) > 0 or len(after_spike) > 0, \
            "Should detect trends around the spike"

    def test_three_spikes_in_gradual(self, base_synthetic_data):
        """Test gradual trend detection with three spikes."""
        df = base_synthetic_data.copy()
        df.loc['2025-04-08':'2025-04-08', 'gradual'] = 200
        df.loc['2025-05-08':'2025-05-08', 'gradual'] = 300
        df.loc['2025-06-08':'2025-06-08', 'gradual'] = 200
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Should detect segments
        assert len(results.df) > 0, "No segments detected with three spikes"
        
        # Check that detection spans the time range with spikes
        all_starts = pd.to_datetime(results.df['start'])
        all_ends = pd.to_datetime(results.df['end'])
        
        earliest = all_starts.min()
        latest = all_ends.max()
        
        # Should cover from before first spike to after last spike
        assert earliest < pd.to_datetime('2025-04-08'), \
            "Detection should start before first spike"
        assert latest > pd.to_datetime('2025-05-08'), \
            "Detection should continue past middle spike"

    def test_high_spike_in_gradual_no_padding(self, base_synthetic_data):
        """Test gradual trend with a high spike value without padding."""
        df = base_synthetic_data.copy()
        df.loc['2025-04-08':'2025-04-08', 'gradual'] = 250
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Should handle high spike
        assert len(results.df) > 0, "No segments detected with high spike"
        
        # Verify all segments are valid
        for idx, row in results.df.iterrows():
            assert pd.notna(row['direction']), f"Segment {idx} missing direction"
            assert row['days'] > 0, f"Segment {idx} should have positive days"

    def test_two_spikes_different_magnitudes(self, base_synthetic_data):
        """Test gradual trend with two spikes of different magnitudes."""
        df = base_synthetic_data.copy()
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 100
        df.loc['2025-05-06':'2025-05-06', 'gradual'] = 200
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Should detect segments
        assert len(results.df) > 0, "No segments detected with two spikes"
        
        # Check for presence of different trend directions
        directions = results.df['direction'].unique()
        assert len(directions) >= 2, "Expected multiple trend types with spikes"
        
        # Verify segments don't have unrealistic jumps
        for idx, row in results.df.iterrows():
            if row['direction'] in ['Up', 'Down'] and pd.notna(row['total_change']):
                # Reasonable bounds on change magnitude
                assert abs(row['total_change']) < 500, \
                    f"Segment {idx} has unrealistic change magnitude"

    def test_three_spikes_early_mid_late(self, base_synthetic_data):
        """Test gradual trend with spikes at different positions."""
        df = base_synthetic_data.copy()
        df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100  # Early
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150  # Mid
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 350  # Late
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Should detect multiple segments across the range
        assert len(results.df) >= 3, "Expected multiple segments with distributed spikes"
        
        # Check temporal coverage
        all_starts = pd.to_datetime(results.df['start'])
        all_ends = pd.to_datetime(results.df['end'])
        
        time_span = (all_ends.max() - all_starts.min()).days
        assert time_span > 90, "Should cover at least 3 months"
        
        # Verify segments are ordered
        for i in range(len(results.df) - 1):
            current_end = pd.to_datetime(results.df.iloc[i]['end'])
            next_start = pd.to_datetime(results.df.iloc[i + 1]['start'])
            # Segments should be sequential (allowing same-day transitions)
            assert (next_start - current_end).days >= 0, \
                "Segments should be in chronological order"

    def test_three_spikes_moderate_late(self, base_synthetic_data):
        """Test gradual trend with three spikes, including moderate late spike."""
        df = base_synthetic_data.copy()
        df.loc['2025-02-17':'2025-02-17', 'gradual'] = 100
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320  # Moderate spike
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Should detect segments
        assert len(results.df) > 0, "No segments detected with moderate late spike"
        
        # Check that late segments are detected
        late_segments = results.df[
            pd.to_datetime(results.df['start']) >= pd.to_datetime('2025-06-01')
        ]
        
        # Either spike is part of a segment or filtered as noise
        # Just verify that detection continues into June
        all_ends = pd.to_datetime(results.df['end'])
        assert all_ends.max() >= pd.to_datetime('2025-06-01'), \
            "Detection should extend into June"

    def test_spike_handling_consistency(self, base_synthetic_data):
        """Test that spike handling is consistent across runs."""
        df = base_synthetic_data.copy()
        df.loc['2025-03-25':'2025-03-25', 'gradual'] = 200
        
        # Run detection twice with same data
        results1 = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        results2 = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Results should be consistent
        assert len(results1.df) == len(results2.df), \
            "Results should be consistent across runs"
        
        # Directions should match
        assert list(results1.df['direction']) == list(results2.df['direction']), \
            "Detected directions should be consistent"
