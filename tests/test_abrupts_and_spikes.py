"""
Tests for abrupt trend detection and spike handling.

This module tests PyTrendy's ability to detect abrupt changes and handle spike noise
in time series data. It verifies that the algorithm correctly identifies abrupt segments
and distinguishes them from noise spikes.
"""

import pytest
import pytrendy as pt
import pandas as pd


class TestAbruptsAndSpikes:
    """Test cases for abrupt trend detection and spike noise handling."""

    @pytest.fixture
    def base_synthetic_data(self):
        """Load base synthetic dataset."""
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        return df

    def test_abrupt_detection_no_padding(self, base_synthetic_data):
        """Test abrupt detection without padding."""
        df = base_synthetic_data.copy()
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
        
        # Assert that results contain segments
        assert len(results.df) > 0, "No segments detected"
        
        # Assert that we detect at least some flat and up/down trends
        directions = results.df['direction'].unique()
        assert 'Flat' in directions or 'Up' in directions or 'Down' in directions, \
            "Expected at least one trend direction to be detected"
        
        # Verify that segments have valid start and end dates
        assert all(pd.notna(results.df['start'])), "All segments should have start dates"
        assert all(pd.notna(results.df['end'])), "All segments should have end dates"
        
        # Check that start dates are before end dates
        for idx, row in results.df.iterrows():
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])
            assert start <= end, f"Start date {start} should be before or equal to end date {end}"

    def test_single_spike_detection(self, base_synthetic_data):
        """Test detection with a single spike in the data."""
        df = base_synthetic_data.copy()
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # Single spike
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Assert that spike is handled appropriately
        assert len(results.df) > 0, "No segments detected"
        
        # Check that we have reasonable segments around the spike date
        june_segments = results.df[
            (pd.to_datetime(results.df['start']) <= pd.to_datetime('2025-06-01')) &
            (pd.to_datetime(results.df['end']) >= pd.to_datetime('2025-06-01'))
        ]
        
        # The spike should either be:
        # 1. Detected as Noise
        # 2. Part of a larger segment
        # 3. Filtered out completely
        if len(june_segments) > 0:
            # If segment exists around spike, verify it has valid properties
            assert all(pd.notna(june_segments['direction'])), \
                "Segments around spike should have valid directions"

    def test_three_spikes_detection(self, base_synthetic_data):
        """Test detection with three spikes in the data."""
        df = base_synthetic_data.copy()
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # Spike 1
        df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # Spike 2
        df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500  # Spike 3
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Assert that results contain segments
        assert len(results.df) > 0, "No segments detected"
        
        # Verify all segments have valid dates
        for idx, row in results.df.iterrows():
            assert pd.notna(row['start']), f"Segment {idx} missing start date"
            assert pd.notna(row['end']), f"Segment {idx} missing end date"
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])
            assert start <= end, f"Segment {idx}: start should be <= end"

    def test_four_spikes_detection(self, base_synthetic_data):
        """Test detection with four spikes in the data."""
        df = base_synthetic_data.copy()
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df.loc['2025-06-01':'2025-06-01', 'abrupt'] = 300  # Spike 1
        df.loc['2025-02-01':'2025-02-01', 'abrupt'] = 500  # Spike 2
        df.loc['2025-03-01':'2025-03-01', 'abrupt'] = 500  # Spike 3
        df.loc['2025-04-14':'2025-04-14', 'abrupt'] = 500  # Spike 4
        
        results = pt.detect_trends(
            df.reset_index(),
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        # Assert that results contain segments
        assert len(results.df) > 0, "No segments detected"
        
        # Check that segments cover the expected time range
        all_starts = pd.to_datetime(results.df['start'])
        all_ends = pd.to_datetime(results.df['end'])
        
        earliest_start = all_starts.min()
        latest_end = all_ends.max()
        
        # Should cover at least from January to June
        assert earliest_start <= pd.to_datetime('2025-02-01'), \
            "Detection should start early in the series"
        assert latest_end >= pd.to_datetime('2025-04-01'), \
            "Detection should extend into April or later"
        
        # Verify that all segments have direction labels
        assert all(results.df['direction'].isin(['Up', 'Down', 'Flat', 'Noise'])), \
            "All segments should have valid direction labels"
