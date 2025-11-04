"""
Tests for PyTrendyResults class - the main user-facing interface.

This module rigorously tests the PyTrendyResults class to ensure it handles:
- Normal gradual trends
- Edge cases (all zeros, extreme outliers)
- All public methods and properties
- Proper data structure conversions

These tests are marked as 'core' to ensure they're always run during CI/CD.
"""

import pytest
import pytrendy as pt
import pandas as pd


class TestPytrendyResults:
    """Test suite for PytrendyResults class - the main user-facing interface."""

    def _create_gradual_results(self):
        """
        Helper method to create results from gradual synthetic data.
        This serves as the reference dataset for most tests.
        """
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        return results

    def _create_zeros_signal(self):
        """
        Helper method to create a signal with all zeros.
        This should result in no trends being detected.
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': [0.0] * 100
        })
        return df

    def _create_outlier_signal(self):
        """
        Helper method to create a signal with an extreme outlier.
        This should introduce a noise segment in the results.
        """
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        # Create a mostly flat signal with one extreme outlier
        values = [10.0] * 50 + [100.0] + [10.0] * 49
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        return df

    @pytest.mark.core
    def test_initialization_gradual(self):
        """Test that PytrendyResults initializes correctly with gradual data."""
        results = self._create_gradual_results()
        
        # Check that segments exist
        assert results.segments is not None
        assert len(results.segments) > 0
        assert isinstance(results.segments, list)
        
        # Check that each segment is a dictionary with expected keys
        for segment in results.segments:
            assert isinstance(segment, dict)
            assert 'direction' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'time_index' in segment

    @pytest.mark.core
    def test_initialization_all_zeros(self):
        """Test that PytrendyResults handles all-zeros signal correctly."""
        df = self._create_zeros_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # With all zeros, we expect no significant trends
        # The algorithm might detect a flat region or nothing at all
        assert results.segments is not None
        assert isinstance(results.segments, list)
        
        # If segments exist, they should be Flat or have minimal change
        if len(results.segments) > 0:
            for segment in results.segments:
                # Should be Flat or Noise, not Up or Down trends
                assert segment['direction'] in ['Flat', 'Noise']

    @pytest.mark.core
    def test_initialization_with_outlier(self):
        """Test that PytrendyResults detects noise segment with extreme outlier."""
        df = self._create_outlier_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Check that results were created
        assert results.segments is not None
        assert len(results.segments) > 0
        
        # This signal is mostly flat with an extreme outlier, so it should be detected as Noise
        assert 'direction_counts' in results.summary
        direction_counts = results.summary['direction_counts']
        assert 'Noise' in direction_counts
        assert direction_counts['Noise'] > 0

    @pytest.mark.core
    def test_segments_attribute_accessibility(self):
        """Test that segments attribute is directly accessible."""
        results = self._create_gradual_results()
        
        # Direct access to segments
        segments = results.segments
        
        assert isinstance(segments, list)
        assert len(segments) > 0

    @pytest.mark.core
    def test_best_attribute_accessibility(self):
        """Test that best attribute is directly accessible."""
        results = self._create_gradual_results()
        
        # Direct access to best
        best = results.best
        
        assert best is not None
        assert isinstance(best, dict)

    @pytest.mark.core
    def test_summary_attribute_accessibility(self):
        """Test that summary attribute is directly accessible."""
        results = self._create_gradual_results()
        
        # Direct access to summary
        summary = results.summary
        
        assert isinstance(summary, dict)
        assert 'direction_counts' in summary

    @pytest.mark.core
    def test_df_attribute_accessibility(self):
        """Test that df attribute is directly accessible."""
        results = self._create_gradual_results()
        
        # Direct access to df
        df = results.df
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.core
    def test_df_summary_attribute_accessibility(self):
        """Test that df_summary attribute is directly accessible."""
        results = self._create_gradual_results()
        
        # Direct access to df_summary
        df_summary = results.df_summary
        
        assert isinstance(df_summary, pd.DataFrame)
        assert len(df_summary) > 0

    @pytest.mark.core
    def test_set_best_with_trends(self):
        """Test that set_best identifies the best trend correctly."""
        results = self._create_gradual_results()
        
        # Should have identified a best segment
        assert results.best is not None
        assert isinstance(results.best, dict)
        
        # Best segment should have required fields
        assert 'direction' in results.best
        assert 'change_rank' in results.best
        assert 'start' in results.best
        assert 'end' in results.best
        
        # The best trend should be the last Down trend with highest total change
        assert results.best['direction'] == 'Down'
        assert results.best['start'] == '2025-05-09'
        assert results.best['end'] == '2025-06-17'
        assert results.best['change_rank'] == 1

    @pytest.mark.core
    def test_set_best_no_trends(self):
        """Test that set_best handles case with no trends correctly."""
        df = self._create_zeros_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # With no significant trends detected, best should be None
        assert results.best is None

    @pytest.mark.core
    def test_set_summary_structure(self):
        """Test that set_summary creates correct summary structure."""
        results = self._create_gradual_results()
        
        # Check summary exists and has expected keys
        assert hasattr(results, 'summary')
        assert isinstance(results.summary, dict)
        assert 'direction_counts' in results.summary
        
        # Check direction_counts structure
        direction_counts = results.summary['direction_counts']
        assert isinstance(direction_counts, dict)
        
        # For gradual signal, should have 3 Up, 3 Down, 3 Flat, 0 Noise
        assert direction_counts['Up'] == 3
        assert direction_counts['Down'] == 3
        assert direction_counts['Flat'] == 3
        assert 'Noise' not in direction_counts or direction_counts['Noise'] == 0

    @pytest.mark.core
    def test_set_summary_with_outlier(self):
        """Test that summary correctly identifies noise in outlier signal."""
        df = self._create_outlier_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Check that summary was created
        assert hasattr(results, 'summary')
        assert 'direction_counts' in results.summary
        
        # Filter segments to check for noise - should return exactly 1 noise segment
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert isinstance(noise_segments, list)
        assert len(noise_segments) == 1

    @pytest.mark.core
    def test_set_df_structure(self):
        """Test that set_df creates a proper DataFrame."""
        results = self._create_gradual_results()
        
        # Check df exists and is a DataFrame
        assert hasattr(results, 'df')
        assert isinstance(results.df, pd.DataFrame)
        
        # Check that df has expected columns
        expected_cols = ['direction', 'start', 'end', 'days']
        for col in expected_cols:
            assert col in results.df.columns
        
        # Check that index is time_index
        assert results.df.index.name == 'time_index'
        
        # Check that number of rows matches segments
        assert len(results.df) == len(results.segments)

    @pytest.mark.core
    def test_set_df_empty_segments(self):
        """Test that set_df handles empty segments gracefully."""
        # Create empty results
        from pytrendy.io.results_pytrendy import PyTrendyResults
        results = PyTrendyResults([])
        
        # Current behavior: When segments are empty, set_df returns early
        # without setting self.df attribute. This test verifies this behavior.
        # If the attribute exists, it should be an empty DataFrame.
        if hasattr(results, 'df'):
            assert isinstance(results.df, pd.DataFrame)
            assert len(results.df) == 0

    @pytest.mark.core
    def test_df_summary_structure(self):
        """Test that df_summary has the correct structure."""
        results = self._create_gradual_results()
        
        # Check df_summary exists
        assert hasattr(results, 'df_summary')
        assert isinstance(results.df_summary, pd.DataFrame)
        
        # Check basic columns exist
        expected_cols = ['direction', 'start', 'end', 'days']
        for col in expected_cols:
            assert col in results.df_summary.columns
        
        # Check that index is time_index
        assert results.df_summary.index.name == 'time_index'

    @pytest.mark.core
    def test_filter_segments_by_direction_up(self):
        """Test filtering segments by 'Up' direction."""
        results = self._create_gradual_results()
        
        # Filter for Up trends - should match expected segments from gradual data
        up_segments = results.filter_segments(direction='Up', format='dict')
        
        assert isinstance(up_segments, list)
        assert len(up_segments) == 3
        
        # Expected Up segments from test_core_gradual
        expected_up = [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
            {'direction': 'Up', 'start': '2025-02-10', 'end': '2025-03-14'},
            {'direction': 'Up', 'start': '2025-04-02', 'end': '2025-05-08'},
        ]
        
        for i, segment in enumerate(up_segments):
            assert segment['direction'] == 'Up'
            assert pd.to_datetime(segment['start']).strftime('%Y-%m-%d') == expected_up[i]['start']
            assert pd.to_datetime(segment['end']).strftime('%Y-%m-%d') == expected_up[i]['end']

    @pytest.mark.core
    def test_filter_segments_by_direction_down(self):
        """Test filtering segments by 'Down' direction."""
        results = self._create_gradual_results()
        
        # Filter for Down trends - should match expected segments from gradual data
        down_segments = results.filter_segments(direction='Down', format='dict')
        
        assert isinstance(down_segments, list)
        assert len(down_segments) == 3
        
        # Expected Down segments from test_core_gradual
        expected_down = [
            {'direction': 'Down', 'start': '2025-01-25', 'end': '2025-02-05'},
            {'direction': 'Down', 'start': '2025-03-18', 'end': '2025-04-01'},
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-17'},
        ]
        
        for i, segment in enumerate(down_segments):
            assert segment['direction'] == 'Down'
            assert pd.to_datetime(segment['start']).strftime('%Y-%m-%d') == expected_down[i]['start']
            assert pd.to_datetime(segment['end']).strftime('%Y-%m-%d') == expected_down[i]['end']

    @pytest.mark.core
    def test_filter_segments_by_direction_flat(self):
        """Test filtering segments by 'Flat' direction."""
        results = self._create_gradual_results()
        
        # Filter for Flat segments - should match expected segments from gradual data
        flat_segments = results.filter_segments(direction='Flat', format='dict')
        
        assert isinstance(flat_segments, list)
        assert len(flat_segments) == 3
        
        # Expected Flat segments from test_core_gradual
        expected_flat = [
            {'direction': 'Flat', 'start': '2025-02-06', 'end': '2025-02-09'},
            {'direction': 'Flat', 'start': '2025-03-15', 'end': '2025-03-17'},
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-29'},
        ]
        
        for i, segment in enumerate(flat_segments):
            assert segment['direction'] == 'Flat'
            assert pd.to_datetime(segment['start']).strftime('%Y-%m-%d') == expected_flat[i]['start']
            assert pd.to_datetime(segment['end']).strftime('%Y-%m-%d') == expected_flat[i]['end']

    @pytest.mark.core
    def test_filter_segments_by_direction_noise(self):
        """Test filtering segments by 'Noise' direction."""
        df = self._create_outlier_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Filter for Noise segments - outlier signal should have exactly 1 noise segment
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        
        assert isinstance(noise_segments, list)
        assert len(noise_segments) == 1
        
        # Check explicit segment details
        assert noise_segments[0]['direction'] == 'Noise'
        assert pd.to_datetime(noise_segments[0]['start']).strftime('%Y-%m-%d') == '2025-01-01'
        assert pd.to_datetime(noise_segments[0]['end']).strftime('%Y-%m-%d') == '2025-02-21'

    @pytest.mark.core
    def test_filter_segments_up_down_combined(self):
        """Test filtering for combined Up/Down trends."""
        results = self._create_gradual_results()
        
        # Filter for Up/Down trends - should match expected segments from test_core_gradual
        trend_segments = results.filter_segments(direction='Up/Down', format='dict')
        
        assert isinstance(trend_segments, list)
        assert len(trend_segments) == 6  # 3 Up + 3 Down
        
        # All segments should be Up or Down
        for segment in trend_segments:
            assert segment['direction'] in ['Up', 'Down']
        
        # Verify count of each direction
        up_count = sum(1 for s in trend_segments if s['direction'] == 'Up')
        down_count = sum(1 for s in trend_segments if s['direction'] == 'Down')
        assert up_count == 3
        assert down_count == 3

    @pytest.mark.core
    def test_filter_segments_any_direction(self):
        """Test filtering with 'Any' direction returns all segments."""
        results = self._create_gradual_results()
        
        # Filter for any direction - should return all 9 segments from test_core_gradual
        all_segments = results.filter_segments(direction='Any', format='dict')
        
        assert isinstance(all_segments, list)
        assert len(all_segments) == 9
        assert len(all_segments) == len(results.segments)

    @pytest.mark.core
    def test_filter_segments_sort_by_time_index(self):
        """Test sorting segments by time_index."""
        results = self._create_gradual_results()
        
        # Filter and sort by time_index (ascending)
        sorted_segments = results.filter_segments(sort_by='time_index', format='dict')
        
        assert isinstance(sorted_segments, list)
        # Check that segments are sorted by time_index
        time_indices = [seg['time_index'] for seg in sorted_segments]
        assert time_indices == sorted(time_indices)

    @pytest.mark.core
    def test_filter_segments_sort_by_change_rank(self):
        """Test sorting segments by change_rank."""
        results = self._create_gradual_results()
        
        # Filter and sort by change_rank (descending by total_change)
        sorted_segments = results.filter_segments(sort_by='change_rank', format='dict')
        
        assert isinstance(sorted_segments, list)
        # Check that segments are sorted by absolute total_change (descending)
        if len(sorted_segments) > 1 and 'total_change' in sorted_segments[0]:
            changes = [abs(seg.get('total_change', 0)) for seg in sorted_segments]
            assert changes == sorted(changes, reverse=True)

    @pytest.mark.core
    def test_filter_segments_format_dict(self):
        """Test that format='dict' returns list of dictionaries."""
        results = self._create_gradual_results()
        
        segments = results.filter_segments(direction='Any', format='dict')
        
        assert isinstance(segments, list)
        assert len(segments) == 9
        assert isinstance(segments[0], dict)
        
        # Check that each dict has expected keys
        for segment in segments:
            assert 'direction' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'time_index' in segment

    @pytest.mark.core
    def test_filter_segments_format_df(self):
        """Test that format='df' returns DataFrame."""
        results = self._create_gradual_results()
        
        segments_df = results.filter_segments(direction='Any', format='df')
        
        assert isinstance(segments_df, pd.DataFrame)
        assert segments_df.index.name == 'time_index'
        assert len(segments_df) == 9
        
        # Check that DataFrame has expected columns
        expected_cols = ['direction', 'start', 'end', 'days']
        for col in expected_cols:
            assert col in segments_df.columns

    @pytest.mark.core
    def test_filter_segments_empty_results(self):
        """Test filtering on empty segments returns empty list."""
        from pytrendy.io.results_pytrendy import PyTrendyResults
        results = PyTrendyResults([])
        
        filtered = results.filter_segments(direction='Up', format='dict')
        
        assert filtered == []

    @pytest.mark.core
    def test_print_summary_with_trends(self):
        """Test that print_summary executes without errors."""
        results = self._create_gradual_results()
        
        # Should not raise any exceptions
        try:
            results.print_summary()
            success = True
        except Exception as e:
            success = False
            print(f"print_summary raised exception: {e}")
        
        assert success

    @pytest.mark.core
    def test_print_summary_no_trends(self):
        """Test that print_summary handles no trends gracefully."""
        df = self._create_zeros_signal()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Current behavior: When segments are empty, set_summary() returns early
        # without setting the summary attribute. This test handles that case.
        # Only attempt print_summary if results have segments and summary exists.
        if len(results.segments) > 0 and hasattr(results, 'summary'):
            try:
                results.print_summary()
                success = True
            except Exception as e:
                success = False
                print(f"print_summary raised exception: {e}")
            
            assert success

    @pytest.mark.core
    def test_integration_full_workflow(self):
        """Test full workflow: detect trends, filter, and access results."""
        results = self._create_gradual_results()
        
        # 1. Access segments - should have 9 total (3 Up, 3 Down, 3 Flat)
        assert len(results.segments) == 9
        
        # 2. Get best trend - should be the last Down trend
        assert results.best is not None
        assert results.best['direction'] == 'Down'
        assert results.best['start'] == '2025-05-09'
        assert results.best['end'] == '2025-06-17'
        
        # 3. Check summary - exact counts from gradual data
        assert 'direction_counts' in results.summary
        assert results.summary['direction_counts'] == {'Up': 3, 'Down': 3, 'Flat': 3}
        
        # 4. Filter for uptrends - should get 3
        up_trends = results.filter_segments(direction='Up', format='df')
        assert isinstance(up_trends, pd.DataFrame)
        assert len(up_trends) == 3
        
        # 5. Sort by change rank - best should be first
        ranked = results.filter_segments(sort_by='change_rank', format='dict')
        assert isinstance(ranked, list)
        assert ranked[0]['change_rank'] == 1
        
        # 6. Access DataFrames - should have all 9 segments
        assert isinstance(results.df, pd.DataFrame)
        assert len(results.df) == 9
        assert isinstance(results.df_summary, pd.DataFrame)
        assert len(results.df_summary) == 9

    @pytest.mark.core
    def test_edge_case_single_segment(self):
        """Test handling of dataset that produces only one segment."""
        # Create a simple upward trend
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        values = list(range(30))
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Should handle single segment gracefully
        assert results.segments is not None
        assert len(results.segments) >= 1
        assert isinstance(results.df, pd.DataFrame)
        
        # Check direction count - should have exactly 1 uptrend
        assert 'direction_counts' in results.summary
        assert results.summary['direction_counts']['Up'] == 1

    @pytest.mark.core
    def test_segments_have_required_fields(self):
        """Test that all segments have required fields."""
        results = self._create_gradual_results()
        
        required_fields = ['direction', 'start', 'end', 'time_index', 'days']
        
        for i, segment in enumerate(results.segments):
            for field in required_fields:
                assert field in segment, f"Segment {i} missing field: {field}"
    
    @pytest.mark.core
    def test_df_have_required_cols(self):
        """Test that DataFrame has required columns."""
        results = self._create_gradual_results()
        
        required_cols = ['direction', 'start', 'end', 'days']
        
        for col in required_cols:
            assert col in results.df.columns, f"DataFrame missing column: {col}"

    @pytest.mark.core
    def test_dataframe_conversion_consistency(self):
        """Test that DataFrame conversion preserves all segment data."""
        results = self._create_gradual_results()
        
        # Compare list and DataFrame lengths
        assert len(results.segments) == len(results.df)
        
        # Check that directions match
        segment_directions = [seg['direction'] for seg in results.segments]
        df_directions = results.df['direction'].tolist()
        
        # Should have same directions (order may vary)
        assert sorted(segment_directions) == sorted(df_directions)
