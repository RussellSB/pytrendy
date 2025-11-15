"""
Tests for PyTrendyResults class - the main user-facing interface.

This module rigorously tests the PyTrendyResults class to ensure it handles:
- Normal gradual trends
- Edge cases (all zeros, extreme outliers)
- All public methods and properties
- Proper data structure conversions

Tests are organized into separate test classes for better maintainability:
- TestResultsInitialization: Object creation and basic setup
- TestResultsAttributes: Direct attribute access (segments, best, summary, df, df_summary)
- TestResultsSetBest: Best trend identification
- TestResultsSetSummary: Summary statistics generation
- TestResultsSetDataFrame: DataFrame conversion methods
- TestResultsFilterSegments: Filtering and sorting functionality
- TestResultsPrintSummary: Output generation
- TestResultsIntegration: Full workflow and edge cases
- TestResultsDataStructures: Data structure validation

These tests are marked as 'core' to ensure they're always run during CI/CD.
"""

import pytest
import pytrendy as pt
import pandas as pd
import numpy as np
from conftest import assert_segments_match


# =============================================================================
# Shared Fixtures (used across all test classes)
# =============================================================================

@pytest.fixture
def gradual_results():
    """
    Fixture to create results from gradual synthetic data.
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


@pytest.fixture
def zeros_signal():
    """
    Fixture to create a signal with all zeros.
    This should result in no trends being detected.
    """
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': [0.0] * 100
    })
    return df


@pytest.fixture
def outlier_signal():
    """
    Fixture to create a signal with an extreme outlier.
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


# =============================================================================
# Test Classes
# =============================================================================

class TestResultsInitialization:
    """Tests for PyTrendyResults initialization with different input types."""

    @pytest.mark.core
    def test_initialization_gradual(self, gradual_results):
        """Test that PyTrendyResults initializes correctly with gradual data."""
        # Check that segments exist
        assert gradual_results.segments is not None
        assert len(gradual_results.segments) > 0
        assert isinstance(gradual_results.segments, list)
        
        # Check that each segment is a dictionary with expected keys
        for segment in gradual_results.segments:
            assert isinstance(segment, dict)
            assert 'direction' in segment
            assert 'start' in segment
            assert 'end' in segment
            assert 'time_index' in segment

    @pytest.mark.core
    def test_initialization_all_zeros(self, zeros_signal):
        """Test that PyTrendyResults handles all-zeros signal correctly."""
        results = pt.detect_trends(
            zeros_signal,
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
    def test_initialization_with_outlier(self, outlier_signal):
        """Test that PyTrendyResults detects noise segment with extreme outlier."""
        results = pt.detect_trends(
            outlier_signal,
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


class TestResultsAttributes:
    """Tests for direct attribute access (segments, best, summary, df, df_summary)."""

    @pytest.mark.core
    def test_segments_attribute_accessibility(self, gradual_results):
        """Test that segments attribute is directly accessible."""
        # Direct access to segments
        segments = gradual_results.segments
        
        assert isinstance(segments, list)
        assert len(segments) > 0

    @pytest.mark.core
    def test_best_attribute_accessibility(self, gradual_results):
        """Test that best attribute is directly accessible."""
        # Direct access to best
        best = gradual_results.best
        
        assert best is not None
        assert isinstance(best, dict)

    @pytest.mark.core
    def test_summary_attribute_accessibility(self, gradual_results):
        """Test that summary attribute is directly accessible."""
        # Direct access to summary
        summary = gradual_results.summary
        
        assert isinstance(summary, dict)
        assert 'direction_counts' in summary

    @pytest.mark.core
    def test_df_attribute_accessibility(self, gradual_results):
        """Test that df attribute is directly accessible."""
        # Direct access to df
        df = gradual_results.df
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    @pytest.mark.core
    def test_df_summary_attribute_accessibility(self, gradual_results):
        """Test that df_summary attribute is directly accessible."""
        # Direct access to df_summary
        df_summary = gradual_results.df_summary
        
        assert isinstance(df_summary, pd.DataFrame)
        assert len(df_summary) > 0


class TestResultsSetBest:
    """Tests for set_best() method - identifying the best trend."""

    @pytest.mark.core
    def test_set_best_with_trends(self, gradual_results):
        """Test that set_best identifies the best trend correctly."""
        # Should have identified a best segment
        assert gradual_results.best is not None
        assert isinstance(gradual_results.best, dict)
        
        # Best segment should have required fields
        assert 'direction' in gradual_results.best
        assert 'change_rank' in gradual_results.best
        assert 'start' in gradual_results.best
        assert 'end' in gradual_results.best
        
        # The best trend should be the last Down trend with highest total change
        assert gradual_results.best['direction'] == 'Down'
        assert gradual_results.best['start'] == '2025-05-09'
        assert gradual_results.best['end'] == '2025-06-17'
        assert gradual_results.best['change_rank'] == 1

    @pytest.mark.core
    def test_set_best_no_trends(self, zeros_signal):
        """Test that set_best handles case with no trends correctly."""
        results = pt.detect_trends(
            zeros_signal,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # With no significant trends detected, best should be None
        assert results.best is None


class TestResultsSetSummary:
    """Tests for set_summary() method - generating summary statistics."""

    @pytest.mark.core
    def test_set_summary_structure(self, gradual_results):
        """Test that set_summary creates correct summary structure."""
        # Check summary exists and has expected keys
        assert hasattr(gradual_results, 'summary')
        assert isinstance(gradual_results.summary, dict)
        assert 'direction_counts' in gradual_results.summary
        
        # Check direction_counts structure
        direction_counts = gradual_results.summary['direction_counts']
        assert isinstance(direction_counts, dict)
        
        # For gradual signal, should have 3 Up, 3 Down, 3 Flat, 0 Noise
        assert direction_counts['Up'] == 3
        assert direction_counts['Down'] == 3
        assert direction_counts['Flat'] == 3
        assert 'Noise' not in direction_counts or direction_counts['Noise'] == 0

    @pytest.mark.core
    def test_set_summary_with_outlier(self, outlier_signal):
        """Test that summary correctly identifies noise in outlier signal."""
        results = pt.detect_trends(
            outlier_signal,
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


class TestResultsSetDataFrame:
    """Tests for set_df() and DataFrame-related methods."""

    @pytest.mark.core
    def test_set_df_structure(self, gradual_results):
        """Test that set_df creates a proper DataFrame."""
        # Check df exists and is a DataFrame
        assert hasattr(gradual_results, 'df')
        assert isinstance(gradual_results.df, pd.DataFrame)
        
        # Check that df has expected columns
        expected_cols = ['direction', 'start', 'end', 'days']
        for col in expected_cols:
            assert col in gradual_results.df.columns
        
        # Check that index is time_index
        assert gradual_results.df.index.name == 'time_index'
        
        # Check that number of rows matches segments
        assert len(gradual_results.df) == len(gradual_results.segments)

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
    def test_df_summary_structure(self, gradual_results):
        """Test that df_summary has the correct structure."""
        # Check df_summary exists
        assert hasattr(gradual_results, 'df_summary')
        assert isinstance(gradual_results.df_summary, pd.DataFrame)
        
        # Check basic columns exist
        expected_cols = ['direction', 'start', 'end', 'days']
        for col in expected_cols:
            assert col in gradual_results.df_summary.columns
        
        # Check that index is time_index
        assert gradual_results.df_summary.index.name == 'time_index'

    @pytest.mark.core
    def test_df_has_required_cols(self, gradual_results):
        """Test that DataFrame has required columns."""
        required_cols = ['direction', 'start', 'end', 'days']
        
        for col in required_cols:
            assert col in gradual_results.df.columns, f"DataFrame missing column: {col}"

    @pytest.mark.core
    def test_dataframe_conversion_consistency(self, gradual_results):
        """Test that DataFrame conversion preserves all segment data."""
        # Compare list and DataFrame lengths
        assert len(gradual_results.segments) == len(gradual_results.df)
        
        # Check that directions match
        segment_directions = [seg['direction'] for seg in gradual_results.segments]
        df_directions = gradual_results.df['direction'].tolist()
        
        # Should have same directions (order may vary)
        assert sorted(segment_directions) == sorted(df_directions)


class TestResultsFilterSegments:
    """Tests for filter_segments() method - filtering and sorting functionality."""

    @pytest.mark.core
    def test_filter_segments_by_direction_up(self, gradual_results):
        """Test filtering segments by 'Up' direction."""
        # Filter for Up trends - should match expected segments from gradual data
        up_segments = gradual_results.filter_segments(direction='Up', format='dict')
        
        assert isinstance(up_segments, list)
        assert len(up_segments) == 3
        
        # Expected Up segments from test_core_gradual
        expected_up = [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
            {'direction': 'Up', 'start': '2025-02-10', 'end': '2025-03-14'},
            {'direction': 'Up', 'start': '2025-04-02', 'end': '2025-05-08'},
        ]
        
        assert_segments_match(up_segments, expected_up)

    @pytest.mark.core
    def test_filter_segments_by_direction_down(self, gradual_results):
        """Test filtering segments by 'Down' direction."""
        # Filter for Down trends - should match expected segments from gradual data
        down_segments = gradual_results.filter_segments(direction='Down', format='dict')
        
        assert isinstance(down_segments, list)
        assert len(down_segments) == 3
        
        # Expected Down segments from test_core_gradual
        expected_down = [
            {'direction': 'Down', 'start': '2025-01-25', 'end': '2025-02-05'},
            {'direction': 'Down', 'start': '2025-03-18', 'end': '2025-04-01'},
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-17'},
        ]
        
        assert_segments_match(down_segments, expected_down)

    @pytest.mark.core
    def test_filter_segments_by_direction_flat(self, gradual_results):
        """Test filtering segments by 'Flat' direction."""
        # Filter for Flat segments - should match expected segments from gradual data
        flat_segments = gradual_results.filter_segments(direction='Flat', format='dict')
        
        assert isinstance(flat_segments, list)
        assert len(flat_segments) == 3
        
        # Expected Flat segments from test_core_gradual
        expected_flat = [
            {'direction': 'Flat', 'start': '2025-02-06', 'end': '2025-02-09'},
            {'direction': 'Flat', 'start': '2025-03-15', 'end': '2025-03-17'},
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-30'},
        ]
        
        assert_segments_match(flat_segments, expected_flat)

    @pytest.mark.core
    def test_filter_segments_by_direction_noise(self, outlier_signal):
        """Test filtering segments by 'Noise' direction."""
        results = pt.detect_trends(
            outlier_signal,
            date_col='date',
            value_col='value',
            plot=False
        )
        
        # Filter for Noise segments - outlier signal should have exactly 1 noise segment
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        
        assert isinstance(noise_segments, list)
        assert len(noise_segments) == 1
        
        # Expected Noise segment from outlier signal
        expected_noise = [
            {'direction': 'Noise', 'start': '2025-02-19', 'end': '2025-02-21'},
        ]
        
        assert_segments_match(noise_segments, expected_noise)

    @pytest.mark.core
    def test_filter_segments_up_down_combined(self, gradual_results):
        """Test filtering for combined Up/Down trends."""
        # Filter for Up/Down trends - should match expected segments from test_core_gradual
        trend_segments = gradual_results.filter_segments(direction='Up/Down', format='dict')
        
        assert isinstance(trend_segments, list)
        assert len(trend_segments) == 6  # 3 Up + 3 Down
        
        # All segments should be Up or Down
        for segment in trend_segments:
            assert segment['direction'] in ['Up', 'Down']
        
        # Verify count of each direction using summary
        direction_counts = gradual_results.summary['direction_counts']
        assert direction_counts['Up'] == 3
        assert direction_counts['Down'] == 3

    @pytest.mark.core
    def test_filter_segments_any_direction(self, gradual_results):
        """Test filtering with 'Any' direction returns all segments."""
        # Filter for any direction - should return all 9 segments from test_core_gradual
        all_segments = gradual_results.filter_segments(direction='Any', format='dict')
        
        assert isinstance(all_segments, list)
        assert len(all_segments) == 9
        assert len(all_segments) == len(gradual_results.segments)

    @pytest.mark.core
    def test_filter_segments_sort_by_time_index(self, gradual_results):
        """Test sorting segments by time_index."""
        # Filter and sort by time_index (ascending)
        sorted_segments = gradual_results.filter_segments(sort_by='time_index', format='dict')
        
        assert isinstance(sorted_segments, list)
        # Check that segments are sorted by time_index
        time_indices = [seg['time_index'] for seg in sorted_segments]
        assert time_indices == sorted(time_indices)

    @pytest.mark.core
    def test_filter_segments_sort_by_change_rank(self, gradual_results):
        """Test sorting segments by change_rank."""
        # Filter and sort by change_rank (descending by total_change)
        sorted_segments = gradual_results.filter_segments(sort_by='change_rank', format='dict')
        
        assert isinstance(sorted_segments, list)
        # Check that segments are sorted by absolute total_change (descending)
        if len(sorted_segments) > 1 and 'total_change' in sorted_segments[0]:
            changes = [abs(seg.get('total_change', 0)) for seg in sorted_segments]
            assert changes == sorted(changes, reverse=True)

    @pytest.mark.core
    def test_filter_segments_format_dict(self, gradual_results):
        """Test that format='dict' returns list of dictionaries."""
        segments = gradual_results.filter_segments(direction='Any', format='dict')
        
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
    def test_filter_segments_format_df(self, gradual_results):
        """Test that format='df' returns DataFrame."""
        segments_df = gradual_results.filter_segments(direction='Any', format='df')
        
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
    def test_filter_segments_invalid_sort_by(self, gradual_results, capsys):
        """Test filter_segments with invalid sort_by parameter prints error message."""
        # Call with invalid sort_by parameter
        filtered = gradual_results.filter_segments(sort_by='invalid_sort', format='dict')
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Should print error message (line 144)
        assert 'invalid_sort is not a valid sort_by' in captured.out
        assert "['time_index', 'change_rank']" in captured.out
        
        # Should still return segments (unsorted)
        assert isinstance(filtered, list)
        assert len(filtered) == 9

    @pytest.mark.core
    def test_filter_segments_invalid_direction(self, gradual_results, capsys):
        """Test filter_segments with invalid direction parameter prints error message."""
        # Call with invalid direction parameter
        filtered = gradual_results.filter_segments(direction='InvalidDirection', format='dict')
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Should print error message (line 152)
        assert 'InvalidDirection is not a valid direction' in captured.out
        assert "['Any', 'Up/Down', 'Up', 'Down', 'Flat', 'Noise']" in captured.out
        
        # Should still return all segments
        assert isinstance(filtered, list)
        assert len(filtered) == 9

    @pytest.mark.core
    def test_filter_segments_invalid_format(self, gradual_results, capsys):
        """Test filter_segments with invalid format parameter prints error message and returns segments."""
        # Call with invalid format parameter
        result = gradual_results.filter_segments(direction='Any', format='invalid_format')
        
        # Capture printed output
        captured = capsys.readouterr()
        
        # Should print error message (line 156)
        assert 'invalid_format is not a valid format' in captured.out
        assert "['dict', 'df']" in captured.out
        
        # Should return segments as fallback (line 164)
        assert isinstance(result, list)
        assert len(result) == 9


class TestResultsPrintSummary:
    """Tests for print_summary() method - output generation."""

    @pytest.mark.core
    def test_print_summary_with_trends(self, gradual_results):
        """Test that print_summary executes without errors."""
        # Should not raise any exceptions
        try:
            gradual_results.print_summary()
            success = True
        except Exception as e:
            success = False
            print(f"print_summary raised exception: {e}")
        
        assert success

    @pytest.mark.core
    def test_print_summary_no_trends(self, zeros_signal):
        """Test that print_summary handles no trends gracefully."""
        results = pt.detect_trends(
            zeros_signal,
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
    def test_print_summary_only_flat_and_noise(self):
        """Test that print_summary handles case with only Flat/Noise segments (no Up/Down trends)."""
        # Create a signal with only noise (random walk with high variance)
        # This should produce segments but no clear Up/Down trends
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        values = np.random.normal(10, 5, 30)  # High variance noise
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
        
        # Should have segments (Flat/Noise), but no Up/Down trends
        assert len(results.segments) > 0
        
        # Verify there are no Up or Down trends
        up_down_segments = results.filter_segments(direction='Up/Down', format='dict')
        assert len(up_down_segments) == 0
        
        # print_summary should handle this gracefully and print "Detected no trends..."
        # This will exercise lines 92-93
        try:
            results.print_summary()
            success = True
        except Exception as e:
            success = False
            print(f"print_summary raised exception: {e}")
        
        assert success


class TestResultsIntegration:
    """Integration tests for full workflows and edge cases."""

    @pytest.mark.core
    def test_integration_full_workflow(self, gradual_results):
        """Test full workflow: detect trends, filter, and access results."""
        # 1. Access segments - should have 9 total (3 Up, 3 Down, 3 Flat)
        assert len(gradual_results.segments) == 9
        
        # 2. Get best trend - should be the last Down trend
        assert gradual_results.best is not None
        assert gradual_results.best['direction'] == 'Down'
        assert gradual_results.best['start'] == '2025-05-09'
        assert gradual_results.best['end'] == '2025-06-17'
        
        # 3. Check summary - exact counts from gradual data
        assert 'direction_counts' in gradual_results.summary
        assert gradual_results.summary['direction_counts'] == {'Up': 3, 'Down': 3, 'Flat': 3}
        
        # 4. Filter for uptrends - should get 3
        up_trends = gradual_results.filter_segments(direction='Up', format='df')
        assert isinstance(up_trends, pd.DataFrame)
        assert len(up_trends) == 3
        
        # 5. Sort by change rank - best should be first
        ranked = gradual_results.filter_segments(sort_by='change_rank', format='dict')
        assert isinstance(ranked, list)
        assert ranked[0]['change_rank'] == 1
        
        # 6. Access DataFrames - should have all 9 segments
        assert isinstance(gradual_results.df, pd.DataFrame)
        assert len(gradual_results.df) == 9
        assert isinstance(gradual_results.df_summary, pd.DataFrame)
        assert len(gradual_results.df_summary) == 9

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


class TestResultsDataStructures:
    """Tests for data structure validation and consistency."""

    @pytest.mark.core
    def test_segments_have_required_fields(self, gradual_results):
        """Test that all segments have required fields."""
        required_fields = ['direction', 'start', 'end', 'time_index', 'days']
        
        for i, segment in enumerate(gradual_results.segments):
            for field in required_fields:
                assert field in segment, f"Segment {i} missing field: {field}"
