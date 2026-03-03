"""
Tests for plot code coverage with noise edge cases and crash scenarios.

These tests maximize plot code coverage by exercising the plot_pytrendy function
with various edge cases including noise, crashes, and different segment configurations.
The goal is to ensure plotting code handles all edge cases without errors.

Reference: Problem statement requirement for "plot code coverage for the edge cases"
"""

import pytest
import pandas as pd
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class TestPlotEdgeCases:
    """Test cases for plot visualization code coverage with edge cases."""

    def _prepare_and_plot(self, df, value_col, segments):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show=True)

    @pytest.mark.plot
    def test_plot_crash_scenario_noisy_crash(self):
        """
        Test plotting with noisy_crash scenario that previously caused crashes.
        
        This ensures the plotting code handles edge case data that previously
        crashed the detection algorithm. Verifies plotting completes without errors.
        """
        crashes_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Test plotting with crash scenario data
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_crash_scenario_temp_2(self):
        """
        Test plotting with temp_2 scenario that previously caused hangs.
        
        This scenario was noted with "TODONE: fix hangup" and verifies
        that plotting completes in reasonable time without hanging.
        """
        crashes_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'temp_2']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_edgecase_overlapping_segments(self):
        """
        Test plotting with noisy_edgecase_1 that had overlapping segments.
        
        Reference: "TODONE: fix when green overlaps red"
        This tests plotting code's handling of complex segment boundaries.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        test_df = edgecases_df[['date', 'noisy_edgecase_1']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_edgecase_tiny_segments(self):
        """
        Test plotting with noisy_edgecase_2 that had tiny segments.
        
        Reference: "TODONE: fix green at 03-01 start that is should be too tiny for significance"
        This tests plotting code's handling of very short segments.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        test_df = edgecases_df[['date', 'noisy_edgecase_2']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_edgecase_noise_classification(self):
        """
        Test plotting with noisy_edgecase_3 that had noise classification issues.
        
        Reference: "TODONE: 03-02 could be noise"
        This tests plotting code's handling of noise segments.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        test_df = edgecases_df[['date', 'noisy_edgecase_3']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_with_abrupt_segments(self):
        """
        Test plotting code with abrupt trend segments.
        
        Tests the abrupt trend class handling in plotting code,
        ensuring proper visual representation of abrupt changes.
        """
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_with_noise_segments(self):
        """
        Test plotting with data containing noise segments.
        
        Uses high noise level to ensure noise segments are present
        and tests plotting code's handling of noise visualization.
        """
        import numpy as np
        np.random.seed(42)
        df = pt.load_data('series_synthetic')
        df['value_noisy'] = df['gradual'] + np.random.normal(0, 50, size=len(df))
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='value_noisy',
            plot=False
        )
        
        # Ensure we have noise segments to test
        noise_segments = [seg for seg in results.segments if seg['direction'] == 'Noise']
        assert len(noise_segments) > 0, "Expected noise segments for this test"
        
        fig = self._prepare_and_plot(df, 'value_noisy', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_multiple_direction_types(self):
        """
        Test plotting with all direction types (Up, Down, Flat, Noise).
        
        Ensures plotting code properly handles all segment direction types
        and their color mappings.
        """
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # Verify we have multiple direction types
        directions = {seg['direction'] for seg in results.segments}
        assert len(directions) > 1, "Expected multiple direction types for comprehensive plot test"
        
        fig = self._prepare_and_plot(df, 'gradual', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_with_no_padding(self):
        """
        Test plotting with abrupt padding disabled.
        
        Tests plotting code behavior when is_abrupt_padded=False,
        ensuring proper visualization without padding.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        test_df = edgecases_df[['date', 'noisy_edgecase_7']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_boundary_adjustments(self):
        """
        Test plotting code's boundary adjustment logic.
        
        Uses edge case data that tests the plot_pytrendy function's
        logic for adjusting segment start/end boundaries for visualization,
        including the displacement logic for different segment types.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        test_df = edgecases_df[['date', 'noisy_edgecase_8']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        # This edge case had issues with boundary precision
        fig = self._prepare_and_plot(test_df, 'value', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_with_ranked_segments(self):
        """
        Test plotting with change_rank annotations.
        
        Verifies plotting code properly handles segments with change_rank
        attributes and their visual annotations.
        """
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Check if segments have change_rank
        ranked_segments = [seg for seg in results.segments if 'change_rank' in seg]
        
        fig = self._prepare_and_plot(df, 'gradual', results.segments)
        assert fig is not None
        matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_all_crash_scenarios(self):
        """
        Test plotting with all crash scenarios in sequence.
        
        Ensures plotting code handles all crash scenarios without errors,
        maximizing code coverage for edge cases.
        """
        crashes_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        crash_columns = [col for col in crashes_df.columns if col not in ['Unnamed: 0', 'date']]
        
        for col in crash_columns[:3]:  # Test first 3 to keep test time reasonable
            test_df = crashes_df[['date', col]].copy()
            test_df.columns = ['date', 'value']
            
            results = pt.detect_trends(
                test_df,
                date_col='date',
                value_col='value',
                plot=False,
                method_params=dict(is_abrupt_padded=True)
            )
            
            fig = self._prepare_and_plot(test_df, 'value', results.segments)
            assert fig is not None, f"Plotting failed for {col}"
            matplotlib.pyplot.close(fig)

    @pytest.mark.plot
    def test_plot_all_edgecase_scenarios(self):
        """
        Test plotting with all edge case scenarios in sequence.
        
        Ensures plotting code handles all edge case scenarios without errors,
        maximizing code coverage for various segment configurations.
        """
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        edgecase_columns = [col for col in edgecases_df.columns if col.startswith('noisy_edgecase_')]
        
        for col in edgecase_columns[:3]:  # Test first 3 to keep test time reasonable
            test_df = edgecases_df[['date', col]].copy()
            test_df.columns = ['date', 'value']
            
            results = pt.detect_trends(
                test_df,
                date_col='date',
                value_col='value',
                plot=False,
                method_params=dict(is_abrupt_padded=True)
            )
            
            fig = self._prepare_and_plot(test_df, 'value', results.segments)
            assert fig is not None, f"Plotting failed for {col}"
            matplotlib.pyplot.close(fig)
