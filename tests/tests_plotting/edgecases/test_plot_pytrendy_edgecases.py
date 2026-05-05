"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
One extra test included to assess plt.show() behaviour only... for test coverage
"""

import pytest
import pandas as pd
from copy import deepcopy
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
from pytrendy.process_signals import process_signals
from pytrendy.post_processing.segments_get import get_segments
from pytrendy.post_processing.segments_analyse import analyse_segments
from pytrendy.post_processing.segments_refine.trend_classify import classify_trends
from pytrendy.post_processing.segments_refine.gradual_expand_contract import expand_contract_segments
from pytrendy.post_processing.segments_refine.abrupt_shaving import shave_abrupt_trends
from pytrendy.post_processing.segments_refine.artifact_cleanup import clean_artifacts
import matplotlib.pyplot as plt


class TestPlotPytrendyEdgeCases:
    """Test edgecases for plot visualization on synthetic data."""

    def _prepare_and_plot(self, df, value_col, segments, suppress_show=True):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show)

    def _synth_1_data(self):
        """Helper to load and prepare synthetic dataset 1 (abrupt, base, no spikes)."""
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        return df.reset_index()


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_base_no_spikes.png', style='default')
    def test_plot_abrupt_base_no_spikes(self):
        """Test visualization of abrupt trends synthetic with no spikes (synth 1), for plot code coverage."""
        df = self._synth_1_data()
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_debug_add_vertical_lines.png', style='default')
    def test_plot_debug_add_vertical_lines(self):
        """Same as previous unit test (synth 1), except tests statements that add lines in plot when grouping disabled."""
        # TODO: organise in a cleaner code way, so can simply be toggled off for a higher level, will also allow more customisable pipeline
        date_col = 'date'
        value_col = 'abrupt'
        df = self._synth_1_data()
        
        # ------ pt.detect_trends() [part 1]
        # unwrapped-equivalent to disable grouping at a lower level     
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        df = df[[value_col]]
        method_params = {'abrupt_padding':0}

        df = process_signals(df, value_col)
        segments = get_segments(df)

        # ------------------ refine_segments()
        # unwrapped-equivalent to disable grouping at a lower level  
        segments_refined = deepcopy(segments)
        segments_refined = classify_trends(df, value_col, segments_refined)
        # No grouping code in between these steps
        segments_refined = expand_contract_segments(df, value_col, segments_refined) # for gradual
        segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt
        segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans overlaps etc from expand/contract
        # No grouping code & further post-processing after these steps

        # ------ pt.detect_trends() [part 2]
        segments = segments_refined.copy()
        segments = analyse_segments(df, value_col, segments)
        fig = plot_pytrendy(df, value_col, segments, suppress_show=True)
        return fig


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_noisy_edgecase_7.png', style='default')
    def test_plot_noisy_edgecase_7(self):
        """Test visualization of noisy edgecase 7, for plot code coverage."""
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            edgecases_df,
            date_col='date',
            value_col='noisy_edgecase_7',
            plot=False
        )
        
        fig = self._prepare_and_plot(edgecases_df, 'noisy_edgecase_7', results.segments)
        return fig


    def test_plot_show_behavior(self, monkeypatch):
        """
        Test that plot_pytrendy triggers plt.show() when suppress_show=False.
        We use monkeypatch to replace plt.show with a fake function that records calls.
        When verified to be called once, we can be confident that the plot is being displayed as expected.
        """
        show_calls = []
        def fake_show(*args, **kwargs):
            show_calls.append((args, kwargs))
        monkeypatch.setattr(plt, 'show', fake_show)

        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False
        )
        self._prepare_and_plot(df, 'gradual', results.segments, suppress_show=False) # False, triggers plt.show()
        assert len(show_calls) == 1

