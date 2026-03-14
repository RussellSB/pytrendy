"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
One extra test included to assess plt.show() behaviour only... for test coverage
"""

import pytest
import pandas as pd
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend


class TestPlotPytrendyEdgeCases:
    """Test edgecases for plot visualization on synthetic data."""

    def _prepare_and_plot(self, df, value_col, segments, suppress_show=True):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show)


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_base_no_spikes.png')
    def test_plot_abrupt_base_no_spikes(self):
        """Test visualization of abrupt trends synthetic with no spikes (synth 1), for plot code coverage."""
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-11', 'abrupt'] = 0
        df.loc['2025-02-16':'2025-03-10', 'abrupt'] = 125
        df.loc['2025-03-18':'2025-04-15', 'abrupt'] = 150
        df.loc['2025-03-20':'2025-04-22', 'abrupt'] = 250
        df.loc['2025-03-25':'2025-04-01', 'abrupt'] = 200
        df = df.reset_index() # must be reset for _pepare_and_plot()

        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_noisy_edgecase_7.png')
    def test_plot_noisy_edgecase_7(self):
        """Test visualization of noisy edgecase 7, for plot code coverage."""
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            edgecases_df,
            date_col='date',
            value_col='noisy_edgecase_7',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
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
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        self._prepare_and_plot(df, 'gradual', results.segments, suppress_show=False)
        assert len(show_calls) == 1