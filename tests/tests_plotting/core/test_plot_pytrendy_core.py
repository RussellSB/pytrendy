"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
"""

import pytest
import pandas as pd
import pytrendy as pt
import matplotlib.pyplot as plt
from pytrendy.io.plot_pytrendy import plot_pytrendy


class TestPlotPytrendyCore:
    """Test core cases for plot visualization on synthetic data."""

    def _prepare_and_plot(self, df, value_col, segments, plot_params=None):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show=True, plot_params=plot_params)

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_gradual_trends.png', style='default')
    def test_plot_gradual_trends(self):
        """Test visualization of gradual trends in synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False
        )
        
        fig = self._prepare_and_plot(df, 'gradual', results.segments)
        return fig

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_trends_no_padding.png', style='default')
    def test_plot_abrupt_trends_no_padding(self):
        """Test visualization of abrupt trends without padding."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False
        )
        
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_trends_with_padding.png', style='default')
    def test_plot_abrupt_trends_with_padding(self):
        """Test visualization of abrupt trends with padding enabled."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params={'abrupt_padding': 28}
        )
        
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig

    @pytest.mark.core
    @pytest.mark.plot
    def test_plot_with_custom_params(self):
        """Test visualisation with custom plot_params."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=False)
        custom_params = {'figsize': (10, 4), 'title': 'Custom Title', 'colors': {'Up': 'lightpink'}}
        fig = self._prepare_and_plot(df, 'gradual', results.segments, plot_params=custom_params)
        assert fig.get_size_inches()[0] == 10
        assert fig.axes[0].get_title() == 'Custom Title'
        plt.close(fig)