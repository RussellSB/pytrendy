"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
"""

import pytest
import pandas as pd
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class TestPlotPytrendyCore:
    """Test core cases for plot visualization on synthetic data."""

    def _prepare_and_plot(self, df, value_col, segments):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show=True)

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_gradual_trends.png')
    def test_plot_gradual_trends(self):
        """Test visualization of gradual trends in synthetic data."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        fig = self._prepare_and_plot(df, 'gradual', results.segments)
        return fig

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_trends_no_padding.png')
    def test_plot_abrupt_trends_no_padding(self):
        """Test visualization of abrupt trends without padding."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_abrupt_trends_with_padding.png')
    def test_plot_abrupt_trends_with_padding(self):
        """Test visualization of abrupt trends with padding enabled."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        fig = self._prepare_and_plot(df, 'abrupt', results.segments)
        return fig
