"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
"""

import pytest
import pandas as pd
import pytrendy as pt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class TestPlotPytrendy:
    """Test cases for plot visualization on synthetic data."""

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
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
        
        # Prepare the dataframe as detect_trends does
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        df_copy = df_copy[['gradual']]
        
        # Process signals to match what detect_trends does
        from pytrendy.process_signals import process_signals
        df_processed = process_signals(df_copy, 'gradual')
        
        # Create the plot and return the figure
        from pytrendy.io.plot_pytrendy import plot_pytrendy
        fig = plot_pytrendy(df_processed, 'gradual', results.segments, return_fig=True)
        
        return fig

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
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
        
        # Prepare the dataframe as detect_trends does
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        df_copy = df_copy[['abrupt']]
        
        # Process signals to match what detect_trends does
        from pytrendy.process_signals import process_signals
        df_processed = process_signals(df_copy, 'abrupt')
        
        # Create the plot and return the figure
        from pytrendy.io.plot_pytrendy import plot_pytrendy
        fig = plot_pytrendy(df_processed, 'abrupt', results.segments, return_fig=True)
        
        return fig

    @pytest.mark.mpl_image_compare(baseline_dir='baseline')
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
        
        # Prepare the dataframe as detect_trends does
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy.set_index('date', inplace=True)
        df_copy = df_copy[['abrupt']]
        
        # Process signals to match what detect_trends does
        from pytrendy.process_signals import process_signals
        df_processed = process_signals(df_copy, 'abrupt')
        
        # Create the plot and return the figure
        from pytrendy.io.plot_pytrendy import plot_pytrendy
        fig = plot_pytrendy(df_processed, 'abrupt', results.segments, return_fig=True)
        
        return fig
