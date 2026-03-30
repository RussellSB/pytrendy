import pytest
import pandas as pd
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy

class TestCustomPlotParams:
    """Test custom plot parameters for plot visualization."""

    def _prepare_and_plot(self, df, value_col, segments, **kwargs):
        """Helper to prepare dataframe and create plot."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[[value_col]]
        return plot_pytrendy(df, value_col, segments, suppress_show=True, **kwargs)

    @pytest.mark.core
    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_custom_plot_params_figsize.png', style='default', remove_text=True)
    def test_custom_plot_params_figsize(self):
        """Test custom figsize in plot parameters."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False)
        )
        
        plot_params = {
            'figsize': (16, 8),
            'title': "Custom Plot Title"
        }
        
        fig = self._prepare_and_plot(df, 'gradual', results.segments, plot_params=plot_params)
        return fig
