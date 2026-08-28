"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
One extra test included to assess plt.show() behaviour only... for test coverage
"""

import pytest
import pandas as pd
from copy import deepcopy
from conftest import build_internal_index
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
        return plot_pytrendy(df=df, value_col=value_col, segments_enhanced=segments, suppress_show=suppress_show)

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
        external_index, internal_index, index_lookup = build_internal_index(df, date_col)
    
        df[date_col] = internal_index
        df.set_index(date_col, inplace=True)
        df = df[[value_col]]
        method_params = {'abrupt_padding': 28, 'avoid_noise': True}

        df = process_signals(df, value_col, method_params)
        segments = get_segments(df)

        # ------------------ refine_segments()
        # unwrapped-equivalent to disable grouping at a lower level  
        segments_refined = deepcopy(segments)
        segments_refined = classify_trends(df, value_col, segments_refined)
        # No grouping code in between these steps
        segments_refined = expand_contract_segments(df, value_col, segments_refined, method_params) # for gradual
        segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt
        segments_refined = clean_artifacts(df, value_col, segments_refined, method_params) # cleans overlaps etc from expand/contract
        # No grouping code & further post-processing after these steps

        # ------ pt.detect_trends() [part 2]
        segments = segments_refined.copy()
        segments = analyse_segments(df, value_col, segments)

        for segment in segments:
            segment['start'] = index_lookup[segment['start']]
            segment['end'] = index_lookup[segment['end']]

        df[date_col] = external_index
        df.set_index(date_col, inplace=True)
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



# =============================================================================
# plot_pytrendy: boundary segments (first/last segment, non-neighbouring gaps)
# =============================================================================

class TestPlotBoundarySegments:
    """Segment positioning at the edges of the index and gaps between segments."""

    def test_first_segment_at_boundary_string(self):
        """String index: first segment starts at index[0] (no prev)."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        segments = [
            {'start': str_idx[0], 'end': str_idx[5], 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 1},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_last_segment_at_boundary_string(self):
        """String index: last segment ends at index[-1] (no next)."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        segments = [
            {'start': str_idx[-10], 'end': str_idx[-1], 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 1},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_non_neighbouring_segments_string(self):
        """String index: segments with gaps (not adjacent)."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        segments = [
            {'start': str_idx[1], 'end': str_idx[5], 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 1},
            {'start': str_idx[20], 'end': str_idx[25], 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

# =============================================================================
# plot_pytrendy: plot customisation (plot_params branches)
# =============================================================================

class TestPlotCustomization:
    """plot_params customisation branches (figsize/title/labels, legend, colours)."""

    def _date_plot_df(self):
        """Build a datetime-indexed DataFrame from synthetic data."""
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')[['gradual']]

    def test_plot_with_custom_params(self):
        """Test plot_params path for date index type."""
        plot_df = self._date_plot_df()
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', date_col='date',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_params = {
            'figsize': (10, 3),
            'title': 'Custom Title',
            'xlabel': 'Custom X',
            'ylabel': 'Custom Y',
            'grid': {'visible': False},
        }

        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='date',
                            suppress_show=True, plot_params=plot_params)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_legend(self):
        """Test legend customisation path."""
        plot_df = self._date_plot_df()
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', date_col='date',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_params = {
            'legend_loc': 'lower right',
        }

        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='date',
                            suppress_show=True, plot_params=plot_params)
        assert fig is not None
        plt.close(fig)

    def test_plot_with_custom_colors(self):
        """Test custom colors path."""
        plot_df = self._date_plot_df()
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', date_col='date',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_params = {
            'colors': {'Up': 'lightgreen', 'Down': 'lightcoral'},
        }

        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='date',
                            suppress_show=True, plot_params=plot_params)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_pytrendy: prev fill branch (lines 172-178, 182)
# =============================================================================

class TestPlotPrevFillDirect:
    """Test the prev fill branch in plot_pytrendy when start displacement is invalid.

    TODO: these examples use hand-crafted segment lists that are a bit contrived
    to force the specific displacement conditions. Redo with more realistic
    synthetic scenarios when a natural dataset produces these patterns.
    """

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_string_prev_fill_direct.png',
                                    style='default')
    def test_string_prev_fill_direct(self):
        """Lines 172-176, 182: string index, Flat→Up adjacent, invalid start displacement."""
        # Custom data with a dip so Up start value < Flat end value
        values = list(range(40))
        values[19] = 25  # Flat end value (high)
        values[20] = 15  # Up start value (low) — displacement invalid
        df = pd.DataFrame({'date': [f'S{i}' for i in range(40)], 'gradual': values})
        pt.detect_trends(df, date_col='date', value_col='gradual',
                         plot=False, method_params={'abrupt_padding': 0})

        # Craft segments: Flat S10-S19 (value 25 at end), adjacent Up S20-S35
        # Up start value (15) < Flat end value (25) makes displacement invalid
        str_idx = [f'S{i}' for i in range(40)]
        plot_df = pd.DataFrame({'gradual': values}, index=str_idx)
        segments = [
            {'start': 'S10', 'end': 'S19', 'direction': 'Flat',
             'change_rank': 1},
            {'start': 'S20', 'end': 'S35', 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_integer_prev_fill_direct.png',
                                    style='default')
    def test_integer_prev_fill_direct(self):
        """Lines 177-178: integer index, Flat→Up adjacent, invalid start displacement."""
        values = list(range(40))
        values[19] = 25  # Flat end value (high)
        values[20] = 15  # Up start value (low) — displacement invalid
        df = pd.DataFrame({'gradual': values})
        pt.detect_trends(df, value_col='gradual',
                         plot=False, method_params={'abrupt_padding': 0})

        plot_df = df[['gradual']]
        segments = [
            {'start': 10, 'end': 19, 'direction': 'Flat',
             'change_rank': 1},
            {'start': 20, 'end': 35, 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        return fig


# =============================================================================
# plot_pytrendy: next noise fill branch (lines 210-216)
# =============================================================================

class TestPlotNextNoiseFillDirect:
    """Test the next noise fill branch in plot_pytrendy when end displacement is invalid.

    TODO: these examples use hand-crafted segment lists that are a bit contrived
    to force the specific displacement conditions. Redo with more realistic
    synthetic scenarios when a natural dataset produces these patterns.
    """

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_date_next_noise_fill_direct.png',
                                    style='default')
    def test_date_next_noise_fill_direct(self):
        """Line 211: date index, Down→Noise adjacent, invalid end displacement."""
        # Custom data where Down end value < next value (invalid for Down)
        values = list(range(40))
        values[24] = 10  # Down end value (low)
        values[25] = 35  # Noise start value (high) — displacement invalid
        df = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=40, freq='D'),
                           'gradual': values})
        pt.detect_trends(df, date_col='date', value_col='gradual',
                         plot=False, method_params={'abrupt_padding': 0})

        plot_df = df.set_index('date')[['gradual']]
        segments = [
            {'start': pd.Timestamp('2025-01-02'), 'end': pd.Timestamp('2025-01-25'),
             'direction': 'Down', 'trend_class': 'gradual', 'change_rank': 1},
            {'start': pd.Timestamp('2025-01-26'), 'end': pd.Timestamp('2025-02-05'),
             'direction': 'Noise', 'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='date', suppress_show=True)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_string_next_noise_fill_direct.png',
                                    style='default')
    def test_string_next_noise_fill_direct(self):
        """Lines 213-214: string index, Down→Noise adjacent, invalid end displacement."""
        values = list(range(40))
        values[24] = 10  # Down end value (low)
        values[25] = 35  # Noise start value (high) — displacement invalid
        df = pd.DataFrame({'date': [f'S{i}' for i in range(40)], 'gradual': values})
        pt.detect_trends(df, date_col='date', value_col='gradual',
                         plot=False, method_params={'abrupt_padding': 0})

        str_idx = [f'S{i}' for i in range(40)]
        plot_df = pd.DataFrame({'gradual': values}, index=str_idx)
        segments = [
            {'start': 'S1', 'end': 'S24', 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 1},
            {'start': 'S25', 'end': 'S35', 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_integer_next_noise_fill_direct.png',
                                    style='default')
    def test_integer_next_noise_fill_direct(self):
        """Line 216: integer index, Down→Noise adjacent, invalid end displacement."""
        values = list(range(40))
        values[24] = 10  # Down end value (low)
        values[25] = 35  # Noise start value (high) — displacement invalid
        df = pd.DataFrame({'gradual': values})
        pt.detect_trends(df, value_col='gradual',
                         plot=False, method_params={'abrupt_padding': 0})

        plot_df = df[['gradual']]
        segments = [
            {'start': 1, 'end': 24, 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 1},
            {'start': 25, 'end': 35, 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        return fig
