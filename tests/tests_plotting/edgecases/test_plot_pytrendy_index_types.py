"""
Tests for plot_pytrendy index-type branches.

Segments are sourced from ``detect_trends`` wherever a natural dataset produces
the branch under test; the index-specific neighbour topology is then rendered
through ``plot_pytrendy`` and compared against an mpl baseline.

One direct-call test is retained: ``test_noise_next_neighbouring_adjustment``
needs two adjacent same-direction segments with a non-date index, which the
detection pipeline does not produce (segments alternate direction), so the
``line_date = seg['end']`` branch is only reachable with a hand-built list.
"""

import pytest
import pandas as pd
import numpy as np

import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
from conftest import assert_segments_in_a_haystack


def _synthetic(column):
    """Load a column from the packaged synthetic dataset."""
    return pt.load_data('series_synthetic')[[column]].copy()


def _edgecase(column):
    """Load a column from the noisy-edgecases test fixture."""
    df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
    return df[[column]].copy()


def _with_string_index(df):
    """Return ``(df, date_col)`` with a string index column prepended."""
    df.insert(0, 'idx', [f'S{i}' for i in range(len(df))])
    return df, 'idx'


def _with_integer_index(df):
    """Return ``(df, date_col)`` with an integer index column prepended."""
    df.insert(0, 'idx', np.arange(len(df)))
    return df, 'idx'


def _with_float_index(df):
    """Return ``(df, date_col)`` with a float index column prepended."""
    df.insert(0, 'idx', np.linspace(0, 1, len(df)))
    return df, 'idx'


# =============================================================================
# plot_pytrendy: string index branches
# =============================================================================

class TestPlotStringIndex:
    """Exercise the index_type=='string' branches in plot_pytrendy."""

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_string_index_basic.png',
                                   style='default')
    def test_string_index_basic(self):
        """String index: gradual trends on a string-spaced index."""
        data, date_col = _with_string_index(_synthetic('gradual'))
        results = pt.detect_trends(data, value_col='gradual', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 'S1', 'end': 'S23'},
            {'direction': 'Flat', 'start': 'S168', 'end': 'S180'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['gradual']], 'gradual',
                             results.segments, index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_string_index_adjacent.png',
                                   style='default')
    def test_string_index_with_adjacent_segments(self):
        """String index: neighbouring segments exercise prev/next adjacency logic."""
        data, date_col = _with_string_index(_edgecase('noisy_edgecase_3'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_3', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Noise', 'start': 'S8', 'end': 'S9'},
            {'direction': 'Flat', 'start': 'S10', 'end': 'S14'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['noisy_edgecase_3']],
                             'noisy_edgecase_3', results.segments,
                             index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_string_index_abrupt.png',
                                   style='default')
    def test_string_index_abrupt_segment(self):
        """String index: abrupt segments trigger start/end adjustment branches."""
        data, date_col = _with_string_index(_synthetic('abrupt'))
        results = pt.detect_trends(data, value_col='abrupt', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 'S58', 'end': 'S59'},
            {'direction': 'Down', 'start': 'S121', 'end': 'S124'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['abrupt']], 'abrupt',
                             results.segments, index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_prev_fill_string.png',
                                   style='default')
    def test_string_index_noise_segment(self):
        """String index: noise segments trigger the noise branches."""
        data, date_col = _with_string_index(_edgecase('noisy_edgecase_4'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_4', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 'S80', 'end': 'S95'},
            {'direction': 'Noise', 'start': 'S96', 'end': 'S98'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['noisy_edgecase_4']],
                             'noisy_edgecase_4', results.segments,
                             index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_string_index_flat.png',
                                   style='default')
    def test_string_index_flat_segment(self):
        """String index: flat segments (no trend_class) render without adjustment."""
        data, date_col = _with_string_index(_edgecase('noisy_edgecase_1'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_1', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 'S0', 'end': 'S4'},
            {'direction': 'Noise', 'start': 'S5', 'end': 'S58'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['noisy_edgecase_1']],
                             'noisy_edgecase_1', results.segments,
                             index_type='string', suppress_show=True)


# =============================================================================
# plot_pytrendy: integer index branches
# =============================================================================

class TestPlotIntegerIndex:
    """Exercise the integer index branches in plot_pytrendy."""

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_integer_index_basic.png',
                                   style='default')
    def test_integer_index_basic(self):
        """Integer index: gradual trends on an integer-spaced index."""
        data, date_col = _with_integer_index(_synthetic('gradual'))
        results = pt.detect_trends(data, value_col='gradual', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ])
        return plot_pytrendy(data.set_index(date_col)[['gradual']], 'gradual',
                             results.segments, index_type='integer', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_integer_index_adjacent.png',
                                   style='default')
    def test_integer_index_with_adjacent_segments(self):
        """Integer index: neighbouring segments exercise prev/next adjacency logic."""
        data, date_col = _with_integer_index(_edgecase('noisy_edgecase_3'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_3', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Noise', 'start': 8, 'end': 9},
            {'direction': 'Flat', 'start': 10, 'end': 14},
        ])
        return plot_pytrendy(data.set_index(date_col)[['noisy_edgecase_3']],
                             'noisy_edgecase_3', results.segments,
                             index_type='integer', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_integer_index_abrupt.png',
                                   style='default')
    def test_integer_index_abrupt_segment(self):
        """Integer index: abrupt segments trigger end adjustment branches."""
        data, date_col = _with_integer_index(_synthetic('abrupt'))
        results = pt.detect_trends(data, value_col='abrupt', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 58, 'end': 59},
            {'direction': 'Down', 'start': 121, 'end': 124},
        ])
        return plot_pytrendy(data.set_index(date_col)[['abrupt']], 'abrupt',
                             results.segments, index_type='integer', suppress_show=True)


# =============================================================================
# plot_pytrendy: float index branches
# =============================================================================

class TestPlotFloatIndex:
    """Exercise the float index branches in plot_pytrendy."""

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_float_index_basic.png',
                                   style='default')
    def test_float_index_basic(self):
        """Float index: gradual trends on a dense float-spaced index."""
        data, date_col = _with_float_index(_synthetic('gradual'))
        results = pt.detect_trends(data, value_col='gradual', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        plot_df = data.set_index(date_col)[['gradual']]
        idx = list(plot_df.index)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': idx[1], 'end': idx[23]},
            {'direction': 'Flat', 'start': idx[36], 'end': idx[39]},
        ])
        return plot_pytrendy(plot_df, 'gradual',
                             results.segments, index_type='float', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_float_index_adjacent.png',
                                   style='default')
    def test_float_index_with_adjacent_segments(self):
        """Float index: neighbouring segments exercise prev/next adjacency logic."""
        data, date_col = _with_float_index(_edgecase('noisy_edgecase_3'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_3', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        plot_df = data.set_index(date_col)[['noisy_edgecase_3']]
        idx = list(plot_df.index)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Noise', 'start': idx[8], 'end': idx[9]},
            {'direction': 'Flat', 'start': idx[10], 'end': idx[14]},
        ])
        return plot_pytrendy(plot_df, 'noisy_edgecase_3', results.segments,
                             index_type='float', suppress_show=True)


# =============================================================================
# plot_pytrendy: noise segment neighbour branches
# =============================================================================

class TestPlotNoiseNeighbour:
    """Neighbour-driven adjustment branches in plot_pytrendy."""

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_noise_neighbour_adjacent.png',
                                   style='default')
    def test_noise_next_neighbouring_adjustment(self):
        """String index: adjacent same-direction segments set the vertical divider line.

        Detection alternates segment direction, so two adjacent Noise segments
        cannot be sourced from ``detect_trends``. This hand-built list is the
        only coverage for the non-date ``line_date = seg['end']`` branch.
        """
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        segments = [
            {'start': str_idx[1], 'end': str_idx[10], 'direction': 'Noise',
             'change_rank': 1},
            {'start': str_idx[11], 'end': str_idx[20], 'direction': 'Noise',
             'change_rank': 2},
        ]

        return plot_pytrendy(plot_df, 'gradual', segments,
                             index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_prev_fill_integer.png',
                                   style='default')
    def test_trend_next_noise_integer(self):
        """Integer index: trend followed by adjacent noise exercises end adjustment."""
        data, date_col = _with_integer_index(_edgecase('noisy_edgecase_4'))
        results = pt.detect_trends(data, value_col='noisy_edgecase_4', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 80, 'end': 95},
            {'direction': 'Noise', 'start': 96, 'end': 98},
        ])
        return plot_pytrendy(data.set_index(date_col)[['noisy_edgecase_4']],
                             'noisy_edgecase_4', results.segments,
                             index_type='integer', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_string_index_basic.png',
                                   style='default')
    def test_prev_not_trend_string(self):
        """String index: Flat (not a trend) followed by adjacent gradual trend."""
        data, date_col = _with_string_index(_synthetic('gradual'))
        results = pt.detect_trends(data, value_col='gradual', date_col=date_col,
                                   plot=False, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 'S36', 'end': 'S39'},
            {'direction': 'Up', 'start': 'S40', 'end': 'S75'},
        ])
        return plot_pytrendy(data.set_index(date_col)[['gradual']], 'gradual',
                             results.segments, index_type='string', suppress_show=True)
