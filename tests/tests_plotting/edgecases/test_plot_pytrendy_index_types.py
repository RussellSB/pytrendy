"""
Tests for plot_pytrendy index-type branches.

These tests exercise the string, integer, and float index handling in
``plot_pytrendy``, as well as the noise/neighbour adjustment branches that
depend on the detected index type.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy


# =============================================================================
# plot_pytrendy: string index branches
# =============================================================================

class TestPlotStringIndex:
    """Exercise the index_type=='string' branches in plot_pytrendy."""

    def _make_string_df(self):
        """Build a string-indexed DataFrame."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'Step {i}' for i in range(len(df))]
        return df.set_index('str_idx')[['gradual']]

    def _str_segments(self, plot_df, specs):
        """Build segments with string start/end from position specs [(start_pos, end_pos, dir, ...)]."""
        str_idx = list(plot_df.index)
        segs = []
        for spec in specs:
            s = {'start': str_idx[spec[0]], 'end': str_idx[spec[1]], 'direction': spec[2]}
            if len(spec) > 3:
                s['trend_class'] = spec[3]
            if len(spec) > 4:
                s['change_rank'] = spec[4]
            segs.append(s)
        return segs

    def test_string_index_basic(self):
        """String index: basic plot without crash."""
        plot_df = self._make_string_df()
        # Use actual detect_trends results remapped to string
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        # Remap integer segments to string index
        str_idx = list(plot_df.index)
        for seg in results.segments:
            seg['start'] = str_idx[seg['start']]
            seg['end'] = str_idx[seg['end']]
        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_string_index_with_adjacent_segments(self):
        """String index: exercise prev/next neighbouring logic."""
        plot_df = self._make_string_df()
        segments = self._str_segments(plot_df, [
            (1, 10, 'Up', 'gradual', 1),
            (11, 20, 'Down', 'gradual', 2),
            (21, 30, 'Up', 'gradual', 3),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_string_index_abrupt_segment(self):
        """String index: abrupt segment triggers start/end adjustment branches."""
        plot_df = self._make_string_df()
        segments = self._str_segments(plot_df, [
            (1, 10, 'Up', 'abrupt', 1),
            (11, 20, 'Down', 'abrupt', 2),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_string_index_noise_segment(self):
        """String index: noise segment triggers noise branches."""
        plot_df = self._make_string_df()
        segments = self._str_segments(plot_df, [
            (1, 10, 'Noise', None, 1),
            (11, 20, 'Up', 'gradual', 2),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_string_index_flat_segment(self):
        """String index: flat segment (no trend_class, not neighbouring)."""
        plot_df = self._make_string_df()
        segments = self._str_segments(plot_df, [
            (1, 10, 'Flat', None, 1),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_pytrendy: integer index branches
# =============================================================================

class TestPlotIntegerIndex:
    """Exercise the integer index branches in plot_pytrendy."""

    def _make_int_df(self):
        """Build an integer-indexed DataFrame."""
        df = pt.load_data('series_synthetic')
        return df.set_index(df.index)[['gradual']]

    def _int_segments(self, plot_df, specs):
        """Build segments with integer start/end from position specs."""
        idx = list(plot_df.index)
        segs = []
        for spec in specs:
            s = {'start': idx[spec[0]], 'end': idx[spec[1]], 'direction': spec[2]}
            if len(spec) > 3:
                s['trend_class'] = spec[3]
            if len(spec) > 4:
                s['change_rank'] = spec[4]
            segs.append(s)
        return segs

    def test_integer_index_basic(self):
        """Integer index: basic plot without crash."""
        plot_df = self._make_int_df()
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        # Remap segments to actual index values
        idx = list(plot_df.index)
        for seg in results.segments:
            seg['start'] = idx[seg['start']]
            seg['end'] = idx[seg['end']]
        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_integer_index_with_adjacent_segments(self):
        """Integer index: exercise prev/next neighbouring logic."""
        plot_df = self._make_int_df()
        segments = self._int_segments(plot_df, [
            (1, 10, 'Up', 'gradual', 1),
            (11, 20, 'Down', 'gradual', 2),
            (21, 30, 'Up', 'gradual', 3),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_integer_index_abrupt_segment(self):
        """Integer index: abrupt segment triggers end adjustment branches."""
        plot_df = self._make_int_df()
        segments = self._int_segments(plot_df, [
            (1, 10, 'Up', 'abrupt', 1),
            (11, 20, 'Down', 'abrupt', 2),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_pytrendy: float index branches
# =============================================================================

class TestPlotFloatIndex:
    """Exercise the float index branches in plot_pytrendy."""

    def _make_float_df(self):
        """Build a float-indexed DataFrame."""
        df = pt.load_data('series_synthetic')
        df['float_idx'] = np.linspace(0, 1, len(df))
        return df.set_index('float_idx')[['gradual']]

    def _float_segments(self, plot_df, specs):
        """Build segments with float start/end from position specs."""
        idx = list(plot_df.index)
        segs = []
        for spec in specs:
            s = {'start': idx[spec[0]], 'end': idx[spec[1]], 'direction': spec[2]}
            if len(spec) > 3:
                s['trend_class'] = spec[3]
            if len(spec) > 4:
                s['change_rank'] = spec[4]
            segs.append(s)
        return segs

    def test_float_index_basic(self):
        """Float index: basic plot without crash."""
        plot_df = self._make_float_df()
        df = pt.load_data('series_synthetic')
        df['float_idx'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_idx',
                                   plot=False, method_params={'abrupt_padding': 0})
        # Segments already have float start/end from detect_trends
        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='float', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_float_index_with_adjacent_segments(self):
        """Float index: exercise prev/next neighbouring logic."""
        plot_df = self._make_float_df()
        segments = self._float_segments(plot_df, [
            (1, 10, 'Up', 'gradual', 1),
            (11, 20, 'Down', 'gradual', 2),
            (21, 30, 'Up', 'gradual', 3),
        ])

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='float', suppress_show=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_pytrendy: noise segment with next neighbour (end adjustment branch)
# =============================================================================

class TestPlotNoiseNeighbour:
    """Test the noise + next_neighbouring branch in plot_pytrendy."""

    def test_noise_next_neighbouring_adjustment(self):
        """Noise segment followed by adjacent noise triggers start adjustment on next."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        # Noise followed by adjacent noise
        segments = [
            {'start': str_idx[1], 'end': str_idx[10], 'direction': 'Noise',
             'change_rank': 1},
            {'start': str_idx[11], 'end': str_idx[20], 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_abrupt_next_noise_integer(self):
        """Abrupt segment followed by adjacent noise triggers end adjustment on next (integer)."""
        df = pt.load_data('series_synthetic')
        plot_df = df.set_index(df.index)[['gradual']]
        idx = list(plot_df.index)

        segments = [
            {'start': idx[1], 'end': idx[10], 'direction': 'Up',
             'trend_class': 'abrupt', 'change_rank': 1},
            {'start': idx[11], 'end': idx[20], 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_prev_not_trend_string(self):
        """String index: prev is not trend, triggers prev fill adjustment."""
        df = pt.load_data('series_synthetic')
        df['str_idx'] = [f'S{i}' for i in range(len(df))]
        plot_df = df.set_index('str_idx')[['gradual']]
        str_idx = list(plot_df.index)

        # Flat (not trend) followed by adjacent gradual
        segments = [
            {'start': str_idx[1], 'end': str_idx[10], 'direction': 'Flat',
             'change_rank': 1},
            {'start': str_idx[11], 'end': str_idx[20],
             'direction': 'Up', 'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)
