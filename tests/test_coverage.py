"""
Tests targeting uncovered lines for 100% test coverage.

Covers:
- plot_pytrendy: string, integer, and float index type branches
- plot_pytrendy: string prev fill (lines 172-178, 182) and next noise fill (lines 210-216)
- detect_trends: integer date_col, NotImplementedError, plot=True
- results_pytrendy: print_summary with non-date index types
- abrupt_shaving: out-of-range guard (line 93)
- artifact_cleanup: empty segments fallback, trend-after-flat overlap (line 115)
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch

import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
from pytrendy.io.results_pytrendy import PyTrendyResults
from conftest import assert_segments_in_a_haystack


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
# plot_pytrendy: edge cases (first segment at boundary, non-neighbouring)
# =============================================================================

class TestPlotEdgeCases:
    """Edge cases: first segment at index start, non-neighbouring segments."""

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
# detect_trends: integer date_col (line 40), NotImplementedError (line 44)
# =============================================================================

class TestDetectTrendsCoverage:
    """Test detect_trends uncovered paths."""

    def test_integer_date_col(self):
        """Line 40: integer dtype in date_col triggers 'integer' index type."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'

    def test_not_implemented_dtype(self):
        """Line 44: unimplemented dtype raises NotImplementedError."""
        df = pt.load_data('series_synthetic')
        df['bool_col'] = True
        with pytest.raises(NotImplementedError, match="unimplemented dtype"):
            pt.detect_trends(df, value_col='gradual', date_col='bool_col',
                             plot=False)

    def test_plot_true_integer_index(self):
        """Lines 165-167: plot=True path with integer index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_float_index(self):
        """Lines 165-167: plot=True path with float index."""
        df = pt.load_data('series_synthetic')
        df['float_col'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_string_index(self):
        """Lines 165-167: plot=True path with string index."""
        df = pt.load_data('series_synthetic')
        df['str_col'] = [f'S{i}' for i in range(len(df))]
        results = pt.detect_trends(df, value_col='gradual', date_col='str_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_with_plot_params(self):
        """Lines 165-167: plot=True with plot_params passed through."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0},
                                   plot_params={'title': 'Test Plot'})
        assert results is not None
        plt.close('all')


# =============================================================================
# results_pytrendy: print_summary with non-date index types
# =============================================================================

class TestResultsPrintSummaryCoverage:
    """Test print_summary with integer/string index types for lines 112, 114."""

    def test_print_summary_integer_index(self):
        """Line 112: print_summary with integer index_type uses 'indexes' descriptor."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        # Should not raise
        results.print_summary()

    def test_print_summary_string_index(self):
        """Line 114: print_summary with string index_type uses 'labels' descriptor."""
        df = pt.load_data('series_synthetic')
        df['str_col'] = [f'S{i}' for i in range(len(df))]
        results = pt.detect_trends(df, value_col='gradual', date_col='str_col',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'string'
        # Should not raise
        results.print_summary()

    def test_print_summary_float_index(self):
        """print_summary with float index_type uses 'indexes' descriptor."""
        df = pt.load_data('series_synthetic')
        df['float_col'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_col',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'float'
        # Should not raise
        results.print_summary()


# =============================================================================
# artifact_cleanup: empty segments fallback (lines 338-339)
# =============================================================================

class TestArtifactCleanupCoverage:
    """Test artifact_cleanup edge cases."""

    def test_fill_flats_empty_segments(self):
        """Lines 338-339: fill_in_flats with empty segment list covers full range."""
        from pytrendy.post_processing.segments_refine.artifact_cleanup import fill_in_flats
        df = pt.load_data('series_synthetic')
        df_int = df.set_index(np.arange(len(df)))[['gradual']]
        # Empty segments list
        result = fill_in_flats(df_int, [])
        assert len(result) == 1
        assert result[0]['direction'] == 'Flat'
        assert result[0]['start'] == df_int.index.min()
        assert result[0]['end'] == df_int.index.max()


# =============================================================================
# abrupt_shaving: out-of-range guard (line 93)
# =============================================================================

class TestAbruptShavingCoverage:
    """Test abrupt_shaving edge cases."""

    def test_abrupt_shaving_leading_subsegment(self):
        """Line 93: leading subsegment triggers out-of-range guard."""
        from pytrendy.post_processing.segments_refine.abrupt_shaving import shave_abrupt_trends
        from pytrendy.process_signals import process_signals
        from pytrendy.post_processing.segments_get import get_segments
        from pytrendy.post_processing.segments_refine.trend_classify import classify_trends

        df = pt.load_data('series_synthetic')
        df_int = df.set_index(np.arange(len(df)))[['abrupt']]
        method_params = {'abrupt_padding': 0, 'avoid_noise': True}

        df_processed = process_signals(df_int, 'abrupt', method_params)
        segments = get_segments(df_processed)
        segments = classify_trends(df_processed, 'abrupt', segments)

        # Run shave - the guard should be exercised if any segment starts at index[0]
        result = shave_abrupt_trends(df_processed, 'abrupt', segments, method_params)
        assert isinstance(result, list)


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


# =============================================================================
# plot_pytrendy: string index prev fill (lines 172-178, 182)
# =============================================================================

class TestPlotPrevFill:
    """Exercise the prev fill branch in plot_pytrendy when start displacement is invalid."""

    def test_string_prev_fill(self):
        """Lines 172-176: string index triggers prev fill branch."""
        df = pd.DataFrame(
            {'date': [f'S{i}' for i in range(40)],
             'value': [90 + i for i in range(10)] + [100] * 10 + [80 - i for i in range(5)] + [60 + i for i in range(15)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'string'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 'S0', 'end': 'S18'},
        ])

    def test_integer_prev_fill(self):
        """Lines 177-178: integer index triggers prev fill branch."""
        df = pd.DataFrame(
            {'value': [90 + i for i in range(10)] + [100] * 10 + [80 - i for i in range(5)] + [60 + i for i in range(15)]})
        results = pt.detect_trends(df, value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 0, 'end': 18},
        ])


class TestPlotNextNoiseFill:
    """Exercise the next-noise fill branch in plot_pytrendy when end displacement is invalid."""

    def test_string_next_noise_fill(self):
        """Lines 212-214: string index triggers next noise fill branch."""
        df = pd.DataFrame(
            {'date': [f'S{i}' for i in range(40)],
             'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'string'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 'S1', 'end': 'S17'},
        ])

    def test_integer_next_noise_fill(self):
        """Lines 215-216: integer index triggers next noise fill branch."""
        df = pd.DataFrame(
            {'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 1, 'end': 17},
        ])

    def test_date_next_noise_fill(self):
        """Line 211: date index triggers next noise fill branch."""
        df = pd.DataFrame(
            {'date': pd.date_range('2025-01-01', periods=40, freq='D'),
             'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'datetime64'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': pd.Timestamp('2025-01-02'), 'end': pd.Timestamp('2025-01-18')},
        ])


class TestDetectIndexTypeInteger:
    """Test _detect_index_type with explicit integer date_col."""

    def test_integer_date_col_returns_integer(self):
        """Line 40: passing an integer-typed column as date_col returns 'integer'."""
        df = pt.load_data('series_synthetic')
        df['int_col'] = np.arange(len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='int_col',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ])
