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

class TestPlotStringPrevFill:
    """Exercise the prev fill branch when start displacement is invalid."""

    def test_string_prev_not_trend_invalid_displacement(self):
        """Lines 172-178, 182: prev is Flat (not trend), neighbouring, start displacement invalid.

        Crafted segments: Flat followed by adjacent Up on string index where
        the Up start value is >= the value one position before it, making the
        left-displacement invalid.  Falls through to the prev fill branch.
        """
        values = list(range(181))
        values[89] = 100
        values[90] = 100
        values[91] = 99   # Up starts here — value[90] >= value[91], displacement invalid
        values[92] = 100
        values[93] = 101
        custom_df = pd.DataFrame({'gradual': values}, index=[f'S{i}' for i in range(181)])

        segments = [
            {'start': 'S80', 'end': 'S90', 'direction': 'Flat',
             'change_rank': 1},
            {'start': 'S91', 'end': 'S110', 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(custom_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_integer_prev_not_trend_invalid_displacement(self):
        """Lines 177-178: integer index, prev Flat neighbouring, start displacement invalid.

        Same pattern as string test but with integer index, exercising the else branch.
        """
        values = list(range(181))
        values[89] = 100
        values[90] = 100
        values[91] = 99   # Up starts here — value[90] >= value[91], displacement invalid
        values[92] = 100
        values[93] = 101
        custom_df = pd.DataFrame({'gradual': values}, index=range(181))

        segments = [
            {'start': 80, 'end': 90, 'direction': 'Flat',
             'change_rank': 1},
            {'start': 91, 'end': 110, 'direction': 'Up',
             'trend_class': 'gradual', 'change_rank': 2},
        ]

        fig = plot_pytrendy(custom_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# plot_pytrendy: string index next noise fill (lines 210-216)
# =============================================================================

class TestPlotStringNextNoiseFill:
    """Exercise the next-noise fill branch when end displacement is invalid."""

    def test_string_next_noise_invalid_displacement(self):
        """Lines 210-214: next is Noise (adjacent), end displacement invalid, string index.

        Crafted segments: Down followed by adjacent Noise on string index where
        the Down end value is < the value one position after it, making the
        right-displacement invalid (valid_down_end requires new_end < value).
        Falls through to the next noise fill branch.
        """
        values = [200 - i for i in range(20)] + [200 + i for i in range(161)]
        custom_df = pd.DataFrame({'gradual': values}, index=[f'S{i}' for i in range(181)])

        segments = [
            {'start': 'S0', 'end': 'S19', 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 1},
            {'start': 'S20', 'end': 'S40', 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(custom_df, 'gradual', segments,
                            index_type='string', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_integer_next_noise_invalid_displacement(self):
        """Lines 215-216: integer index, next Noise adjacent, end displacement invalid.

        Same pattern as string test but with integer index, exercising the else branch.
        """
        values = [200 - i for i in range(20)] + [200 + i for i in range(161)]
        custom_df = pd.DataFrame({'gradual': values}, index=range(181))

        segments = [
            {'start': 0, 'end': 19, 'direction': 'Down',
             'trend_class': 'gradual', 'change_rank': 1},
            {'start': 20, 'end': 40, 'direction': 'Noise',
             'change_rank': 2},
        ]

        fig = plot_pytrendy(custom_df, 'gradual', segments,
                            index_type='integer', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    def test_date_next_noise_invalid_displacement(self):
        """Line 211: date index, next Noise adjacent, end displacement invalid.

        Exercises the date branch of the next noise fill logic.
        """
        values = [200 - i for i in range(20)] + [200 + i for i in range(161)]
        dates = pd.date_range('2025-01-01', periods=181, freq='D')
        custom_df = pd.DataFrame({'gradual': values}, index=dates)

        segments = [
            {'start': dates[0].strftime('%Y-%m-%d'), 'end': dates[19].strftime('%Y-%m-%d'),
             'direction': 'Down', 'trend_class': 'gradual', 'change_rank': 1},
            {'start': dates[20].strftime('%Y-%m-%d'), 'end': dates[40].strftime('%Y-%m-%d'),
             'direction': 'Noise', 'change_rank': 2},
        ]

        fig = plot_pytrendy(custom_df, 'gradual', segments,
                            index_type='date', suppress_show=True)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# abrupt_shaving: out-of-range guard when new_start < df.index[0] (line 93)
# =============================================================================




# =============================================================================
# artifact_cleanup: trend after flat with similar size (line 115)
# =============================================================================

class TestArtifactCleanupTrendAfterFlat:
    """Test that has_partial_overlap_prev catches trend-after-flat overlap."""

    def test_trend_after_flat_similar_size(self):
        """Line 115: curr is trend, prev is flat, similar size, overlapping.

        Uses detect_trends on data that produces a flat region followed by
        a short overlapping trend of similar length, triggering the overlap
        cleanup at line 115.
        """
        # Data with a plateau followed by a small bump — produces Flat then short Up
        values = [100] * 15 + list(range(100, 110)) + [110] * 15
        df = pd.DataFrame({'value': values})

        results = pt.detect_trends(df, value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results is not None
        assert len(results.segments) > 0


# =============================================================================
# detect_trends: integer date_col path (line 40)
# =============================================================================

class TestDetectIndexTypeInteger:
    """Test _detect_index_type with explicit integer date_col."""

    def test_integer_date_col_returns_integer(self):
        """Line 40: passing an integer-typed column as date_col returns 'integer'."""
        df = pt.load_data('series_synthetic')
        df['int_col'] = np.arange(len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='int_col',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
