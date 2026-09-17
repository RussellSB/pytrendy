"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different trend types using pytest-mpl for image comparison,
plus a test that asserts the plt.show() behaviour.
"""

import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from conftest import build_internal_index, assert_segments_in_a_haystack
import pytrendy as pt
from pytrendy.io.plot_pytrendy import plot_pytrendy
from pytrendy.process_signals import process_signals
from pytrendy.post_processing.segments_get import get_segments
from pytrendy.post_processing.segments_analyse import analyse_segments
from pytrendy.post_processing.segments_refine.trend_classify import classify_trends
from pytrendy.post_processing.segments_refine.gradual_expand_contract import expand_contract_segments
from pytrendy.post_processing.segments_refine.abrupt_shaving import shave_abrupt_trends
from pytrendy.post_processing.segments_refine.artifact_cleanup import clean_artifacts
import matplotlib.dates as mdates
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
        """Test visualization of abrupt trends synthetic with no spikes (synth 1)."""
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
        """Test visualization of noisy edgecase 7."""
        edgecases_df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            edgecases_df,
            date_col='date',
            value_col='noisy_edgecase_7',
            plot=False
        )
        
        fig = self._prepare_and_plot(edgecases_df, 'noisy_edgecase_7', results.segments)
        return fig


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_weekly_spaced_no_gaps.png', style='default')
    def test_plot_weekly_spaced_no_gaps(self):
        """Regression: weekly-spaced dates must shade without boundary gaps.

        Non-daily point spacing previously left ~6-day white bands between
        segments because boundary displacement assumed a one-day step. Weekly
        majors are also spacing-matched to the data's own weekday (Sundays),
        rather than the default Tuesday-aligned ``WeekdayLocator``.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date')['gradual'].resample('W').last()
        weekly.index = weekly.index.strftime('%Y-%m-%d')
        dfw = weekly.reset_index()
        dfw.columns = ['date', 'gradual']

        results = pt.detect_trends(dfw, date_col='date', value_col='gradual', plot=False)
        fig = self._prepare_and_plot(dfw, 'gradual', results.segments)
        major_weekdays = {
            pd.Timestamp(mdates.num2date(t)).strftime('%a')
            for t in fig.axes[0].get_xticks()
        }
        assert major_weekdays == {'Sun'}
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_weekly_datetime64_ticks.png', style='default')
    def test_plot_weekly_datetime64_ticks(self):
        """Weekly datetime64 index gets Sunday-aligned majors and no daily minor ticks.

        A real datetime64 column previously skipped the locator block entirely,
        leaving matplotlib's AutoDateLocator to pick a handful of auto ticks. Its
        boundaries also took the integer/float displacement branch (a hard-coded
        one-day step), so weekly points left ~6-day white bands between shaded
        segments; both are regression-guarded by this baseline.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date')['gradual'].resample('W').last().reset_index()
        weekly.columns = ['date', 'gradual']

        results = pt.detect_trends(weekly, date_col='date', value_col='gradual', plot=False)
        plot_df = weekly.set_index('date')[['gradual']]
        fig = plot_pytrendy(df=plot_df, value_col='gradual', segments_enhanced=results.segments,
                            index_type='datetime64', suppress_show=True)
        assert len(fig.axes[0].xaxis.get_minorticklocs()) == 0
        major_weekdays = {
            pd.Timestamp(mdates.num2date(t)).strftime('%a')
            for t in fig.axes[0].get_xticks()
        }
        assert major_weekdays == {'Sun'}
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_fortnightly_pinned_ticks.png', style='default')
    def test_fortnightly_pinned_ticks(self):
        """Fortnightly dates get majors pinned to every observation.

        A ``WeekdayLocator(interval=2)`` anchors its interval grid to the Unix
        epoch, so its majors landed one week off the 14-day observations (0/38 on
        data points). Non-daily spacing now pins majors to the data's own index
        positions, so every major (and its 'major' gridline) sits on a sampled
        point.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        long = pd.concat([df] * 3, ignore_index=True)
        long['date'] = pd.date_range(long['date'].iloc[0], periods=len(long), freq='D')
        weekly = long.set_index('date')['gradual'].resample('W').last()
        fortnightly = pd.DataFrame({
            'date': pd.date_range('2025-01-05', periods=len(weekly) // 2, freq='14D'),
            'gradual': weekly.iloc[::2].values,
        })

        results = pt.detect_trends(fortnightly, date_col='date', value_col='gradual',
                                   plot=False, method_params={'abrupt_padding': 0})
        plot_df = fortnightly.set_index('date')[['gradual']]
        fig = plot_pytrendy(df=plot_df, value_col='gradual', segments_enhanced=results.segments,
                            index_type='datetime64', suppress_show=True)

        major = {
            pd.Timestamp(mdates.num2date(t)).tz_localize(None).normalize()
            for t in fig.axes[0].get_xticks()
        }
        assert major == set(pd.DatetimeIndex(fortnightly['date']).normalize())
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
    """Segment positioning at the edges of the index and gaps between segments.

    Detection fills every uncovered range with a Flat segment, so the plotted
    index is always fully covered and these first/last/gap placements cannot be
    produced end-to-end; the hand-built lists exercise them directly at the
    plotting layer.
    """

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_boundary_first_segment_string.png',
                                   style='default')
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

        return plot_pytrendy(plot_df, 'gradual', segments,
                             index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_boundary_last_segment_string.png',
                                   style='default')
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

        return plot_pytrendy(plot_df, 'gradual', segments,
                             index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_boundary_non_neighbouring_string.png',
                                   style='default')
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

        return plot_pytrendy(plot_df, 'gradual', segments,
                             index_type='string', suppress_show=True)


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

    def _date_results(self):
        """Detect trends on the gradual synthetic series for plotting."""
        df = pt.load_data('series_synthetic')
        return pt.detect_trends(df, value_col='gradual', date_col='date',
                                plot=False, method_params={'abrupt_padding': 0})

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_custom_params.png',
                                   style='default')
    def test_plot_with_custom_params(self):
        """Test plot_params path for date index type."""
        plot_df = self._date_plot_df()
        results = self._date_results()

        plot_params = {
            'figsize': (10, 3),
            'title': 'Custom Title',
            'xlabel': 'Custom X',
            'ylabel': 'Custom Y',
            'grid': {'visible': False},
        }

        return plot_pytrendy(plot_df, 'gradual', results.segments,
                             index_type='date',
                             suppress_show=True, plot_params=plot_params)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_custom_legend.png',
                                   style='default')
    def test_plot_with_custom_legend(self):
        """Test legend customisation path."""
        plot_df = self._date_plot_df()
        results = self._date_results()

        plot_params = {
            'legend_loc': 'lower right',
        }

        return plot_pytrendy(plot_df, 'gradual', results.segments,
                             index_type='date',
                             suppress_show=True, plot_params=plot_params)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_custom_colors.png',
                                   style='default')
    def test_plot_with_custom_colors(self):
        """Test custom colors path."""
        plot_df = self._date_plot_df()
        results = self._date_results()

        plot_params = {
            'colors': {'Up': 'lightgreen', 'Down': 'lightcoral'},
        }

        return plot_pytrendy(plot_df, 'gradual', results.segments,
                             index_type='date',
                             suppress_show=True, plot_params=plot_params)



# =============================================================================
# plot_pytrendy: displacement branches sourced from detect_trends
# =============================================================================

class TestPlotPrevFillBranches:
    """Prev-fill displacement branch, driven by detection.

    A Flat (non-trend) segment adjacent to a trend whose start displacement is
    invalid triggers the prev-fill body. The noise-heavy ``noisy_edgecase_4``
    series produces that topology; date, string, and integer lookups each take a
    different sub-branch of the displacement logic.
    """

    def _detect(self, index_kind: str):
        """Detect on noisy_edgecase_4 for the requested lookup index kind."""
        df = pd.read_csv(
            'tests/tests_crashes_edgecases/data/noisy_edgecases.csv'
        )[['noisy_edgecase_4']].copy()

        if index_kind == 'date':
            df.insert(0, 'idx', pd.date_range('2025-01-01', periods=len(df), freq='D'))
        elif index_kind == 'string':
            df.insert(0, 'idx', [f'S{i}' for i in range(len(df))])
        else:
            df.insert(0, 'idx', np.arange(len(df)))

        results = pt.detect_trends(df, value_col='noisy_edgecase_4', date_col='idx',
                                   plot=False, method_params={'abrupt_padding': 0})
        return df, results

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_prev_fill_date.png',
                                   style='default')
    def test_prev_fill_date(self):
        """Date lookup: prev-fill branch from detected segments."""
        df, results = self._detect('date')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': pd.Timestamp('2025-03-22'), 'end': pd.Timestamp('2025-04-06')},
            {'direction': 'Up', 'start': pd.Timestamp('2025-04-11'), 'end': pd.Timestamp('2025-05-05')},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_4']], 'noisy_edgecase_4',
                             results.segments, index_type=results.index_type, suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_prev_fill_string.png',
                                   style='default')
    def test_prev_fill_string(self):
        """String lookup: prev-fill branch from detected segments."""
        df, results = self._detect('string')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 'S80', 'end': 'S95'},
            {'direction': 'Up', 'start': 'S100', 'end': 'S124'},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_4']], 'noisy_edgecase_4',
                             results.segments, index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_prev_fill_integer.png',
                                   style='default')
    def test_prev_fill_integer(self):
        """Integer lookup: prev-fill branch from detected segments."""
        df, results = self._detect('integer')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 80, 'end': 95},
            {'direction': 'Up', 'start': 100, 'end': 124},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_4']], 'noisy_edgecase_4',
                             results.segments, index_type='integer', suppress_show=True)


# =============================================================================
# plot_pytrendy: next-noise displacement branch
# =============================================================================

class TestPlotNextNoiseBranches:
    """Next-noise displacement branch, driven by detection.

    A trend segment followed by an adjacent Noise segment whose end displacement
    is invalid triggers the next-noise body. ``noisy_edgecase_6`` produces a
    gradual Down run running straight into a Noise block; date, string, and
    integer lookups each take a different sub-branch.
    """

    def _detect(self, index_kind: str):
        """Detect on noisy_edgecase_6 for the requested lookup index kind."""
        df = pd.read_csv(
            'tests/tests_crashes_edgecases/data/noisy_edgecases.csv'
        )[['noisy_edgecase_6']].copy()

        if index_kind == 'date':
            df.insert(0, 'idx', pd.date_range('2025-01-01', periods=len(df), freq='D'))
        elif index_kind == 'string':
            df.insert(0, 'idx', [f'S{i}' for i in range(len(df))])
        else:
            df.insert(0, 'idx', np.arange(len(df)))

        results = pt.detect_trends(df, value_col='noisy_edgecase_6', date_col='idx',
                                   plot=False, method_params={'abrupt_padding': 0})
        return df, results

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_next_noise_date.png',
                                   style='default')
    def test_next_noise_date(self):
        """Date lookup: next-noise branch from detected segments."""
        df, results = self._detect('date')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': pd.Timestamp('2025-03-10'), 'end': pd.Timestamp('2025-03-25')},
            {'direction': 'Noise', 'start': pd.Timestamp('2025-03-26'), 'end': pd.Timestamp('2025-04-11')},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_6']], 'noisy_edgecase_6',
                             results.segments, index_type=results.index_type, suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_next_noise_string.png',
                                   style='default')
    def test_next_noise_string(self):
        """String lookup: next-noise branch from detected segments."""
        df, results = self._detect('string')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 'S68', 'end': 'S83'},
            {'direction': 'Noise', 'start': 'S84', 'end': 'S100'},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_6']], 'noisy_edgecase_6',
                             results.segments, index_type='string', suppress_show=True)

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                   filename='test_plot_next_noise_integer.png',
                                   style='default')
    def test_next_noise_integer(self):
        """Integer lookup: next-noise branch from detected segments."""
        df, results = self._detect('integer')
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 68, 'end': 83},
            {'direction': 'Noise', 'start': 84, 'end': 100},
        ])
        return plot_pytrendy(df.set_index('idx')[['noisy_edgecase_6']], 'noisy_edgecase_6',
                             results.segments, index_type='integer', suppress_show=True)


# =============================================================================
# plot_pytrendy: spacing-aware numeric (float) boundary displacement
# =============================================================================

class TestPlotFloatIndexSpacing:
    """Regression: non-contiguous numeric indexes must displace boundaries by
    the actual point spacing, not a hard-coded one step.

    A float index (e.g. ``linspace(0, 4.3, 181)``) previously paid a ``±1``
    step that was ~23% of the whole axis: Flat segments over-shaded from the
    axis start and adjacent segments left white bands between them.
    """

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_float_dense_no_gaps.png',
                                    style='default')
    def test_float_dense_no_gaps(self):
        """Dense float index: shaded segments tile edge-to-edge, no white bands."""
        df = pt.load_data('series_synthetic')
        df['float_idx'] = np.linspace(0, 4.3, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_idx',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_df = df.set_index('float_idx')[['gradual']]
        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='float', suppress_show=True)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_float_sparse_no_gaps.png',
                                    style='default')
    def test_float_sparse_no_gaps(self):
        """Sparse float index (gap 0.2): segments tile edge-to-edge, no white bands."""
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date')['gradual'].resample('W').last()
        sparse = pd.DataFrame({'float_idx': np.arange(len(weekly)) * 0.2,
                               'gradual': weekly.values})
        results = pt.detect_trends(sparse, value_col='gradual', date_col='float_idx',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_df = sparse.set_index('float_idx')[['gradual']]
        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='float', suppress_show=True)
        return fig


# =============================================================================
# plot_pytrendy: _adjacent_to guard branches
# =============================================================================

class TestAdjacentToGuards:
    """Direct-call tests for the ``_adjacent_to`` None-guards.

    These use hand-crafted segment lists to hit guards the integration pipeline
    cannot produce: a boundary absent from the plotted index (``KeyError``) and a
    duplicated index value (non-int ``get_loc``). They are the honest coverage
    for ``plot_pytrendy`` lines 38-41 and stay direct by design (#284).
    """

    @pytest.mark.plot
    def test_date_boundary_absent_from_index(self):
        """Lines 38-39: ``_adjacent_to`` returns None when the boundary is absent.

        ``index.get_loc`` raises ``KeyError`` for a boundary past the plotted
        index (integration never produces this), so the except-guard returns
        None and the neighbour check falls back; the mask/xlim then clip the
        overhanging end.
        """
        dates = pd.date_range('2025-01-01', periods=40, freq='D')
        plot_df = pd.DataFrame({'gradual': np.arange(40, dtype=float)}, index=dates)
        segments = [
            {'start': dates[1], 'end': dates[-1] + pd.Timedelta(days=1),  # absent from index
             'direction': 'Down', 'trend_class': 'gradual', 'change_rank': 1},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='date', suppress_show=True)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.plot
    def test_duplicated_index_boundary_returns_none(self):
        """Lines 40-41: ``_adjacent_to`` returns None for a duplicated index value.

        On a non-unique index ``index.get_loc(value)`` returns a slice rather
        than an int, which the guard treats as "no well-defined adjacent point".
        Duplicate dates are not validated upstream (``prepare_index`` never
        checks uniqueness), so a duplicated boundary is this guard's real
        trigger.
        """
        dates = pd.date_range('2025-01-01', periods=40, freq='D')
        dup = dates[20]
        index = dates.insert(20, dup)  # dup now appears twice
        plot_df = pd.DataFrame(
            {'gradual': np.arange(len(index), dtype=float)}, index=index)
        segments = [
            {'start': dates[10], 'end': dup, 'direction': 'Flat', 'change_rank': 1},
        ]

        fig = plot_pytrendy(plot_df, 'gradual', segments,
                            index_type='date', suppress_show=True)
        assert fig is not None
        plt.close(fig)
