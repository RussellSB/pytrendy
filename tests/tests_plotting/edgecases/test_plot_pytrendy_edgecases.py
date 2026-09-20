"""
Tests for plot visualization functionality.

These tests verify that the plot_pytrendy function generates consistent
visualizations for different types of trends using pytest-mpl for image comparison.
One extra test included to assess plt.show() behaviour only... for test coverage
"""

import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from conftest import build_internal_index
import pytrendy as pt
from pytrendy.io import prep_signal_params
from pytrendy.io.plot_pytrendy import (plot_pytrendy, _pinned_tick_positions,
                                       _date_tick_spec, _sampling_locator,
                                       _minor_locator, _format_for, _divides,
                                       _MAX_MINOR_TICKS)
from pytrendy.process_signals import process_signals
from pytrendy.post_processing.segments_get import get_segments
from pytrendy.post_processing.segments_analyse import analyse_segments
from pytrendy.post_processing.segments_refine.trend_classify import classify_trends
from pytrendy.post_processing.segments_refine.gradual_expand_contract import expand_contract_segments
from pytrendy.post_processing.segments_refine.abrupt_shaving import shave_abrupt_trends
from pytrendy.post_processing.segments_refine.artifact_cleanup import clean_artifacts
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator


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
        signal_params = prep_signal_params.prep_signal_params()

        df = process_signals(df, value_col, method_params, signal_params)
        segments = get_segments(df, signal_params)

        # ------------------ refine_segments()
        # unwrapped-equivalent to disable grouping at a lower level  
        segments_refined = deepcopy(segments)
        segments_refined = classify_trends(df, value_col, segments_refined)
        # No grouping code in between these steps
        segments_refined = expand_contract_segments(df, value_col, segments_refined, method_params) # for gradual
        segments_refined = shave_abrupt_trends(df, value_col, segments_refined, method_params) # for abrupt
        segments_refined = clean_artifacts(df, value_col, segments_refined, method_params, signal_params) # cleans overlaps etc from expand/contract
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


    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_weekly_spaced_no_gaps.png', style='default')
    def test_plot_weekly_spaced_no_gaps(self):
        """Regression: weekly-spaced dates must shade without boundary gaps.

        Non-daily point spacing previously left ~6-day white bands between
        segments because boundary displacement assumed a one-day step. Weekly
        dates now get month majors with weekly positional minors; the baseline
        guards the gap-free shading.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date')['gradual'].resample('W').last()
        weekly.index = weekly.index.strftime('%Y-%m-%d')
        dfw = weekly.reset_index()
        dfw.columns = ['date', 'gradual']

        results = pt.detect_trends(dfw, date_col='date', value_col='gradual', plot=False)
        fig = self._prepare_and_plot(dfw, 'gradual', results.segments)
        assert isinstance(fig.axes[0].xaxis.get_major_locator(), FixedLocator)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_weekly_datetime64_ticks.png', style='default')
    def test_plot_weekly_datetime64_ticks(self):
        """Weekly datetime64 index gets month majors and weekly positional minors.

        A real datetime64 column previously skipped the locator block entirely,
        leaving matplotlib's AutoDateLocator to pick a handful of auto ticks. Its
        boundaries also took the integer/float displacement branch (a hard-coded
        one-day step), so weekly points left ~6-day white bands between shaded
        segments; both are regression-guarded by this baseline. The minors are
        the data's own weekly observations, so the ruler sits on sampled points.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        weekly = df.set_index('date')['gradual'].resample('W').last().reset_index()
        weekly.columns = ['date', 'gradual']

        results = pt.detect_trends(weekly, date_col='date', value_col='gradual', plot=False)
        plot_df = weekly.set_index('date')[['gradual']]
        fig = plot_pytrendy(df=plot_df, value_col='gradual', segments_enhanced=results.segments,
                            index_type='datetime64', suppress_show=True)
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
        assert len(ax.xaxis.get_minorticklocs()) > 0
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_fortnightly_pinned_ticks.png', style='default')
    def test_fortnightly_month_majors_with_positional_minors(self):
        """Fortnightly dates get month majors and fortnightly positional minors.

        A ``WeekdayLocator(interval=2)`` anchors its interval grid to the Unix
        epoch, so its majors landed one week off the 14-day observations. The
        multi-day path now steps majors up to a calendar unit (month majors here)
        and emits the data's own spacing as positional minors, so the ruler sits
        on sampled points.
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

        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), FixedLocator)
        assert len(ax.xaxis.get_minorticklocs()) > 0
        return fig

    def _long_daily_series(self, periods=731):
        """Build a deterministic daily series of *periods* points by tiling synthetic data."""
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        long = pd.concat([df] * 6, ignore_index=True)
        long['date'] = pd.date_range(long['date'].iloc[0], periods=len(long), freq='D')
        return long.iloc[:periods].reset_index(drop=True)

    def _fortnightly_series(self, periods):
        """Build a fortnightly series of *periods* observations from tiled synthetic data."""
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        long = pd.concat([df] * 6, ignore_index=True)
        long['date'] = pd.date_range(long['date'].iloc[0], periods=len(long), freq='D')
        weekly = long.set_index('date')['gradual'].resample('W').last()
        return pd.DataFrame({
            'date': pd.date_range('2024-01-07', periods=periods, freq='14D'),
            'gradual': weekly.iloc[::2].values[:periods],
        })

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./', filename='test_plot_two_year_daily_tick_thinning.png', style='default')
    def test_plot_two_year_daily_tick_thinning(self):
        """A ~2-year daily series coarsens majors to months, with no minors.

        Weekly majors (and a daily minor ruler) previously produced a wall of
        ~104 labels plus ~626 minor ticks at 2 years. Span-adaptive density,
        keyed on ``span_steps`` (730 index steps here), keeps ~24 month majors;
        the minor ruler stops past a year, so the axis reads as a calendar again.
        """
        two_years = self._long_daily_series(731)  # ~2 years daily
        results = pt.detect_trends(two_years, date_col='date', value_col='gradual', plot=False)
        fig = self._prepare_and_plot(two_years, 'gradual', results.segments)

        ax = fig.axes[0]
        assert len(ax.get_xticks()) <= 30
        assert len(ax.xaxis.get_minorticklocs()) == 0
        return fig

    def test_daily_short_span_keeps_weekly_majors(self):
        """180-step daily data keeps the original weekly majors (baseline guard)."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=False)
        fig = self._prepare_and_plot(df, 'gradual', results.segments)

        locator = fig.axes[0].xaxis.get_major_locator()
        assert isinstance(locator, mdates.WeekdayLocator)
        assert locator._get_interval() == 1

    def test_daily_medium_span_biweekly_majors(self):
        """~1-year daily data (365 steps) coarsens majors to biweekly (interval=2)."""
        one_year = self._long_daily_series(366)
        results = pt.detect_trends(one_year, date_col='date', value_col='gradual', plot=False)
        fig = self._prepare_and_plot(one_year, 'gradual', results.segments)

        locator = fig.axes[0].xaxis.get_major_locator()
        assert isinstance(locator, mdates.WeekdayLocator)
        assert locator._get_interval() == 2

    def test_long_fortnightly_gets_month_majors_and_positional_minors(self):
        """52-point fortnightly data gets month majors and every point as minors.

        Fortnightly spacing is non-daily, so majors step up to a calendar unit
        (``MonthLocator(1)`` over ~2 years) and the data's own points become the
        positional minor ruler (52, under the 900 cap).
        """
        fortnightly = self._fortnightly_series(52)
        major, minor, pinned, _ = _date_tick_spec(
            pd.DatetimeIndex(fortnightly['date']), len(fortnightly) - 1)
        assert major is None and len(pinned) == 24
        assert len(minor) == len(fortnightly)

    @staticmethod
    def _pinned_ticks(index, index_type):
        """Plot *index* with no segments and return ``(major ticks, locator)``."""
        df = pd.DataFrame({'value': np.arange(len(index), dtype=float)}, index=index)
        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=[],
                            index_type=index_type, suppress_show=True)
        ax = fig.axes[0]
        if index_type in ('date', 'datetime64'):
            ticks = [
                pd.Timestamp(mdates.num2date(t)).tz_localize(None).normalize()
                for t in ax.get_xticks()
            ]
        else:
            ticks = list(ax.get_xticks())
        return ticks, ax.xaxis.get_major_locator()

    def test_two_year_monthly_gets_month_majors_and_minors(self):
        """A ~2-year monthly series keeps month majors with monthly minors.

        Below the ~4-year switch, monthly spacing labels every other month and
        the 24 month-ends remain the positional minor ruler.
        """
        monthly = pd.date_range('2020-01-31', periods=24, freq='ME')
        _, locator = self._pinned_ticks(monthly, 'datetime64')
        assert isinstance(locator, FixedLocator)
        _, minor, pinned, _ = _date_tick_spec(monthly, len(monthly) - 1)
        assert len(pinned) == len(monthly) // 2 and len(minor) == len(monthly)

    def test_three_year_monthly_labels_every_other_month(self):
        """A ~3-year monthly series has two-month majors and monthly minors."""
        monthly = pd.date_range('2020-01-31', periods=36, freq='ME')
        major, minor, pinned, _ = _date_tick_spec(monthly, len(monthly) - 1)
        assert major is None and len(pinned) == 18
        assert len(minor) == len(monthly)

    def test_four_year_monthly_stays_two_monthly_at_boundary(self):
        """A 48-point monthly span remains below the four-year threshold."""
        monthly = pd.date_range('2020-01-31', periods=48, freq='ME')
        major, minor, pinned, _ = _date_tick_spec(monthly, len(monthly) - 1)
        assert major is None and len(pinned) == 24
        assert len(minor) == len(monthly)

    def test_five_year_monthly_gets_year_majors_and_minors(self):
        """A ~5-year monthly series steps up to year majors with monthly minors."""
        monthly = pd.date_range('2020-01-31', periods=60, freq='ME')
        major, minor, pinned, _ = _date_tick_spec(monthly, len(monthly) - 1)
        assert major is None and len(pinned) == 5
        assert len(minor) == len(monthly)

    def test_yearly_series_gets_year_majors_and_minors(self):
        """A yearly series uses year majors with the yearly observations as minors."""
        yearly = pd.date_range('2015-01-01', periods=8, freq='YS')
        major, minor, pinned, _ = _date_tick_spec(yearly, len(yearly) - 1)
        assert major is None and len(pinned) == len(yearly)
        assert len(minor) == len(yearly)

    def test_weekly_one_year_gets_month_majors_and_weekly_minors(self):
        """52 weekly points get month majors, not a finer daily/weekly locator."""
        weekly = pd.date_range('2020-01-05', periods=52, freq='7D')
        major, minor, pinned, _ = _date_tick_spec(weekly, len(weekly) - 1)
        assert major is None and len(pinned) == 12
        assert len(minor) == len(weekly)

    def test_long_weekly_minors_thin_but_do_not_drop(self):
        """10-year weekly keeps all 522 minors; 30-year thins to the 900 cap."""
        ten = pd.date_range('2015-01-04', periods=522, freq='7D')
        major, minor, pinned, _ = _date_tick_spec(ten, len(ten) - 1)
        assert major is None and len(pinned) <= 40
        assert len(minor) == 522  # under the 900 cap -> every observation
        thirty = pd.date_range('1995-01-01', periods=1566, freq='7D')
        major, minor, pinned, _ = _date_tick_spec(thirty, len(thirty) - 1)
        assert major is None and len(pinned) <= 40
        assert len(minor) == 783  # ceil(1566 / 900) = 2 -> every 2nd observation

    def test_fifty_year_yearly_scales_year_interval(self):
        """50 yearly points exceed the 40-label cap, so the year interval scales."""
        yearly = pd.date_range('1975-01-01', periods=50, freq='YS')
        major, minor, pinned, _ = _date_tick_spec(yearly, len(yearly) - 1)
        assert major is None and len(pinned) == 25
        assert len(minor) == len(yearly)

    def test_integer_index_thins_past_40(self):
        """The positional pin/thin rule is agnostic to index granularity.

        A numerical index is just another positional index: up to 40 points
        every value is a tick, past it every ``ceil(n/40)``th value is. Covered
        on the helper directly, because the numeric plot baselines
        (``TestPlotFloatIndexSpacing``) intentionally keep matplotlib's own
        ticks; this guards the positional fallback the non-daily date path uses.
        """
        index = pd.Index(np.arange(181))
        step = int(np.ceil(len(index) / 40))
        assert step == 5
        assert list(_pinned_tick_positions(index)) == list(index[::step])
        assert list(_pinned_tick_positions(pd.Index(np.arange(30)))) == list(range(30))

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

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_inverted_default_colors.png',
                                    style='default')
    def test_plot_inverted_default_colors(self):
        """Issue #194: inverted-defaults palette exercises every colour format.

        Up is a named colour, Down short hex, Flat full hex and Noise a
        ``tab:*`` colour, so one figure covers all four matplotlib colour
        formats that previously crashed the annotation colour derivation.
        """
        plot_df = self._date_plot_df()
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', date_col='date',
                                   plot=False, method_params={'abrupt_padding': 0})

        plot_params = {
            'colors': {
                'Up': 'lightcoral',    # named
                'Down': '#9e9',        # short hex
                'Flat': '#D8BFD8',     # full hex
                'Noise': 'tab:blue',   # non-'light' named
            },
        }

        fig = plot_pytrendy(plot_df, 'gradual', results.segments,
                            index_type='date',
                            suppress_show=True, plot_params=plot_params)
        return fig

    def test_annotation_color_formats(self):
        """Issue #194: colour derivation preserves defaults and accepts every format."""
        from pytrendy.io.plot_pytrendy import _annotation_color

        # Default 'light*' names keep the exact legacy result (baselines unchanged)
        assert _annotation_color('lightgreen') == 'green'
        assert _annotation_color('lightgray') == 'gray'

        # Every other valid matplotlib colour yields a valid, darker RGBA colour
        for color in ['#ff0000', '#F00', 'red', 'tab:blue',
                      (0.1, 0.2, 0.3), (0.1, 0.2, 0.3, 0.5)]:
            r, g, b, a = mcolors.to_rgba(_annotation_color(color))
            assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1

        # None defers to matplotlib's own default rather than raising
        assert _annotation_color(None) is None


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
    """Direct-call coverage for the ``_adjacent_to`` None-guards.

    TODO: these use hand-crafted segment lists to hit guards the integration
    pipeline cannot produce (boundaries absent from / duplicated in the plotted
    index); redo with realistic scenarios if unsorted/duplicate input ever
    becomes supported (#284).
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
        Duplicate dates are not validated upstream (``prep_index`` never
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



# =============================================================================
# plot_pytrendy: intraday tick granularity (cadence-anchored minors)
# =============================================================================

class TestIntradayTickGranularity:
    """Sub-daily cadences get true-granularity minors and span-widened majors.

    Minors mark the data's own sampling gap (30-minute bars -> every 30 minutes,
    hourly -> every hour), thinned under matplotlib's tick limit; majors widen
    with the span in calendar days and pick a time-bearing label format.
    """

    @staticmethod
    def _intraday_plot_df(periods, freq):
        """Deterministic sub-daily frame with the given cadence."""
        index = pd.date_range('2026-01-01', periods=periods, freq=freq)
        values = 50 + 10 * np.sin(np.arange(periods) / (periods / 6.0))
        return pd.DataFrame({'value': values}, index=index)

    def _plot(self, periods, freq):
        """Plot one frame with a single mid-span Up segment; return (fig, index)."""
        plot_df = self._intraday_plot_df(periods, freq)
        n = len(plot_df)
        segments = [{'start': plot_df.index[n // 4], 'end': plot_df.index[3 * n // 4],
                     'direction': 'Up', 'trend_class': 'gradual', 'change_rank': 1}]
        fig = plot_pytrendy(plot_df, 'value', segments,
                            index_type='date', suppress_show=True)
        return fig, plot_df.index

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_intraday_one_day_30min.png',
                                    style='default')
    def test_plot_intraday_one_day_30min(self):
        """1 day of 30-minute bars: 2-hour majors, 30-minute minors, date+time labels."""
        fig, _ = self._plot(48, '30min')
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), mdates.HourLocator)
        assert np.allclose(np.diff(ax.get_xticks()), 2 / 24)
        assert isinstance(ax.xaxis.get_minor_locator(), mdates.MinuteLocator)
        assert ax.xaxis.get_major_formatter().fmt == '%Y-%m-%d\n%H:%M'
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_intraday_three_day_30min.png',
                                    style='default')
    def test_plot_intraday_three_day_30min(self):
        """3 days of 30-minute bars: 6-hour majors, 30-minute minors, date+time labels."""
        fig, _ = self._plot(145, '30min')
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), mdates.HourLocator)
        assert np.allclose(np.diff(ax.get_xticks()), 6 / 24)
        assert isinstance(ax.xaxis.get_minor_locator(), mdates.MinuteLocator)
        assert ax.xaxis.get_major_formatter().fmt == '%Y-%m-%d\n%H:%M'
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_intraday_sixty_day_30min.png',
                                    style='default')
    def test_plot_intraday_sixty_day_30min(self):
        """60 days of 30-minute bars: weekly majors, thinned 2-hour minors.

        2880 projected 30-minute minors exceed the 900 legibility target, so the
        interval is coarsened x4 to 2 hours (~720 ticks) and no MAXTICKS warning
        is raised. This baseline is unchanged from the previous commit.
        """
        fig, _ = self._plot(2881, '30min')
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), mdates.WeekdayLocator)
        assert isinstance(ax.xaxis.get_minor_locator(), mdates.HourLocator)
        assert len(ax.xaxis.get_minorticklocs()) <= _MAX_MINOR_TICKS
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_intraday_three_day_hourly.png',
                                    style='default')
    def test_plot_intraday_three_day_hourly(self):
        """3 days of hourly bars: 6-hour majors, true hourly minors."""
        fig, _ = self._plot(73, 'h')
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), mdates.HourLocator)
        assert np.allclose(np.diff(ax.get_xticks()), 6 / 24)
        assert isinstance(ax.xaxis.get_minor_locator(), mdates.HourLocator)
        return fig

    @pytest.mark.plot
    @pytest.mark.mpl_image_compare(baseline_dir='./',
                                    filename='test_plot_intraday_three_day_45min.png',
                                    style='default')
    def test_plot_intraday_three_day_45min(self):
        """3 days of 45-minute bars: 6-hour majors, positional 45-minute minors.

        A 45-minute gap has no minute/hour locator, so the true-granularity
        ruler is emitted as explicit observation positions while the majors
        still land on the smallest expressible hour multiple (6 h).
        """
        fig, _ = self._plot(97, '45min')
        ax = fig.axes[0]
        assert isinstance(ax.xaxis.get_major_locator(), mdates.HourLocator)
        assert np.allclose(np.diff(ax.get_xticks()), 6 / 24)
        assert len(ax.xaxis.get_minorticklocs()) > 0
        assert ax.xaxis.get_major_formatter().fmt == '%Y-%m-%d\n%H:%M'
        return fig


class TestIntradayTickSpec:
    """Direct coverage for the cadence ladder and its fiddly fallbacks."""

    @staticmethod
    def _spec(periods, freq):
        index = pd.date_range('2026-01-01', periods=periods, freq=freq)
        return _date_tick_spec(index, len(index) - 1)

    def test_intraday_ladder_widens_with_span(self):
        """30-minute bars step through hours -> 6 hours -> days -> weeks -> months."""
        assert isinstance(self._spec(48, '30min')[0], mdates.HourLocator)
        assert isinstance(self._spec(145, '30min')[0], mdates.HourLocator)
        assert isinstance(self._spec(337, '30min')[0], mdates.DayLocator)
        assert isinstance(self._spec(9601, '30min')[0], mdates.WeekdayLocator)
        assert isinstance(self._spec(14401, '30min')[0], mdates.WeekdayLocator)
        assert isinstance(self._spec(24001, '30min')[0], mdates.MonthLocator)

    def test_cadences_without_an_expressible_unit_fall_back_to_pinned(self):
        """A 5- or 7-hour gap divides no hour/day unit, so majors pin to observations."""
        for periods, freq in [(5, '5h'), (7, '7h')]:
            major, _, pinned, _ = self._spec(periods, freq)
            assert major is None and pinned is not None

    def test_per_rung_search_finds_smallest_expressible_interval(self):
        """45-/90-minute gaps get 3-hour majors within a day, 6-hour within three."""
        assert isinstance(self._spec(33, '45min')[0], mdates.HourLocator)   # ~1 day
        assert isinstance(self._spec(97, '45min')[0], mdates.HourLocator)   # 3 days
        assert isinstance(self._spec(17, '90min')[0], mdates.HourLocator)
        assert isinstance(self._spec(25, 'h')[0], mdates.HourLocator)       # exactly 1 h

    def test_major_interval_is_the_smallest_expressible(self):
        """45-minute cadence: 3-hour majors under a day, 6-hour under three days."""
        for periods, expected in [(33, 3 / 24), (97, 6 / 24)]:
            index = pd.date_range('2026-01-01', periods=periods, freq='45min')
            major, _, _, _ = _date_tick_spec(index, len(index) - 1)
            ticks = major.tick_values(index[0], index[-1])
            assert np.allclose(np.diff(ticks), expected)

    def test_sampling_locator_rejects_unexpressible_gaps(self):
        """No locator can mark 45-/90-minute, 7-hour or sub-minute gaps."""
        assert _sampling_locator(45 / 1440) is None
        assert _sampling_locator(90 / 1440) is None
        assert _sampling_locator(7 / 24) is None
        assert _sampling_locator(90 / 86400) is None  # 90 seconds
        assert _sampling_locator(0) is None

    def test_divides_rejects_gaps_larger_than_the_unit(self):
        """A gap wider than the unit cannot divide it (and no unit is <= 0)."""
        assert _divides(2, 1) is False
        assert _divides(0, 1) is False
        assert _divides(0.5, 1) is True

    def test_minor_thinning_lands_on_expressible_interval(self):
        """Coarsening snaps up to a whole-hour/day multiple, never dropping the ruler."""
        index = pd.date_range('2026-01-01', periods=2881, freq='30min')
        assert isinstance(_minor_locator(1 / 48, 2880, index), mdates.HourLocator)
        assert isinstance(_minor_locator(1 / 48, 9600, index), mdates.HourLocator)
        assert isinstance(_minor_locator(1 / 48, 48000, index), mdates.DayLocator)
        # ceil(2000 / 900) = 3 -> 1.5 h (unexpressible), so it steps up to 2 h.
        assert isinstance(_minor_locator(1 / 48, 2000, index), mdates.HourLocator)

    def test_minor_target_is_900(self):
        """60 days of 30-minute bars thin x4 to 2-hour minors (~720 ticks)."""
        index = pd.date_range('2026-01-01', periods=2881, freq='30min')
        locator = _minor_locator(1 / 48, len(index) - 1, index)
        assert isinstance(locator, mdates.HourLocator)
        ticks = locator.tick_values(index[0], index[-1])
        assert np.allclose(np.diff(ticks), 2 / 24)
        assert len(ticks) <= _MAX_MINOR_TICKS

    def test_inexpressible_minors_are_positional(self):
        """45-minute bars emit observation positions, not a locator."""
        index = pd.date_range('2026-01-01', periods=97, freq='45min')
        _, minor, pinned, _ = _date_tick_spec(index, len(index) - 1)
        assert pinned is None
        assert isinstance(minor, pd.Index)
        assert len(minor) == len(index)  # under the 900 target -> every observation

    def test_minors_stop_past_a_year(self):
        """A >365-day intraday span returns to a calendar with no minor ruler."""
        index = pd.date_range('2026-01-01', periods=8785, freq='h')  # ~366 days
        _, minor, _, _ = _date_tick_spec(index, len(index) - 1)
        assert minor is None

    def test_format_keys_to_major_unit_not_span(self):
        """Hour-or-finer majors show date+time; day-or-coarser read as dates."""
        assert _format_for(mdates.HourLocator()) == '%Y-%m-%d\n%H:%M'
        assert _format_for(mdates.MinuteLocator()) == '%Y-%m-%d\n%H:%M'
        assert _format_for(mdates.DayLocator()) == '%Y-%m-%d'
        assert _format_for(mdates.WeekdayLocator()) == '%Y-%m-%d'
        assert _format_for(mdates.MonthLocator()) == '%Y-%m-%d'
        assert _format_for(None) == '%Y-%m-%d'
