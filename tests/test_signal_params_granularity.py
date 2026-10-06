"""
Granularity-aware smoothing-window regression tests.

``window_smooth`` is inferred from the index cadence when the caller does not
override it (see ``pytrendy.io.prep_signal_params``). These tests build synthetic
series at each supported cadence — a seasonal ripple riding on one clean linear
uptrend between flat bookends, plus small seeded Gaussian noise — and check that
the uptrend survives the cadence-appropriate smoothing window instead of being
fragmented by the ripple. The weekly override test pins the historical
fixed-window (15) behaviour, and the final test documents the default weekly (9)
behaviour.
"""

import numpy as np
import pandas as pd

import pytrendy as pt
from conftest import assert_segments_in_a_haystack, assert_segments_match


def _seasonal(freq, n, cycle_steps, up_start, up_end, height=40.0, base=100.0,
              ripple_frac=0.15, noise=0.5, seed=0, ripple=True):
    """Build a series with a seasonal ripple over a single linear uptrend.

    The trend is flat at ``base`` before ``up_start``, ramps linearly to
    ``base + height`` across ``[up_start, up_end]``, then stays flat. The ripple
    amplitude is a fraction of the trend range.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = np.full(n, base, dtype=float)
    span = up_end - up_start
    ramp = np.arange(up_start, up_end + 1)
    trend[up_start:up_end + 1] = base + height * (ramp - up_start) / span
    trend[up_end + 1:] = base + height

    if ripple:
        ripple_values = ripple_frac * height * np.sin(2 * np.pi * t / cycle_steps)
    else:
        ripple_values = 0.0

    value = trend + ripple_values + rng.normal(0, noise, n)
    dates = pd.date_range('2020-01-01', periods=n, freq=freq)
    return pd.DataFrame({'date': dates, 'value': value})


def _weekly_synthetic():
    """Weekly-spaced copy of the bundled synthetic series (180 periods)."""
    df = pt.load_data('series_synthetic')
    df['weekly_date'] = pd.date_range(start='2026-01-01', periods=len(df), freq='W')
    return df


class TestSignalParamsGranularity:
    """Cadence-aware ``window_smooth`` keeps a single uptrend intact per cadence."""

    def test_30min(self):
        """30-minute data (window 97): daily ripple must not fragment the uptrend."""
        df = _seasonal('30min', 336, 48, 120, 216)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2020-01-03 12:00'),
             'end': pd.Timestamp('2020-01-05 13:00')},
        ])

    def test_hourly(self):
        """Hourly data (window 49): the 4-day uptrend stays a single Up segment."""
        df = _seasonal('h', 336, 24, 120, 216)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2020-01-06 18:00'),
             'end': pd.Timestamp('2020-01-09 06:00')},
        ])
        assert [s['direction'] for s in results.segments].count('Up') == 1

    def test_15min(self):
        """15-minute data (window 193): uptrend detected without error."""
        df = _seasonal('15min', 672, 96, 240, 432)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2020-01-03 10:15'),
             'end': pd.Timestamp('2020-01-05 14:30')},
        ])

    def test_monthly(self):
        """Monthly data (window 25): yearly ripple must not hide the 18-month uptrend."""
        df = _seasonal('ME', 60, 12, 21, 39)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2021-10-31'),
             'end': pd.Timestamp('2023-04-30')},
        ])

    def test_quarterly(self):
        """Quarterly data (window 9): the 8-quarter uptrend is detected."""
        df = _seasonal('QE', 32, 4, 12, 20)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2022-03-31'),
             'end': pd.Timestamp('2025-06-30')},
        ])

    def test_yearly(self):
        """Yearly data (window 5): a clear 6-year uptrend is detected."""
        df = _seasonal('YE', 12, 6, 3, 9, ripple=False, noise=0.1)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2022-12-31'),
             'end': pd.Timestamp('2031-12-31')},
        ])

    def test_yearly_short_clamp(self):
        """A series shorter than the yearly window is clamped, not rejected."""
        rng = np.random.RandomState(1)
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=8, freq='YE'),
            'value': 100 + np.arange(8) * 5 + rng.normal(0, 0.1, 8),
        })
        # The inferred window (5) is clamped below the frame length; without the
        # clamp an unclamped window would make Savitzky-Golay raise a ValueError.
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2021-12-31'),
             'end': pd.Timestamp('2027-12-31')},
        ])

    def test_else_case_45min(self):
        """An unmapped cadence (45min) falls back to the default window of 15."""
        df = _seasonal('45min', 200, 32, 70, 130, ripple=False)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2020-01-03 00:00'),
             'end': pd.Timestamp('2020-01-05 05:15')},
        ])

    def test_override_wins(self):
        """An explicit window_smooth wins over cadence inference (historical 15)."""
        df = _weekly_synthetic()
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='weekly_date',
            plot=False,
            method_params={'abrupt_padding': 0},
            signal_params={'window_smooth': 15},
        )
        expected_segments = [
            {'direction': 'Up',   'start': pd.Timestamp('2026-01-11'), 'end': pd.Timestamp('2026-06-14')},
            {'direction': 'Down', 'start': pd.Timestamp('2026-06-21'), 'end': pd.Timestamp('2026-09-06')},
            {'direction': 'Flat', 'start': pd.Timestamp('2026-09-13'), 'end': pd.Timestamp('2026-10-04')},
            {'direction': 'Up',   'start': pd.Timestamp('2026-10-11'), 'end': pd.Timestamp('2027-06-13')},
            {'direction': 'Down', 'start': pd.Timestamp('2027-06-20'), 'end': pd.Timestamp('2027-09-26')},
            {'direction': 'Up',   'start': pd.Timestamp('2027-10-03'), 'end': pd.Timestamp('2028-06-11')},
            {'direction': 'Down', 'start': pd.Timestamp('2028-06-18'), 'end': pd.Timestamp('2029-03-18')},
            {'direction': 'Flat', 'start': pd.Timestamp('2029-03-25'), 'end': pd.Timestamp('2029-06-17')},
        ]
        assert_segments_match(results.segments, expected_segments)

    def test_weekly(self):
        """Weekly data (window 9): a 4-step cycle must not fragment the uptrend."""
        df = _seasonal('W', 104, 4, 37, 57, ripple_frac=0.1)
        results = pt.detect_trends(df, value_col='value', date_col='date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up',
             'start': pd.Timestamp('2020-09-13'),
             'end': pd.Timestamp('2021-02-07')},
        ])
