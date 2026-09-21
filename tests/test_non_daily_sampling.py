"""
Regression tests for spacing-aware (duration-preserving) detection windows.

Detection windows are calibrated in days and converted to point counts from the
series' median sampling gap. These tests cover the scaling helpers and verify
that non-daily data (fortnightly, monthly) recovers cycles that the historical
fixed point counts collapsed, while daily behaviour is byte-identical.
"""

import numpy as np
import pandas as pd
import pytest

import pytrendy as pt
from pytrendy._spacing import compute_gap_days, scale_window
from pytrendy.io import prep_signal_params


def _base_daily(n: int = 547, period: float = 52.0, amp: float = 20.0) -> pd.DataFrame:
    """Daily ~52-day seasonal series; the control used by the issue's audit."""
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    t = np.arange(n)
    return pd.DataFrame({'date': dates, 'value': 100.0 + amp * np.sin(2 * np.pi * t / period)})


def _fortnightly(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index('date').resample('W').last().iloc[::2].reset_index()


def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index('date').resample('ME').last().reset_index()


def _segments(df: pd.DataFrame) -> list[dict]:
    results = pt.detect_trends(df, date_col='date', value_col='value', plot=False)
    return results.segments


def _directions(segments: list[dict]) -> list[str]:
    return [segment['direction'] for segment in segments]


class TestSpacingHelpers:
    """Unit coverage for the duration-preserving scaling helpers."""

    def test_scale_window_daily_reproduces_historical_counts(self):
        assert scale_window(15, 1.0) == 15
        assert scale_window(7, 1.0) == 7
        assert scale_window(3, 1.0) == 3

    def test_scale_window_floors_at_minimum(self):
        # 15 days at monthly spacing would round to 1 point, floored to 2 for savgol.
        assert scale_window(15, 31.0, minimum=2) == 2
        assert scale_window(7, 31.0) == 1
        assert scale_window(3, 14.0) == 1

    @pytest.mark.parametrize('bad_gap', [0, -3.0, float('nan')])
    def test_scale_window_invalid_gap_falls_back_to_daily(self, bad_gap):
        assert scale_window(15, bad_gap) == 15

    def test_compute_gap_days_daily_and_fortnightly(self):
        daily = pd.date_range('2025-01-01', periods=30, freq='D')
        fortnightly = pd.date_range('2025-01-01', periods=30, freq='14D')
        assert compute_gap_days(daily, 'datetime64') == 1.0
        assert compute_gap_days(fortnightly, 'datetime64') == 14.0

    def test_compute_gap_days_date_strings(self):
        dates = pd.Series(['2025-01-01', '2025-01-31', '2025-03-02'])
        assert compute_gap_days(dates, 'date') == 30.0

    def test_compute_gap_days_short_series_returns_daily(self):
        assert compute_gap_days(pd.Series([pd.Timestamp('2025-01-01')]), 'datetime64') == 1.0

    def test_compute_gap_days_non_date_index_returns_daily(self):
        assert compute_gap_days(pd.Series([0, 7, 14]), 'integer') == 1.0

    def test_compute_gap_days_duplicate_dates_returns_daily(self):
        duplicated = pd.Series(pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-01']))
        assert compute_gap_days(duplicated, 'datetime64') == 1.0

    def test_derived_defaults_per_cadence(self):
        """The derived point counts are documented in the PR's cadence table."""
        assert scale_window(15, 14.0, minimum=2) == 2
        assert scale_window(15, 31.0, minimum=2) == 2
        for base in (7, 7, 3, 1):
            assert scale_window(base, 14.0) == 1
            assert scale_window(base, 31.0) == 1

    def test_intraday_windows_span_a_day(self):
        """Sub-daily cadences derive from the ~24 h rung (24 steps at 1 h, 48 at 30 min)."""
        hourly = prep_signal_params.prep_signal_params(None, 1 / 24)
        assert hourly['window_smooth'] == 24
        half_hourly = prep_signal_params.prep_signal_params(None, 1 / 48)
        assert half_hourly['window_smooth'] == 48
        # Grouping distance and minimum lengths are fractions of the same span.
        assert half_hourly['grouping_distance'] == 22
        assert half_hourly['min_trend_length'] == 9
        assert half_hourly['min_flat_noise_length'] == 3

    def test_window_clamped_to_series_length(self):
        """A derived window can never exceed a short series."""
        clamped = prep_signal_params.prep_signal_params(None, 1 / 48, n_obs=20)
        assert clamped['window_smooth'] == 20

    def test_invalid_gap_falls_back_to_daily_span(self):
        """A non-finite gap keeps the historical daily window, matching scale_window."""
        assert prep_signal_params.prep_signal_params(None, float('nan'))['window_smooth'] == 15


class TestNonDailySampling:
    """End-to-end detection on cadences the fixed point counts collapsed."""

    def test_fortnightly_detects_cycles(self):
        segments = _segments(_fortnightly(_base_daily()))
        directions = _directions(segments)
        # Historical behaviour: a single Flat covering the whole series.
        assert len(segments) > 1
        assert directions.count('Down') >= 1
        assert directions.count('Up') >= 1
        assert not (directions == ['Flat'])

    def test_monthly_detects_up_leg_without_noise_ramp(self):
        segments = _segments(_monthly(_base_daily()))
        directions = _directions(segments)
        # Historical behaviour: 2 Flat + 1 spurious Noise band over the ramp.
        assert 'Noise' not in directions
        assert 'Up' in directions
        assert directions.count('Flat') <= 2

    def test_daily_equivalence_seasonal(self):
        """Daily segments must be byte-identical to stock develop (gap=1 path)."""
        expected = [
            ('Up', '2025-01-02', '2025-01-14'),
            ('Down', '2025-01-15', '2025-02-09'),
            ('Up', '2025-02-10', '2025-03-07'),
            ('Down', '2025-03-08', '2025-04-02'),
            ('Up', '2025-04-03', '2025-04-28'),
            ('Down', '2025-04-29', '2025-05-24'),
            ('Up', '2025-05-25', '2025-06-19'),
            ('Down', '2025-06-20', '2025-07-15'),
            ('Up', '2025-07-16', '2025-08-10'),
            ('Down', '2025-08-11', '2025-09-05'),
            ('Up', '2025-09-06', '2025-10-01'),
            ('Down', '2025-10-02', '2025-10-27'),
            ('Up', '2025-10-28', '2025-11-22'),
            ('Down', '2025-11-23', '2025-12-18'),
            ('Up', '2025-12-19', '2026-01-13'),
            ('Down', '2026-01-14', '2026-02-08'),
            ('Up', '2026-02-09', '2026-03-06'),
            ('Down', '2026-03-07', '2026-04-01'),
            ('Up', '2026-04-02', '2026-04-27'),
            ('Down', '2026-04-28', '2026-05-23'),
            ('Up', '2026-05-24', '2026-06-18'),
            ('Down', '2026-06-19', '2026-07-01'),
        ]
        actual = [
            (s['direction'], pd.Timestamp(s['start']).strftime('%Y-%m-%d'),
             pd.Timestamp(s['end']).strftime('%Y-%m-%d'))
            for s in _segments(_base_daily())
        ]
        assert actual == expected

    def test_daily_synthetic_unchanged(self):
        """Single-column daily frame on the bundled dataset is unchanged."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, date_col='date', value_col='gradual', plot=False)
        assert len(results.segments) == 8
        assert results.segments[0]['direction'] == 'Up'


class TestWindowOverrides:
    """signal_params overrides are honoured as raw point counts (no scaling)."""

    def test_all_window_overrides_accepted(self):
        segments = _segments_with_overrides(
            _fortnightly(_base_daily()),
            window_smooth=15,
            window_flat=7,
            window_noise=7,
            grouping_distance=7,
            min_trend_length=3,
        )
        assert isinstance(segments, list)
        assert len(segments) >= 1

    def test_min_trend_length_override_changes_detection(self):
        df = _fortnightly(_base_daily())
        default_segments = _segments(df)
        long_min_segments = _segments_with_overrides(df, min_trend_length=50)
        # A 50-point minimum trend cannot be met by a 40-point series.
        assert 'Up' not in _directions(long_min_segments)
        assert 'Down' not in _directions(long_min_segments)
        assert len(long_min_segments) <= len(default_segments)

    def test_grouping_distance_override_changes_detection(self):
        df = _fortnightly(_base_daily())
        tight = _segments_with_overrides(df, grouping_distance=1)
        loose = _segments_with_overrides(df, grouping_distance=100)
        # Grouping distance 100 merges every same-direction run into one segment.
        assert len(loose) <= len(tight)

    def test_window_flat_override_accepted(self):
        segments = _segments_with_overrides(_fortnightly(_base_daily()), window_flat=3, window_noise=3)
        assert len(segments) >= 1


def _segments_with_overrides(df: pd.DataFrame, **overrides) -> list[dict]:
    results = pt.detect_trends(
        df, date_col='date', value_col='value', plot=False, signal_params=dict(overrides)
    )
    return results.segments
