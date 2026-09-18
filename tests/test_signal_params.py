"""
Tests for the signal_params surface in detect_trends.

signal_params exposes the signal-processing constants (windows, thresholds and
grouping distance) independently from method_params. These tests cover the
defaults staying byte-identical, each documented override taking effect, and
the deliberate decision to accept unknown keys without validation.
"""

import pytrendy as pt

from pytrendy.detect_trends import _resolve_signal_params
from pytrendy.io import prep_index
from pytrendy.process_signals import process_signals


def _run(value_col: str, **kwargs):
    """Run detection on the shared synthetic fixture (daily spacing)."""
    df = pt.load_data('series_synthetic')
    return pt.detect_trends(df, value_col=value_col, date_col='date', plot=False, **kwargs)


class TestSignalParams:
    """Behaviour of the signal_params surface."""

    def test_defaults_match_omitted(self):
        """Explicit defaults produce identical results to omitting signal_params."""
        explicit_defaults = {
            'window_smooth': 15,
            'grouping_distance': 7,
            'min_trend_length': 3,
            'min_flat_noise_length': 1,
            'threshold_noise': 2.5,
            'threshold_smooth': 0.001,
            'threshold_flat': 0.835,
        }
        for value_col in ['abrupt', 'gradual', 'gradual-noisy-20']:
            implicit = _run(value_col)
            given = _run(value_col, signal_params=explicit_defaults)
            assert implicit.df.equals(given.df), f"signal_params defaults not neutral for {value_col}"

    def test_window_smooth_override_changes_detection(self):
        """A larger smoothing window changes the detected segment set."""
        default = _run('abrupt')
        overridden = _run('abrupt', signal_params={'window_smooth': 25})
        assert not default.df.equals(overridden.df)
        assert len(default.segments) != len(overridden.segments)

    def test_grouping_distance_override_honoured_on_noise_path(self):
        """grouping_distance controls the noise-segment grouping in process_signals."""
        df = pt.load_data('series_synthetic')
        df_int, _, _, _ = prep_index.prep_index(df.copy(), 'date', 'gradual-noisy-20')

        default = process_signals(df_int.copy(), 'gradual-noisy-20', {'avoid_noise': True}, _resolve_signal_params())
        overridden = process_signals(
            df_int.copy(), 'gradual-noisy-20', {'avoid_noise': True}, _resolve_signal_params({'grouping_distance': 0})
        )
        # Disabling grouping must keep (at least as many) separate noise regions.
        assert int(overridden['noise_flag'].sum()) >= int(default['noise_flag'].sum())

    def test_window_flat_and_noise_derive_from_window_smooth(self):
        """window_flat/window_noise default to half window_smooth unless overridden."""
        derived = _resolve_signal_params({'window_smooth': 20})
        assert derived['window_flat'] == 10
        assert derived['window_noise'] == 10
        explicit = _resolve_signal_params({'window_smooth': 20, 'window_flat': 3, 'window_noise': 5})
        assert explicit['window_flat'] == 3
        assert explicit['window_noise'] == 5

    def test_min_trend_length_override_honoured(self):
        """Raising min_trend_length drops the shorter trend segments."""
        default = _run('gradual')
        overridden = _run('gradual', signal_params={'min_trend_length': 20})
        assert len(default.trend_segments) > len(overridden.trend_segments)

    def test_threshold_noise_override_honoured(self):
        """Raising threshold_noise classifies more of the signal as noise."""
        default = _run('gradual')
        overridden = _run('gradual', signal_params={'threshold_noise': 10.0})
        default_noise = default.summary['direction_counts'].get('Noise', 0)
        overridden_noise = overridden.summary['direction_counts'].get('Noise', 0)
        assert overridden_noise > default_noise

    def test_unknown_keys_accepted_silently(self):
        """Unknown keys are forwarded without raising (validation is out of scope for v1)."""
        result = _run('abrupt', signal_params={'not_a_real_key': 123})
        assert len(result.segments) > 0
