"""
Tests for edge case scenarios caused by unusual data in trend detection algorithm.

These tests verify that the trend detection algorithm handles situations where the data is doing something 
we wouldn't typically expect.

Reference: tests/tests_crashes_edgecases/data/TESTDATA.md
"""

import pytest
import pandas as pd
import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestUncommonValues:
    """Test cases for Scenarios with Uncommon or Unusual Values"""

    def test_low_value_series(self):
        """Test that algorithm handles a low/normalised value series (data in range [0, 1]) reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/low_value_series.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='trend',
            plot=False,
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2000-01-02', 'end': '2000-01-14'},
            {'direction': 'Flat', 'start': '2000-01-15', 'end': '2000-01-18'},
            {'direction': 'Down', 'start': '2000-01-19', 'end': '2000-03-19'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_zero_baseline_market_entry_no_padding(self):
        """Test that a new-market series (long zero baseline, abrupt activation) detects a short
        Up trend when abrupt_padding=0."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/zero_baseline_edgecases_1.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='zero_baseline_market_entry_1',
            plot=False,
            method_params=dict(abrupt_padding=0)
        )

        expected_segments = [
            {'direction': 'Up', 'start': '2026-03-21', 'end': '2026-03-23'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_zero_baseline_market_entry_with_padding(self):
        """Test that a new-market series with abrupt_padding=28 correctly extends the Up segment
        rather than collapsing the entire window to Flat.

        Expected: the padded Up segment should extend to 2026-04-20 (28 days after activation)
        and the trailing Flat should cover 2026-04-21 to 2026-05-15.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/zero_baseline_edgecases_1.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='zero_baseline_market_entry_1',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        # Desired: at minimum an Up segment survives and is extended by padding.
        expected_segments = [
            {'direction': 'Up', 'start': '2026-03-21', 'end': '2026-04-20'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    @pytest.mark.core
    def test_zero_baseline_no_noise_segment(self):
        """Test that zero-baseline leading edge does not produce a Noise segment.

        The centred rolling mean in noise detection looks ahead at an abrupt transition,
        producing signal≈noise and a false low SNR on the last few zero days. This was
        fixed by suppressing noise_flag when value=0, previous=0, and signal!=0.

        Reference: issue #163, Problem 1
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/zero_baseline_edgecases_2.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='zero_baseline_market_entry_2',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        # Flat → Up, no Noise segment between them
        expected_segments = [
            {'direction': 'Flat', 'start': '2026-02-01', 'end': '2026-05-05'},
            {'direction': 'Up', 'start': '2026-05-06', 'end': '2026-05-13'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    @pytest.mark.core
    def test_zero_baseline_up_detected(self):
        """Regression test for Up detection on zero-baseline market entry (Problem 2 / smaller ramp).

        Reference: issue #163, Problem 2
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/zero_baseline_edgecases_2.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='zero_baseline_market_entry_3',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )
        expected_segments = [
            {'direction': 'Up', 'start': '2026-05-06', 'end': '2026-05-13'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    @pytest.mark.core
    def test_zero_baseline_spikes_detected_as_noise(self):
        """Spikes on a zero baseline are detected as Noise, not trends.

        Reference: issue #163 extension — zero-baseline with scattered spikes.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/zero_baseline_edgecases_2.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='zero_baseline_spikes',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        # Each spike produces a Noise segment (spike date is inside the window)
        expected_noise = [
            {'direction': 'Noise', 'start': '2026-02-06', 'end': '2026-02-08'},
            {'direction': 'Noise', 'start': '2026-02-28', 'end': '2026-03-02'},
            {'direction': 'Noise', 'start': '2026-03-31', 'end': '2026-04-02'},
            {'direction': 'Noise', 'start': '2026-04-19', 'end': '2026-04-21'},
            {'direction': 'Noise', 'start': '2026-05-11', 'end': '2026-05-13'},
            {'direction': 'Noise', 'start': '2026-05-26', 'end': '2026-05-28'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_noise)

    @pytest.mark.core
    def test_gradual_ramp_90day_detected(self):
        """A sustained 90-day gradual ramp is detected as a single uptrend,
        both with default params and with abrupt_padding=28 + avoid_noise=False.

        Regression test for issue #195: long gradual ramps were truncated because
        the flat detection threshold was too aggressive during gradual ramps.

        Reference: issue #195
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/gradual_ramp_edgecases.csv')
        expected = [
            {'direction': 'Up', 'start': '2026-04-07', 'end': '2026-06-29'},
            {'direction': 'Down', 'start': '2026-09-27', 'end': '2026-10-26'},
        ]

        results = pt.detect_trends(df, date_col='date', value_col='gradual_ramp_90d', plot=False,
                                   method_params=dict(abrupt_padding=28))
        assert_segments_in_a_haystack(results.segments, expected)
