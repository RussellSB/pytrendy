"""
Tests for the gradual_padding method parameter.

Verifies that gradual segments can be padded forward by a specified number of days,
with correct overlap avoidance and dataset-end clamping.
"""

import pytest
import pytrendy as pt
from conftest import assert_segments_match, assert_segments_in_a_haystack


class TestGradualPadding:
    """Test cases for gradual_padding behaviour."""

    @pytest.mark.core
    def test_gradual_padding_28(self):
        """Test gradual padding of 28 days extends segments into flat regions."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params={'gradual_padding': 28}
        )
        expected_segments = [
            {'direction': 'Up',   'start': '2025-01-02', 'end': '2025-01-24'},
            {'direction': 'Down', 'start': '2025-01-25', 'end': '2025-02-09'},
            {'direction': 'Up',   'start': '2025-02-10', 'end': '2025-03-17'},
            {'direction': 'Down', 'start': '2025-03-18', 'end': '2025-04-01'},
            {'direction': 'Up',   'start': '2025-04-02', 'end': '2025-05-08'},
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-30'},
        ]
        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_gradual_padding_clamps_to_nonflat(self):
        """Test large padding is truncated before the next non-Flat segment.

        Uses gradual_ramp_edgecases where a long gradual Up would extend far
        with 168 days of padding but gets clamped before an abrupt Down segment,
        demonstrating the overlap-avoidance logic with a varied dataset.
        """
        import pandas as pd
        df = pd.read_csv('tests/tests_crashes_edgecases/data/gradual_ramp_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual_ramp_90d',
            plot=False,
            method_params={'gradual_padding': 168}
        )
        # 168-day padding would theoretically reach Dec 14, but the abrupt Down
        # at Sep 27 caps the extension at Sep 19 (the Flat before it is absorbed).
        expected_segments = [
            {'direction': 'Flat', 'start': '2026-01-01', 'end': '2026-04-06'},
            {'direction': 'Up',   'start': '2026-04-07', 'end': '2026-09-19'},
            {'direction': 'Flat', 'start': '2026-09-20', 'end': '2026-09-26'},
            {'direction': 'Down', 'start': '2026-09-27', 'end': '2026-09-28'},
            {'direction': 'Flat', 'start': '2026-09-29', 'end': '2026-12-31'},
        ]
        assert_segments_match(results.segments, expected_segments)
