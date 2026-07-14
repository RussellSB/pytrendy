"""
Tests for the gradual_padding method parameter.

Verifies that gradual segments can be padded forward by a specified number of days,
with correct overlap avoidance, dataset-end clamping, and the padded flag.
"""

import pytest
import pytrendy as pt
from conftest import assert_segments_match


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
            method_params=dict(gradual_padding=28)
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
    def test_gradual_padding_28_padded_flags(self):
        """Test that the padded flag is set correctly on gradual segments."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(gradual_padding=28)
        )
        # Segments that got extended should have padded=True
        # Up (2025-01-02 to 2025-01-24): next segment starts immediately, no room to pad
        assert results.segments[0].get('padded', False) is False
        # Down (2025-01-25 to 2025-02-09): padded from original ~2025-02-05 into the flat
        assert results.segments[1]['padded'] is True
        # Up (2025-02-10 to 2025-03-17): padded from original ~2025-03-14 into the flat
        assert results.segments[2]['padded'] is True
        # Down (2025-03-18 to 2025-04-01): next segment starts immediately, no room to pad
        assert results.segments[3].get('padded', False) is False
        # Up (2025-04-02 to 2025-05-08): next segment starts immediately, no room to pad
        assert results.segments[4].get('padded', False) is False
        # Down (2025-05-09 to 2025-06-30): padded to dataset end
        assert results.segments[5]['padded'] is True

    @pytest.mark.core
    def test_gradual_padding_168_clamps_to_nonflat(self):
        """Test large padding is truncated at the first overlapping non-Flat segment."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(gradual_padding=168)
        )
        # Even with 168 days of padding, overlap avoidance truncates to the same
        # boundaries as 28 days because the next non-Flat segment caps the extension.
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
    def test_gradual_padding_zero_no_change(self):
        """Test that gradual_padding=0 (default) produces the same result as no padding."""
        df = pt.load_data('series_synthetic')
        results_default = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
        )
        results_explicit = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(gradual_padding=0)
        )
        assert_segments_match(results_default.segments, results_explicit.segments)
