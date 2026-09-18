"""
Tests for post-processing pipeline behaviour.

These tests drive the public ``detect_trends`` entry point and assert on
segments via the shared conftest helpers. A single irreducible direct call is
retained: ``fill_in_flats`` with an empty segment list, which the integration
pipeline cannot produce because a constant series is always classified Flat
rather than yielding zero segments.
"""

import pytest

import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestAbruptTrends:
    """Abrupt trend detection through the public entry point."""

    @pytest.mark.core
    def test_abrupt_trends_classified(self):
        """detect_trends classifies abrupt up/down runs on the synthetic abrupt series."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            value_col='abrupt',
            date_col='date',
            plot=False,
            method_params={'abrupt_padding': 0},
        )

        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': '2025-02-28', 'end': '2025-03-01'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-05-05'},
        ])


class TestFillInFlats:
    """Direct coverage for the empty-input guard in ``fill_in_flats``.

    ``detect_trends`` always produces at least one segment after refinement
    (a constant series is classified Flat), so the empty-input branch cannot be
    reached end-to-end. This direct call is the only coverage for the
    ``artifact_cleanup`` empty-segments guard.
    """

    def test_fill_flats_empty_segments(self):
        """An empty segment list is filled with a single Flat covering the full range."""
        import numpy as np
        from pytrendy.post_processing.segments_refine.artifact_cleanup import fill_in_flats

        df = pt.load_data('series_synthetic')
        df_int = df.set_index(np.arange(len(df)))[['gradual']]

        result = fill_in_flats(df_int, [])

        assert result == [{
            'direction': 'Flat',
            'start': df_int.index.min(),
            'end': df_int.index.max(),
        }]
