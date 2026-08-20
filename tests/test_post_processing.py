"""
Tests for post-processing pipeline internals.

These tests exercise the segment refinement helpers (artifact cleanup and
abrupt shaving) directly, covering branch conditions that the full
``detect_trends`` entry point does not reach on the standard datasets.
"""

import numpy as np

import pytrendy as pt


class TestArtifactCleanup:
    """Test artifact_cleanup edge cases."""

    def test_fill_flats_empty_segments(self):
        """fill_in_flats with an empty segment list fills the full range as a single Flat."""
        from pytrendy.post_processing.segments_refine.artifact_cleanup import fill_in_flats

        df = pt.load_data('series_synthetic')
        df_int = df.set_index(np.arange(len(df)))[['gradual']]

        result = fill_in_flats(df_int, [])

        assert len(result) == 1
        assert result[0]['direction'] == 'Flat'
        assert result[0]['start'] == df_int.index.min()
        assert result[0]['end'] == df_int.index.max()


class TestAbruptShaving:
    """Test abrupt_shaving edge cases."""

    def test_abrupt_shaving_leading_subsegment(self):
        """shave_abrupt_trends processes abrupt segments and returns the refined list."""
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

        result = shave_abrupt_trends(df_processed, 'abrupt', segments, method_params)

        assert isinstance(result, list)
