"""
Unit tests for the `assert_segments_match` test helper in conftest.py.

This helper now tolerates minor floating point noise (rounding to 6 decimal
places) when comparing `start`/`end` values that are floats, to support the
new float-index test cases, while still requiring exact equality for
non-float values (e.g. integers, strings, dates).
"""

import pytest
from conftest import assert_segments_match


class TestAssertSegmentsMatchFloatTolerance:
    """Tests for the float-rounding tolerance added to assert_segments_match."""

    def test_matches_exact_float_values(self):
        """Identical float values should match."""
        detected = [{'direction': 'Up', 'start': 0.1, 'end': 0.2}]
        expected = [{'direction': 'Up', 'start': 0.1, 'end': 0.2}]
        assert_segments_match(detected, expected)  # should not raise

    def test_matches_float_within_rounding_tolerance(self):
        """Floating point noise beyond the 6th decimal place should still match."""
        detected = [{'direction': 'Up', 'start': 0.1 + 1e-9, 'end': 0.2 - 1e-9}]
        expected = [{'direction': 'Up', 'start': 0.1, 'end': 0.2}]
        assert_segments_match(detected, expected)  # should not raise

    def test_mismatched_float_beyond_tolerance_raises(self):
        """Differences within the first 6 decimal places should still fail."""
        detected = [{'direction': 'Up', 'start': 0.100002, 'end': 0.2}]
        expected = [{'direction': 'Up', 'start': 0.1, 'end': 0.2}]
        with pytest.raises(AssertionError):
            assert_segments_match(detected, expected)

    def test_non_float_values_use_exact_equality(self):
        """Integer start/end values should still require exact equality."""
        detected = [{'direction': 'Up', 'start': 5, 'end': 10}]
        expected = [{'direction': 'Up', 'start': 5, 'end': 10}]
        assert_segments_match(detected, expected)  # should not raise

        mismatched_expected = [{'direction': 'Up', 'start': 5, 'end': 11}]
        with pytest.raises(AssertionError):
            assert_segments_match(detected, mismatched_expected)

    def test_string_values_use_exact_equality(self):
        """String start/end labels should still require exact equality."""
        detected = [{'direction': 'Up', 'start': 'Step 1', 'end': 'Step 5'}]
        expected = [{'direction': 'Up', 'start': 'Step 1', 'end': 'Step 5'}]
        assert_segments_match(detected, expected)  # should not raise

        mismatched_expected = [{'direction': 'Up', 'start': 'Step 1', 'end': 'Step 6'}]
        with pytest.raises(AssertionError):
            assert_segments_match(detected, mismatched_expected)

    def test_direction_mismatch_still_raises(self):
        """Mismatched direction should raise regardless of float tolerance changes."""
        detected = [{'direction': 'Up', 'start': 0.1, 'end': 0.2}]
        expected = [{'direction': 'Down', 'start': 0.1, 'end': 0.2}]
        with pytest.raises(AssertionError):
            assert_segments_match(detected, expected)