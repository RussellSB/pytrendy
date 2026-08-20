"""
Shared test utilities and fixtures for pytrendy tests.

This module contains helper functions and pytest fixtures that are used
across multiple test files.
"""

import pandas as pd
import numpy as np

from pytrendy.io import prepare_index


def build_internal_index(df: pd.DataFrame, date_col: str) -> tuple:
    """
    Build the internal index framework for unwrapped-pipeline tests.

    Uses the production ``prepare_index.build_index_lookup`` helper so tests that
    bypass the main entry point stay in sync with the detection logic in a single place.

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of the column to use as the external index.

    Returns:
        tuple: ``(external_index, internal_index, index_lookup)`` where
        ``external_index`` holds the datetime index values, ``internal_index``
        is the positional integer sequence, and ``index_lookup`` maps internal
        to external index values.
    """
    external_index = pd.to_datetime(df[date_col])
    internal_index = np.arange(len(df))
    index_lookup = prepare_index.build_index_lookup(external_index)
    return external_index, internal_index, index_lookup


def _assert_boundary_equal(boundary: str, actual, expected, segment_index: int) -> None:
    """
    Assert a single segment boundary (start or end) equals the expected value.

    Float boundaries are compared with 6-decimal rounding to tolerate the
    floating-point representation differences introduced by float index columns.

    Args:
        boundary: Boundary name ('start' or 'end'), used for the error message.
        actual: The detected boundary value.
        expected: The expected boundary value.
        segment_index: Index of the segment, used for the error message.
    """
    if isinstance(actual, float):
        assert round(actual, 6) == round(expected, 6), \
            f"Segment {segment_index}: Expected {boundary} '{expected}', got '{actual}'"
    else:
        assert actual == expected, \
            f"Segment {segment_index}: Expected {boundary} '{expected}', got '{actual}'"


def assert_segments_match(detected_segments, expected_segments) -> None:
    """
    Assert detected segments exactly match expected segments (count, order, direction, boundaries).

    Args:
        detected_segments: List of segment dicts with keys 'direction', 'start', 'end'.
        expected_segments: List of segment dicts with the same structure.

    Raises:
        AssertionError: If the segments don't match in count, direction, or boundaries.
    """
    assert len(detected_segments) == len(expected_segments), \
        f"Expected {len(expected_segments)} segments, got {len(detected_segments)}"

    for i, (detected, expected) in enumerate(zip(detected_segments, expected_segments)):
        assert detected['direction'] == expected['direction'], \
            f"Segment {i}: Expected direction '{expected['direction']}', got '{detected['direction']}'"

        _assert_boundary_equal('start', detected['start'], expected['start'], i)
        _assert_boundary_equal('end', detected['end'], expected['end'], i)


def assert_segments_in_a_haystack(detected_segments, expected_segments) -> None:
    """
    Assert expected segments appear within detected segments (subset match, order-independent).

    Args:
        detected_segments: List of segment dicts with keys 'direction', 'start', 'end'.
        expected_segments: List of segment dicts expected to be present in the detected segments.

    Raises:
        AssertionError: If any expected segment is not found in the detected segments.
    """
    unmatched = [(segment['direction'], segment['start'], segment['end']) for segment in detected_segments]

    for expected in expected_segments:
        expected_tuple = (expected['direction'], expected['start'], expected['end'])

        if expected_tuple not in unmatched:
            raise AssertionError(f"Expected {expected_tuple} could not be found in detected trends.")

        unmatched.remove(expected_tuple)
