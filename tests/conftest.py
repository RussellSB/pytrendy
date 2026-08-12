"""
Shared test utilities and fixtures for pytrendy tests.

This module contains helper functions and pytest fixtures that are used
across multiple test files.
"""

import pandas as pd
import numpy as np


def build_internal_index(df: pd.DataFrame, date_col: str) -> tuple:
    """
    Build the internal index framework for unwrapped-pipeline tests.

    Mirrors the external_index / internal_index / index_lookup construction
    used by ``detect_trends``, so tests that bypass the main entry point stay
    in sync with the production logic in a single place.

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
    index_lookup = dict(zip(internal_index, np.asarray(external_index)))
    return external_index, internal_index, index_lookup


def _check_value(key, detected, expected, i):
    """
    Helper function to check a single value (start or end) against expected value.

    Args:
        key: The key name ('start' or 'end')
        detected: The detected value
        expected: The expected value
        i: Segment index for error messages
    """
    if isinstance(detected, float):
        assert round(detected, 6) == round(expected, 6), \
            f"Segment {i}: Expected {key} '{expected}', got '{detected}'"
    else:
        assert detected == expected, \
            f"Segment {i}: Expected {key} '{expected}', got '{detected}'"

def assert_segments_match(detected_segments, expected_segments):
    """
    Helper function to validate that detected segments match expected segments.
    
    This function compares detected trend segments against expected segments,
    validating that the direction, start time, and end time match for each segment.
    
    Args:
        detected_segments: List of dictionaries, each representing a detected segment.
            Each dictionary must have the following keys:
                - 'direction': str, the direction of the segment ('Up', 'Down', 'Flat', 'Noise')
                - 'start': str, Timestamp, int, or float, the start time of the segment
                - 'end': str, Timestamp, int, or float, the end time of the segment
        expected_segments: List of dictionaries with the same structure as detected_segments.
            Each dictionary must have the following keys:
                - 'direction': str, the direction of the segment ('Up', 'Down', 'Flat', 'Noise')
                - 'start': str, Timestamp, int, or float, the start time of the segment
                - 'end': str, Timestamp, int, or float, the end time of the segment
    
    Raises:
        AssertionError: If the segments don't match in count, direction, or time boundaries.
    """
    # Assert number of segments matches
    assert len(detected_segments) == len(expected_segments), \
        f"Expected {len(expected_segments)} segments, got {len(detected_segments)}"
    
    # Assert each segment matches expected values
    for i, (detected, expected) in enumerate(zip(detected_segments, expected_segments)):
        assert detected['direction'] == expected['direction'], \
            f"Segment {i}: Expected direction '{expected['direction']}', got '{detected['direction']}'"
        
        _check_value('start', detected['start'], expected['start'], i)
        _check_value('end', detected['end'], expected['end'], i)


def assert_segments_in_a_haystack(detected_segments, expected_segments):
    """
    Similar to assert_segments_match but allows for expected segments to be a subset of detected segments.
    
    Args:
        detected_segments: List of dictionaries, each representing a detected segment.
            Each dictionary must have the following keys:
                - 'direction': str, the direction of the segment ('Up', 'Down', 'Flat', 'Noise')
                - 'start': str or Timestamp, the start date of the segment
                - 'end': str or Timestamp, the end date of the segment
        expected_segments: List of dictionaries with the same structure as detected_segments.
            Each dictionary must have the following keys:
                - 'direction': str, the direction of the segment ('Up', 'Down', 'Flat', 'Noise')
                - 'start': str, the start date of the segment in 'YYYY-MM-DD' format
                - 'end': str, the end date of the segment in 'YYYY-MM-DD' format
    
    Raises:
        AssertionError: If expected segments (needle) are not found in the detected segments (haystack).
    """
    unmatched_detected = [(segment['direction'], segment['start'], segment['end']) for segment in detected_segments]
    for _, expected in enumerate(expected_segments):
        expected_tuple = (expected['direction'], expected['start'], expected['end'])

        if expected_tuple not in unmatched_detected:
            assert False, f"Expected {expected_tuple} could not be found in detected trends."
        
        unmatched_detected.remove(expected_tuple) # removes matched detected with expected and continues
