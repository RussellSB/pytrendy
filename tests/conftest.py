"""
Shared test utilities and fixtures for pytrendy tests.

This module contains helper functions and pytest fixtures that are used
across multiple test files.
"""

import pandas as pd


def assert_segments_match(detected_segments, expected_segments):
    """
    Helper function to validate that actual segments match expected segments.
    
    This function compares detected trend segments against expected segments,
    validating that the direction, start date, and end date match for each segment.
    
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
        AssertionError: If the segments don't match in count, direction, or date boundaries.
    """
    # Assert number of segments matches
    assert len(detected_segments) == len(expected_segments), \
        f"Expected {len(expected_segments)} segments, got {len(detected_segments)}"
    
    # Assert each segment matches expected values
    for i, (actual, expected) in enumerate(zip(detected_segments, expected_segments)):
        assert actual['direction'] == expected['direction'], \
            f"Segment {i}: Expected direction '{expected['direction']}', got '{actual['direction']}'"
        assert pd.to_datetime(actual['start']).strftime('%Y-%m-%d') == expected['start'], \
            f"Segment {i}: Expected start '{expected['start']}', got '{actual['start']}'"
        assert pd.to_datetime(actual['end']).strftime('%Y-%m-%d') == expected['end'], \
            f"Segment {i}: Expected end '{expected['end']}', got '{actual['end']}'"


def assert_segments_in_a_haystack(detected_segments, expected_segments):
    """
    Similar to assert_segments_match but allows for expected segments to be a subset of actual segments.
    
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
    unmatched_actual = [( 
        segment['direction'],
        pd.to_datetime(segment['start']).strftime('%Y-%m-%d'),
        pd.to_datetime(segment['end']).strftime('%Y-%m-%d'),
    ) for segment in detected_segments]

    for i, expected in enumerate(expected_segments):
        expected_tuple = (expected['direction'], expected['start'], expected['end'])

        if expected_tuple in unmatched_actual:
            unmatched_actual.remove(expected_tuple)
            continue

        raise AssertionError(f"Expected {expected_tuple} could not be found in detected trends.")