"""
Shared test utilities and fixtures for pytrendy tests.

This module contains helper functions and pytest fixtures that are used
across multiple test files.
"""

import pandas as pd

def assert_segments_match(detected_segments, expected_segments):
    """
    Helper function to validate that detected segments match expected segments.
    
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
    for i, (detected, expected) in enumerate(zip(detected_segments, expected_segments)):
        assert detected['direction'] == expected['direction'], \
            f"Segment {i}: Expected direction '{expected['direction']}', got '{detected['direction']}'"
        assert detected['start'] == expected['start'], \
            f"Segment {i}: Expected start '{expected['start']}', got '{detected['start']}'"
        assert detected['end'] == expected['end'], \
            f"Segment {i}: Expected end '{expected['end']}', got '{detected['end']}'"


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
