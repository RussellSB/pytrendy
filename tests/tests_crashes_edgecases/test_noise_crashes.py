"""
Tests for noise crash scenarios in trend detection algorithm.

These tests verify that the trend detection algorithm does not crash or hang
when processing edge cases that were historically found to cause execution errors
or infinite loops. The test data comes from random noise generation but captured
real-world failures to enable continuous regression testing.

Reference: tests/test.py lines 265-335 (Previous Crash Instances section)
Reference: tests/data/TESTDATA.md - noisy_crashes.csv description

Note: This test file was created from exploratory tests in test.py to provide
formal test coverage ensuring the algorithm never crashes on these scenarios again.
Each test loads a specific column from noisy_crashes.csv that previously caused
the algorithm to crash or hang.
"""

import pytest
import time
import pandas as pd
import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestNoiseCrashes:
    """Test cases for noise scenarios that previously caused crashes or hangs."""

    
    def test_temp_scenario(self):
        """
        Test that algorithm handles temp scenario without crashing.
        
        Additional crash scenario preserved in test data.
        Verifies algorithm stability on this edge case.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='temp', #TODO: reword temp, temp_2 etc to be consequetively classed
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )

        # Expected segments to find a subset of based on current behavior
        expected_segments = [
            {'direction': 'Down', 'start': '2025-05-15', 'end': '2025-06-07'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_temp_2_scenario(self):
        """
        Test that algorithm handles temp_2 scenario without hanging.
        
        Reference: test.py line 327 (temp_2.csv)
        
        This scenario was noted with "TODONE: fix hangup".
        This test ensures the algorithm completes in reasonable time without hanging (through pytest-timeout).
        And also ensures it returns a sensible segment.

        Before it would enter an infinite loop due to incorrect abrupt shaving logic. 
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='temp_2',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )

        # Expected segments to find a subset of based on current behavior
        expected_segments = [
            {'direction': 'Down', 'start': '2025-05-10', 'end': '2025-06-06'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_crash_scenario(self):
        """
        Test that algorithm handles noisy_crash scenario without crashing.
        
        Reference: test.py line 269 (temp_noisy_crash_7.csv) and related
        
        This scenario was found to crash pytrendy with execution errors.
        The test verifies that detect_trends completes successfully without
        hanging or throwing exceptions.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Flat', 'start': '2025-03-31', 'end': '2025-04-10'}, 
            {'direction': 'Up', 'start': '2025-04-11', 'end': '2025-05-04'}, 
            {'direction': 'Down', 'start': '2025-05-05', 'end': '2025-06-12'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_crash_2_scenario(self):
        """
        Test that algorithm handles noisy_crash_2 scenario without crashing.
        
        Reference: test.py line 315 (temp_noisy_crash_2.csv)
        
        This scenario represents another crash case found during testing.
        Verifies the algorithm completes without errors.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash_2',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Down', 'start': '2025-03-16', 'end': '2025-03-30'}, 
            {'direction': 'Up', 'start': '2025-04-03', 'end': '2025-05-04'}, 
            {'direction': 'Down', 'start': '2025-05-05', 'end': '2025-06-03'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_crash_4_scenario(self):
        """
        Test that algorithm handles noisy_crash_4 scenario without crashing.
        
        Reference: test.py line 286 (temp_noisy_crash_4.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Verifies the fix remains stable.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash_4',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-12', 'end': '2025-05-06'}, 
            {'direction': 'Down', 'start': '2025-05-07', 'end': '2025-06-12'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_crash_5_scenario(self):
        """
        Test that algorithm handles noisy_crash_5 scenario without crashing.
        
        Reference: test.py line 280 (temp_noisy_crash_5.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Verifies the algorithm processes it successfully.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash_5',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-11', 'end': '2025-04-29'}, 
            {'direction': 'Flat', 'start': '2025-04-30', 'end': '2025-05-17'}, 
            {'direction': 'Down', 'start': '2025-05-18', 'end': '2025-06-12'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_crash_6_scenario(self):
        """
        Test that algorithm handles noisy_crash_6 scenario without crashing.
        
        Reference: test.py line 274 (temp_noisy_crash_6.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Ensures continued stability on this edge case.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash_6',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-12', 'end': '2025-05-08'}, 
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-17'}, 
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-29'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_crash_7_scenario(self):
        """
        Test that algorithm handles noisy_crash_7 scenario without crashing.
        
        Reference: test.py line 269 (temp_noisy_crash_7.csv)
        
        This scenario was noted with "TODONE: fix when padded out of bound # TODONE: crash fix".
        This was one of the critical crash fixes. Verifies algorithm stability.
        """
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_crashes.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_crash_7',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Expected segments to find a subset of based on current behavior
        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-03-31', 'end': '2025-05-09'}, #TODO: may expect some change if noise disabled
            {'direction': 'Down', 'start': '2025-05-14', 'end': '2025-06-04'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)
