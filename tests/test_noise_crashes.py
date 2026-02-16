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


class TestNoiseCrashes:
    """Test cases for noise scenarios that previously caused crashes or hangs."""


    def test_noisy_crash_scenario(self):
        """
        Test that algorithm handles noisy_crash scenario without crashing.
        
        Reference: test.py line 269 (temp_noisy_crash_7.csv) and related
        
        This scenario was found to crash pytrendy with execution errors.
        The test verifies that detect_trends completes successfully without
        hanging or throwing exceptions.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash']].copy()
        test_df.columns = ['date', 'value']
        
        # Test with padded=True (original crash scenario)
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        # Assert that we got results and didn't crash
        assert len(results.segments) > 0


    def test_noisy_crash_2_scenario(self):
        """
        Test that algorithm handles noisy_crash_2 scenario without crashing.
        
        Reference: test.py line 315 (temp_noisy_crash_2.csv)
        
        This scenario represents another crash case found during testing.
        Verifies the algorithm completes without errors.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash_2']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    def test_noisy_crash_4_scenario(self):
        """
        Test that algorithm handles noisy_crash_4 scenario without crashing.
        
        Reference: test.py line 286 (temp_noisy_crash_4.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Verifies the fix remains stable.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash_4']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    def test_noisy_crash_5_scenario(self):
        """
        Test that algorithm handles noisy_crash_5 scenario without crashing.
        
        Reference: test.py line 280 (temp_noisy_crash_5.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Verifies the algorithm processes it successfully.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash_5']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    def test_noisy_crash_6_scenario(self):
        """
        Test that algorithm handles noisy_crash_6 scenario without crashing.
        
        Reference: test.py line 274 (temp_noisy_crash_6.csv)
        
        This scenario was fixed with comment "TODONE: doesnt crash now".
        Ensures continued stability on this edge case.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash_6']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    def test_noisy_crash_7_scenario(self):
        """
        Test that algorithm handles noisy_crash_7 scenario without crashing.
        
        Reference: test.py line 269 (temp_noisy_crash_7.csv)
        
        This scenario was noted with "TODONE: fix when padded out of bound # TODONE: crash fix".
        This was one of the critical crash fixes. Verifies algorithm stability.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'noisy_crash_7']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    @pytest.mark.core
    def test_temp_2_scenario(self):
        """
        Test that algorithm handles temp_2 scenario without hanging.
        
        Reference: test.py line 327 (temp_2.csv)
        
        This scenario was noted with "TODONE: fix hangup".
        This test ensures the algorithm completes in reasonable time without hanging.
        Before it would enter an infinite loop due to incorrect abrupt shaving logic. 
        Marking core as this test was pretty severe for this
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'temp_2']].copy()
        test_df.columns = ['date', 'value']

        start_time = time.perf_counter()
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )

        elapsed_seconds = time.perf_counter() - start_time
        
        assert len(results.segments) > 0
        assert elapsed_seconds < 4.0, (
            f"temp_2 scenario timed out: {elapsed_seconds:.3f}s (threshold: 4.0s)"
        )


    def test_temp_scenario(self):
        """
        Test that algorithm handles temp scenario without crashing.
        
        Additional crash scenario preserved in test data.
        Verifies algorithm stability on this edge case.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        test_df = crashes_df[['date', 'temp']].copy()
        test_df.columns = ['date', 'value']
        
        results = pt.detect_trends(
            test_df,
            date_col='date',
            value_col='value',
            plot=False,
            method_params=dict(is_abrupt_padded=True)
        )
        
        assert len(results.segments) > 0


    def test_all_crash_scenarios_batch(self):
        """
        Test that all crash scenarios can be processed in sequence without issues.
        
        This test verifies that processing multiple crash scenarios in succession
        does not cause any state-related issues or cumulative errors.
        """
        crashes_df = pd.read_csv('tests/data/noisy_crashes.csv')
        crash_columns = [col for col in crashes_df.columns if col not in ['Unnamed: 0', 'date']]
        
        for col in crash_columns:
            test_df = crashes_df[['date', col]].copy()
            test_df.columns = ['date', 'value']
            
            # Each scenario should complete without crashing
            results = pt.detect_trends(
                test_df,
                date_col='date',
                value_col='value',
                plot=False,
                method_params=dict(is_abrupt_padded=True)
            )
            
            assert results is not None, f"Failed on column {col}"
            assert len(results.segments) > 0, f"No segments detected for {col}"
