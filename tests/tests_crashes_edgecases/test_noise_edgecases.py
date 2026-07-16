"""
Tests for noise edge case scenarios in trend detection algorithm.

These tests verify that the trend detection algorithm handles gradual
inputs degraded with subtle levels of random noise, such that it becomes
most challenging to distinguish between true trends and noise. 
The scenarios are designed to be particularly difficult for the algorithm, 
and are based on real-world cases where noise can obscure underlying trends. 

Reference: tests/tests_crashes_edgecases/data/TESTDATA.md - noisy_edgecases.csv description
"""

import pandas as pd
import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestNoiseEdgeCases:
    """Test cases for noise scenarios that cause edge case behaviour in trend detection."""

    def test_noisy_edgecase_1_scenario(self):
        """Test that algorithm handles noisy_edgecase_1 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_1',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Flat', 'start': '2025-04-29', 'end': '2025-05-03'},
            {'direction': 'Noise', 'start': '2025-05-24', 'end': '2025-06-25'}, #TODO: double check, mock test.py shows 05-23 instead of 05-24
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_2_scenario(self):
        """Test that algorithm handles noisy_edgecase_2 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_2',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Noise', 'start': '2025-01-01', 'end': '2025-02-08'},
            {'direction': 'Noise', 'start': '2025-05-15', 'end': '2025-06-29'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_3_scenario(self):
        """Test that algorithm handles noisy_edgecase_3 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_3',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-20'},
            {'direction': 'Flat', 'start': '2025-06-21', 'end': '2025-06-30'}, 
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_4_scenario(self):
        """Test that algorithm handles noisy_edgecase_4 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_4',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-11', 'end': '2025-04-26'},
            {'direction': 'Flat', 'start': '2025-04-27', 'end': '2025-05-16'}, 
            {'direction': 'Down', 'start': '2025-05-17', 'end': '2025-06-12'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_5_scenario(self):
        """Test that algorithm handles noisy_edgecase_5 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_5',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-04'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_6_scenario(self):
        """Test that algorithm handles noisy_edgecase_6 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_6',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Down', 'start': '2025-03-10', 'end': '2025-03-25'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_7_scenario(self):
        """Test that algorithm handles noisy_edgecase_7 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_7',
            plot=False
        )

        expected_segments = [ 
            {'direction': 'Noise', 'start': '2025-03-31', 'end': '2025-05-09'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_8_scenario(self):
        """Test that algorithm handles noisy_edgecase_8 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_8',
            plot=False
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-14', 'end': '2025-05-09'},
            {'direction': 'Down', 'start': '2025-05-10', 'end': '2025-05-20'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_9_scenario(self):
        """Test that algorithm handles noisy_edgecase_9 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_9',
            plot=False
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-16', 'end': '2025-04-21'},
            {'direction': 'Flat', 'start': '2025-04-22', 'end': '2025-05-09'},
            {'direction': 'Down', 'start': '2025-05-10', 'end': '2025-05-25'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)

    def test_noisy_edgecase_10_scenario(self):
        """Test that algorithm handles noisy_edgecase_10 scenario reasonably."""
        df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='noisy_edgecase_10',
            plot=False,
            method_params=dict(abrupt_padding=28)
        )

        expected_segments = [ 
            {'direction': 'Up', 'start': '2025-04-15', 'end': '2025-05-03'},
            {'direction': 'Down', 'start': '2025-05-04', 'end': '2025-05-26'},
            {'direction': 'Noise', 'start': '2025-05-27', 'end': '2025-06-12'}
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    def test_noisy_edgecase_11_scenario(self):
            """
            Test that algorithm handles noisy_edgecase_11 scenario reasonably.
            Helps with test coverage on clean artifacts has_overlap_prev, prev_is_flat condition
            """
            df = pd.read_csv('tests/tests_crashes_edgecases/data/noisy_edgecases.csv')
            results = pt.detect_trends(
                df,
                date_col='date',
                value_col='noisy_edgecase_11',
                plot=False,
                method_params=dict(abrupt_padding=28)
            )

            expected_segments = [ 
                {'direction': 'Up', 'start': '2025-04-13', 'end': '2025-05-08'},
                {'direction': 'Noise', 'start': '2025-06-06', 'end': '2025-06-29'}
            ]
            assert_segments_in_a_haystack(results.segments, expected_segments)
