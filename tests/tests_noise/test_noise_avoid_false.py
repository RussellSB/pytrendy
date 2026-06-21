"""
Tests for behaviour of trend detection when user toggles avoid_noise as False (default: True).
This satisfies sitations where users dont really care about noise in the treatment signal.
"""

import pytest
import pytrendy as pt
from conftest import assert_segments_in_a_haystack


class TestNoiseAvoidFalse:
    """Test cases for noise spike detection with gradual trends."""

    @pytest.mark.core
    def test_gradual_four_spikes_noise_avoid_false(self):
        """
        Test trends with four spikes distributed across the series,
        in the setting that user doesnt care about noise for treatment signal
        Verifies  that four spikes with varying values are correctly ignored and trends are detected over them.

        Granted, a bit of a mad man example, not sure why anyone would reasonably want this.
        But good to test that the worst case scenario is ignored as expected when specified.
        
        Reference: test.py spike test 1.7, Modified instance with avoid_noise=False
        
        This test 
        """
        # spike test 1.7 - add 4 spikes
        df = pt.load_data('series_synthetic')
        df.set_index('date', inplace=True)
        df.loc['2025-02-28':'2025-02-28', 'gradual'] = 125  # spike
        df.loc['2025-04-09':'2025-04-09', 'gradual'] = 150  # spike
        df.loc['2025-05-08':'2025-05-08', 'gradual'] = 300  # spike
        df.loc['2025-06-03':'2025-06-03', 'gradual'] = 320  # spike
        df = df.reset_index()
        
        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            method_params=dict(is_abrupt_padded=False
                               , avoid_noise=False # main parameter tested
                        )
        )
        
        # Expect no noise segments representing the four spikes
        noise_segments = results.filter_segments(direction='Noise', format='dict')
        assert len(noise_segments) == 0, 'Expected all 4 spikes to be ignored with avoid_noise=False'
        
        # Assert for trends overlapping spikes, now that avoid_noise=False
        # Expected trends ignorant of noise
        expected_segments = [
            {'direction': 'Up', 'start': '2025-02-06', 'end': '2025-02-28'},
            {'direction': 'Up', 'start': '2025-04-02', 'end': '2025-05-08'},
            {'direction': 'Down', 'start': '2025-05-09', 'end': '2025-06-17'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)


    @pytest.mark.core
    def test_abrupt_trends_with_padding_avoid_false(self):
        """
        Test detection of abrupt trends with padding enabled in scenario where avoid_noise=False.
        This is the main scenario that `avoid_noise=False` solves.

        When treatment is 0 pre/post activation, sometimes undesired noise segments are detected over the initial changepoints.
        This is probably due to noise refinement logic that needs further tweaking to avoid this (though this was already catered for).
        Quick fix for now however, is just to let the user specifiy if they're happy with ignoring noise.
        """
        df = pt.load_data('series_synthetic')

        # Setting 0 activity around dummy "spend activation"
        df.set_index('date', inplace=True)
        df.loc['2025-01-01':'2025-02-27', 'abrupt'] = 0
        df.loc['2025-05-05':'2025-06-30', 'abrupt'] = 0
        df = df.reset_index()

        results = pt.detect_trends(
            df,
            date_col='date',
            value_col='abrupt',
            plot=False,
            method_params=dict(
                                abrupt_padding=28
                                , avoid_noise=False # main parameter tested
                            )
        )
        expected_segments = [ # noise segments should be ignored, and no longer block precise trend detect for dummy "new market"
            {'direction': 'Up', 'start': '2025-02-27', 'end': '2025-04-26'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-06-02'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)
