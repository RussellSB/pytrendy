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
        in the setting that user doesnt care about noise for treatment signal.

        With avoid_noise=False, the pipeline skips noise detection entirely and
        works on the raw signal, so trends are detected over / through the spikes.

        Granted, a bit of a mad man example, not sure why anyone would reasonably want this.
        But good to test that the worst case scenario is handled as expected when specified.

        Reference: test.py spike test 1.7, Modified instance with avoid_noise=False
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
        # With avoid_noise=False the pipeline works on the raw signal (spikes not masked
        # in value_cleaned), so the first Up starts later than the pre-fix expectation.
        expected_segments = [
            {'direction': 'Up', 'start': '2025-02-10', 'end': '2025-02-28'},
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
        With avoid_noise=False, noise detection is skipped entirely so the pipeline works on the raw signal.
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
        expected_segments = [ # raw signal used; zero baseline not masked so Up ends earlier
            {'direction': 'Up', 'start': '2025-02-27', 'end': '2025-03-29'},
            {'direction': 'Down', 'start': '2025-05-02', 'end': '2025-06-02'},
        ]
        assert_segments_in_a_haystack(results.segments, expected_segments)
