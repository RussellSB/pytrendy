"""
TODO Add description here
"""
import pytest
import pytrendy as pt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from conftest import assert_segments_match, assert_segments_in_a_haystack


class TestNonDateCases:
    """Test cases where non-date indexes are used"""

    @pytest.mark.core
    def test_integer_index(self):
        """Test standard gradual trend but with no date index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            value_col='gradual',
            plot=False,
            method_params={'abrupt_padding': 0}
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Down', 'start': 24, 'end': 35},
            {'direction': 'Flat', 'start': 36, 'end': 39},
            {'direction': 'Up', 'start': 40, 'end': 75},
            {'direction': 'Down', 'start': 76, 'end': 90},
            {'direction': 'Up', 'start': 91, 'end': 127},
            {'direction': 'Down', 'start': 128, 'end': 167},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ]

        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_float_index(self):
        """Test standard gradual trend but with float lookup."""
        df = pt.load_data('series_synthetic')
        df['float_lookup'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='float_lookup',
            plot=False,
            method_params={'abrupt_padding': 0}
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up',   'start': 0.005556, 'end': 0.127778},
            {'direction': 'Down', 'start': 0.133333, 'end': 0.194444},
            {'direction': 'Flat', 'start': 0.200000, 'end': 0.216667},
            {'direction': 'Up',   'start': 0.222222, 'end': 0.416667},
            {'direction': 'Down', 'start': 0.422222, 'end': 0.500000},
            {'direction': 'Up',   'start': 0.505556, 'end': 0.705556},
            {'direction': 'Down', 'start': 0.711111, 'end': 0.927778},
            {'direction': 'Flat', 'start': 0.933333, 'end': 1.000000},
        ]

        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_string_index(self):
        """Test standard gradual trend but with string lookup."""
        df = pt.load_data('series_synthetic')
        df['string_lookup'] = [f"Step {i}" for i in range(len(df))]
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='string_lookup',
            plot=False,
            method_params={'abrupt_padding': 0}
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up',   'start': 'Step 1',   'end': 'Step 23'},
            {'direction': 'Down', 'start': 'Step 24',  'end': 'Step 35'},
            {'direction': 'Flat', 'start': 'Step 36',  'end': 'Step 39'},
            {'direction': 'Up',   'start': 'Step 40',  'end': 'Step 75'},
            {'direction': 'Down', 'start': 'Step 76',  'end': 'Step 90'},
            {'direction': 'Up',   'start': 'Step 91',  'end': 'Step 127'},
            {'direction': 'Down', 'start': 'Step 128', 'end': 'Step 167'},
            {'direction': 'Flat', 'start': 'Step 168', 'end': 'Step 180'},
        ]

        assert_segments_match(results.segments, expected_segments)

    @pytest.mark.core
    def test_weekly_date_index(self):
        """Test standard gradual trend with weekly-spaced dates."""
        df = pt.load_data('series_synthetic')
        # Create weekly dates starting from 2026-01-01
        df['weekly_date'] = pd.date_range(start='2026-01-01', periods=len(df), freq='W')
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='weekly_date',
            plot=False,
            method_params={'abrupt_padding': 0}
        )
        
        # Expected segments based on current behavior
        expected_segments = [
            {'direction': 'Up',   'start': pd.Timestamp('2026-01-11'),  'end': pd.Timestamp('2026-06-14')},
            {'direction': 'Down', 'start': pd.Timestamp('2026-06-21'),  'end': pd.Timestamp('2026-09-06')},
            {'direction': 'Flat', 'start': pd.Timestamp('2026-09-13'),  'end': pd.Timestamp('2026-10-04')},
            {'direction': 'Up',   'start': pd.Timestamp('2026-10-11'),  'end': pd.Timestamp('2027-06-13')},
            {'direction': 'Down', 'start': pd.Timestamp('2027-06-20'),  'end': pd.Timestamp('2027-09-26')},
            {'direction': 'Up',   'start': pd.Timestamp('2027-10-03'),  'end': pd.Timestamp('2028-06-11')},
            {'direction': 'Down', 'start': pd.Timestamp('2028-06-18'),  'end': pd.Timestamp('2029-03-18')},
            {'direction': 'Flat', 'start': pd.Timestamp('2029-03-25'),  'end': pd.Timestamp('2029-06-17')},
        ]

        assert_segments_match(results.segments, expected_segments)

class TestDetectTrendsCoverage:
    """Test detect_trends uncovered paths across index types."""

    def test_not_implemented_dtype(self):
        """Line 44: unimplemented dtype raises NotImplementedError."""
        df = pt.load_data('series_synthetic')
        df['bool_col'] = True
        with pytest.raises(NotImplementedError, match="unimplemented dtype"):
            pt.detect_trends(df, value_col='gradual', date_col='bool_col',
                             plot=False)

    def test_plot_true_date_index(self):
        """Lines 165-167: plot=True path with date index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', date_col='date',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_integer_index(self):
        """Lines 165-167: plot=True path with integer index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_float_index(self):
        """Lines 165-167: plot=True path with float index."""
        df = pt.load_data('series_synthetic')
        df['float_col'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_string_index(self):
        """Lines 165-167: plot=True path with string index."""
        df = pt.load_data('series_synthetic')
        df['str_col'] = [f'S{i}' for i in range(len(df))]
        results = pt.detect_trends(df, value_col='gradual', date_col='str_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert results is not None
        plt.close('all')

    def test_plot_true_with_plot_params(self):
        """Lines 165-167: plot=True with plot_params passed through."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0},
                                   plot_params={'title': 'Test Plot'})
        assert results is not None
        plt.close('all')


class TestDetectIndexTypeInteger:
    """Test detect_index_type with explicit integer date_col."""

    def test_int_date_col(self):
        """Line 40: passing an integer-typed column as date_col returns 'integer'."""
        df = pt.load_data('series_synthetic')
        df['int_col'] = np.arange(len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='int_col',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ])


class TestPrevFill:
    """Exercise the prev fill branch in plot_pytrendy when start displacement is invalid."""

    def test_string_prev_not_trend_invalid_displacement(self):
        """Lines 172-176, 182: string index, prev is Flat neighbouring, start displacement invalid."""
        df = pd.DataFrame(
            {'date': [f'S{i}' for i in range(40)],
             'value': [90 + i for i in range(10)] + [100] * 10 + [80 - i for i in range(5)] + [60 + i for i in range(15)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'string'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 'S0', 'end': 'S18'},
        ])

    def test_integer_prev_not_trend_invalid_displacement(self):
        """Lines 177-178: integer index, prev Flat neighbouring, start displacement invalid."""
        df = pd.DataFrame(
            {'value': [90 + i for i in range(10)] + [100] * 10 + [80 - i for i in range(5)] + [60 + i for i in range(15)]})
        results = pt.detect_trends(df, value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Flat', 'start': 0, 'end': 18},
        ])


class TestNextNoiseFill:
    """Exercise the next-noise fill branch in plot_pytrendy when end displacement is invalid."""

    def test_string_next_noise_invalid_displacement(self):
        """Lines 212-214: string index, next Noise adjacent, end displacement invalid."""
        df = pd.DataFrame(
            {'date': [f'S{i}' for i in range(40)],
             'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'string'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 'S1', 'end': 'S17'},
        ])

    def test_integer_next_noise_invalid_displacement(self):
        """Lines 215-216: integer index, next Noise adjacent, end displacement invalid."""
        df = pd.DataFrame(
            {'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'integer'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': 1, 'end': 17},
        ])

    def test_date_next_noise_invalid_displacement(self):
        """Line 211: date index, next Noise adjacent, end displacement invalid."""
        df = pd.DataFrame(
            {'date': pd.date_range('2025-01-01', periods=40, freq='D'),
             'value': [200 - i for i in range(20)] + [200 + i for i in range(20)]})
        results = pt.detect_trends(df, date_col='date', value_col='value',
                                   plot=False, method_params={'abrupt_padding': 0})
        assert results.index_type == 'datetime64'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Down', 'start': pd.Timestamp('2025-01-02'), 'end': pd.Timestamp('2025-01-18')},
        ])
