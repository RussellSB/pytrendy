"""Tests for detect_trends index handling and non-date lookup types."""
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

class TestIndexTypeRouting:
    """detect_trends entry-point routing across index types."""

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
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
            {'direction': 'Flat', 'start': '2025-06-18', 'end': '2025-06-30'},
        ])
        plt.close('all')

    def test_plot_true_integer_index(self):
        """Lines 165-167: plot=True path with integer index."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ])
        plt.close('all')

    def test_plot_true_float_index(self):
        """Lines 165-167: plot=True path with float index."""
        df = pt.load_data('series_synthetic')
        df['float_col'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(df, value_col='gradual', date_col='float_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': df['float_col'].iloc[1], 'end': df['float_col'].iloc[23]},
            {'direction': 'Flat', 'start': df['float_col'].iloc[168], 'end': df['float_col'].iloc[180]},
        ])
        plt.close('all')

    def test_plot_true_string_index(self):
        """Lines 165-167: plot=True path with string index."""
        df = pt.load_data('series_synthetic')
        df['str_col'] = [f'Step {i}' for i in range(len(df))]
        results = pt.detect_trends(df, value_col='gradual', date_col='str_col',
                                   plot=True, method_params={'abrupt_padding': 0})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 'Step 1', 'end': 'Step 23'},
            {'direction': 'Flat', 'start': 'Step 168', 'end': 'Step 180'},
        ])
        plt.close('all')

    def test_plot_true_with_plot_params(self):
        """Lines 165-167: plot=True with plot_params passed through."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, value_col='gradual', plot=True,
                                   method_params={'abrupt_padding': 0},
                                   plot_params={'title': 'Test Plot'})
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': 1, 'end': 23},
            {'direction': 'Flat', 'start': 168, 'end': 180},
        ])
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


class TestLegacyPositionalOrder:
    """detect_trends rejects the deprecated (date_col, value_col) positional order."""

    def test_old_positional_order_raises(self):
        """Passing (date_col, value_col) positionally raises a TypeError."""
        df = pt.load_data('series_synthetic')
        with pytest.raises(TypeError, match="value_col first"):
            pt.detect_trends(df, 'date', 'gradual', plot=False)

    def test_new_positional_order_ok(self):
        """Passing (value_col, date_col) positionally works without error."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(df, 'gradual', 'date', plot=False)
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
        ])


class TestIndexSorting:
    """detect_trends sorts unsorted sortable indexes before detection."""

    def test_unsorted_date_column(self):
        """Shuffled date column produces the same segments as sorted input."""
        df = pt.load_data('series_synthetic')
        expected = pt.detect_trends(df, value_col='gradual', date_col='date',
                                    plot=False, method_params={'abrupt_padding': 0})
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        actual = pt.detect_trends(shuffled, value_col='gradual', date_col='date',
                                  plot=False, method_params={'abrupt_padding': 0})
        assert_segments_match(actual.segments, expected.segments)

    def test_unsorted_integer_index(self):
        """Scrambled integer index (date_col=None) sorts back to original order."""
        df = pt.load_data('series_synthetic')
        expected = pt.detect_trends(df, value_col='gradual', plot=False,
                                    method_params={'abrupt_padding': 0})
        perm = np.random.RandomState(42).permutation(len(df))
        scrambled = df.iloc[perm].copy()
        scrambled.index = perm  # each row labelled by its original position
        actual = pt.detect_trends(scrambled, value_col='gradual', plot=False,
                                  method_params={'abrupt_padding': 0})
        assert_segments_match(actual.segments, expected.segments)

    def test_datetime_index_fallback(self):
        """date_col=None honours a DatetimeIndex, returning Timestamp boundaries."""
        df = pt.load_data('series_synthetic')
        df = df.set_index(pd.date_range('2025-01-01', periods=len(df), freq='D'))
        results = pt.detect_trends(df, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        assert results.index_type == 'datetime64'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': pd.Timestamp('2025-01-02'), 'end': pd.Timestamp('2025-01-24')},
        ])

    def test_float_index_nan_warns(self):
        """Float index containing NaN raises a UserWarning."""
        df = pt.load_data('series_synthetic')
        df['float_nan'] = np.linspace(0, 1, len(df))
        df.loc[5, 'float_nan'] = np.nan
        with pytest.warns(UserWarning, match="NaN"):
            pt.detect_trends(df, value_col='gradual', date_col='float_nan',
                             plot=False, method_params={'abrupt_padding': 0})

    def test_string_date_index_fallback(self):
        """date_col=None with a string-date index sorts chronologically and keeps labels."""
        df = pt.load_data('series_synthetic')
        df.index = [d.strftime('%Y-%m-%d') for d in pd.date_range('2025-01-01', periods=len(df), freq='D')]
        scrambled = df.iloc[np.random.RandomState(7).permutation(len(df))]
        results = pt.detect_trends(scrambled, value_col='gradual', plot=False,
                                   method_params={'abrupt_padding': 0})
        assert results.index_type == 'date'
        assert_segments_in_a_haystack(results.segments, [
            {'direction': 'Up', 'start': '2025-01-02', 'end': '2025-01-24'},
        ])
