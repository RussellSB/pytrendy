"""
Tests for detect_trends's date_col dtype resolution logic.

`detect_trends` now inspects the dtype of `date_col` (or generates an implicit
integer index when `date_col` is None) to decide how segment boundaries should
be represented downstream: as dates, integers, floats, string labels, or
pre-parsed datetime64 values. These tests exercise each of those resolution
branches, including the fallback and error paths.
"""

import pytest
import pandas as pd
import numpy as np
import pytrendy as pt


class TestIndexDtypeResolution:
    """Tests for detect_trends's automatic detection of date_col dtype / index_type."""

    @pytest.mark.core
    def test_default_date_col_none_produces_integer_index(self):
        """When date_col is not supplied, an implicit integer index should be used."""
        df = pt.load_data('series_synthetic')
        results = pt.detect_trends(
            df,
            value_col='gradual',
            plot=False,
        )

        assert results.index_type == 'integer'
        assert all(isinstance(seg['start'], (int, np.integer)) for seg in results.segments)
        assert all(isinstance(seg['end'], (int, np.integer)) for seg in results.segments)

    @pytest.mark.core
    def test_explicit_integer_date_col_matches_implicit_index(self):
        """Passing an explicit integer date_col should behave the same as date_col=None."""
        df = pt.load_data('series_synthetic')
        df['int_index'] = np.arange(len(df))
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='int_index',
            plot=False,
        )

        assert results.index_type == 'integer'
        assert results.segments[0]['direction'] == 'Up'
        assert results.segments[0]['start'] == 1
        assert results.segments[0]['end'] == 23

    @pytest.mark.core
    def test_datetime64_column_sets_datetime64_index_type(self):
        """A date_col already parsed to datetime64 should be classified as 'datetime64',
        distinct from the 'date' index_type used for parseable string columns.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='date',
            plot=False,
        )

        assert results.index_type == 'datetime64'
        # Underlying detection is unaffected: boundaries match the date-string variant.
        assert results.segments[0]['direction'] == 'Up'
        assert pd.Timestamp(results.segments[0]['start']) == pd.Timestamp('2025-01-02')
        assert pd.Timestamp(results.segments[0]['end']) == pd.Timestamp('2025-01-24')

    @pytest.mark.core
    def test_datetime64_column_with_plot_raises_not_implemented(self):
        """plot_pytrendy does not yet accept the 'datetime64' index_type, so requesting a
        plot with a pre-parsed datetime64 date_col should surface a clear error rather
        than silently mis-rendering.
        """
        df = pt.load_data('series_synthetic')
        df['date'] = pd.to_datetime(df['date'])

        with pytest.raises(NotImplementedError, match="Index Type datetime64 not yet implemented"):
            pt.detect_trends(
                df,
                value_col='gradual',
                date_col='date',
                plot=True,
            )

    @pytest.mark.core
    def test_partially_parseable_string_falls_back_to_string_index(self, capsys):
        """If not every value in a string date_col parses to a date, pytrendy should fall
        back to treating the column as a string lookup rather than raising.
        """
        df = pt.load_data('series_synthetic')
        df.loc[0, 'date'] = 'not-a-date'  # breaks full-column date parsing

        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='date',
            plot=False,
        )

        assert results.index_type == 'string'
        captured = capsys.readouterr()
        assert 'treating as string lookup' in captured.out

    @pytest.mark.core
    def test_unsupported_dtype_raises_not_implemented_error(self):
        """An unsupported dtype for date_col (e.g. boolean) should raise NotImplementedError."""
        df = pt.load_data('series_synthetic')
        df['flag_col'] = [i % 2 == 0 for i in range(len(df))]

        with pytest.raises(NotImplementedError, match="unimplimented dtype"):
            pt.detect_trends(
                df,
                value_col='gradual',
                date_col='flag_col',
                plot=False,
            )

    @pytest.mark.core
    def test_float_date_col_sets_float_index_type(self):
        """A float date_col should be classified as the 'float' index_type."""
        df = pt.load_data('series_synthetic')
        df['float_lookup'] = np.linspace(0, 1, len(df))
        results = pt.detect_trends(
            df,
            value_col='gradual',
            date_col='float_lookup',
            plot=False,
        )

        assert results.index_type == 'float'
        assert all(isinstance(seg['start'], float) for seg in results.segments)