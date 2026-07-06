"""
Tests for plot_pytrendy's new `index_type` parameter.

`plot_pytrendy` now accepts an `index_type` argument ("date", "integer", "float",
or "string") that governs how segment boundaries and neighbour adjacency are
computed. These tests are functional smoke tests (no image comparison baseline)
verifying that each supported index_type renders without error, that invalid
index types are rejected, and that the index type is announced via print.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytrendy.io.plot_pytrendy import plot_pytrendy


class TestPlotPytrendyIndexTypes:
    """Tests for plot_pytrendy support of non-date index types."""

    def test_invalid_index_type_raises(self):
        """An unsupported index_type should raise NotImplementedError before any plotting occurs."""
        dates = pd.date_range('2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({'value': np.arange(10, dtype=float)}, index=dates)
        segments = [{'direction': 'Up', 'start': dates[0], 'end': dates[-1]}]

        with pytest.raises(NotImplementedError, match="Index Type bogus not yet implemented"):
            plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                          index_type='bogus', suppress_show=True)

    def test_prints_index_type(self, capsys):
        """plot_pytrendy should print the index_type it was invoked with."""
        df = pd.DataFrame({'value': [0.0, 1.0, 2.0]}, index=pd.RangeIndex(3))
        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=[],
                             index_type='integer', suppress_show=True)
        plt.close(fig)

        captured = capsys.readouterr()
        assert 'internal index type integer' in captured.out

    def test_plot_with_integer_index_single_segment(self):
        """A single Up segment on an integer index should render without error."""
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=pd.RangeIndex(20))
        segments = [{'direction': 'Up', 'start': 5, 'end': 15, 'trend_class': 'gradual'}]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='integer', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_integer_index_neighbouring_segments(self):
        """Two touching same-direction segments should exercise the neighbour-adjustment
        and vertical-line-drawing logic paths for integer indices."""
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=pd.RangeIndex(20))
        segments = [
            {'direction': 'Up', 'start': 0, 'end': 9, 'trend_class': 'gradual', 'change_rank': 2},
            {'direction': 'Up', 'start': 10, 'end': 19, 'trend_class': 'gradual', 'change_rank': 1},
        ]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='integer', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_float_index(self):
        """A segment addressed by float labels should render without error."""
        index = np.linspace(0, 1, 20)
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=index)
        segments = [{'direction': 'Down', 'start': float(index[5]), 'end': float(index[15]),
                     'trend_class': 'gradual'}]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='float', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_string_index(self):
        """A segment addressed by string labels should render without error, including the
        change_rank annotation path (which is intentionally skipped for string indices)."""
        labels = [f"Step {i}" for i in range(20)]
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=labels)
        segments = [{'direction': 'Up', 'start': 'Step 5', 'end': 'Step 15',
                     'trend_class': 'gradual', 'change_rank': 1}]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='string', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_string_index_neighbouring_segments(self):
        """Two touching same-direction string-indexed segments should render without error."""
        labels = [f"Step {i}" for i in range(20)]
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=labels)
        segments = [
            {'direction': 'Down', 'start': 'Step 0', 'end': 'Step 9', 'trend_class': 'gradual'},
            {'direction': 'Down', 'start': 'Step 10', 'end': 'Step 19', 'trend_class': 'gradual'},
        ]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='string', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_noise_and_abrupt_segment_integer_index(self):
        """Noise and abrupt segments follow different start/end adjustment branches; verify
        they render without error on an integer index."""
        df = pd.DataFrame({'value': np.arange(20, dtype=float)}, index=pd.RangeIndex(20))
        segments = [
            {'direction': 'Noise', 'start': 3, 'end': 4},
            {'direction': 'Up', 'start': 5, 'end': 6, 'trend_class': 'abrupt'},
        ]

        fig = plot_pytrendy(df=df, value_col='value', segments_enhanced=segments,
                            index_type='integer', suppress_show=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)