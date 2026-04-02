"""
Tests for debug functionality.

These tests verify that the debug process works as expected. 

This is work in progress.
"""

import pytest
import numpy as np
import pandas as pd
import pytrendy as pt
from conftest import assert_segments_match
import matplotlib.pyplot as plt


def compare_debug(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame, date_col: str='date', value_col: str='gradual') -> tuple[pt.PyTrendyResults, pt.PyTrendyResults]:
    """Function that exists exclusively to test detect_trends in an identical way. Runs detect trends with supplied dataframe in both debug and normal mode and returns the results for later comparison.

    Args:
        monkeypatch (pytest.MonkeyPatch): 
            Monkeypatch for suppressing plotting.
        df (pd.DataFrame):
            Pandas dataframe containing the trend we are testing. Must contain date_col and value_col.
        date_col (str, optional): 
            The date column.
            Defaults to 'date'.
        value_col (str, optional): 
            The value column.
            Defaults to 'gradual'.

    Returns:
        tuple[pt.PyTrendyResults, pt.PyTrendyResults]: The Pytrendy results with and without debug mode activated, respectively.
    """

    show_calls = []
    def fake_show(*args, **kwargs):
        show_calls.append((args, kwargs))
    monkeypatch.setattr(plt, 'show', fake_show)

    results_debug = pt.detect_trends(
        df,
        date_col='date',
        value_col='gradual',
        plot=False,
        debug=True,
        method_params=dict(is_abrupt_padded=False)
    )
    
    results_no_debug = pt.detect_trends(
        df,
        date_col='date',
        value_col='gradual',
        plot=False,
        debug=False,
        method_params=dict(is_abrupt_padded=False)
    )

    return results_debug, results_no_debug


class TestDebug:
    """Test cases for data loader functionality."""

    @pytest.mark.core
    def test_debug_mode_equivalency(self, monkeypatch):
        """Test that the series series_synthetic data produces identical outputs when in debug mode vs when not in debug mode."""
        df = pt.load_data('series_synthetic')
        results_debug, results_no_debug = compare_debug(monkeypatch, df)
        assert_segments_match(results_debug.segments, results_no_debug.segments)

    @pytest.mark.core
    def test_debug_mode_equivalency_noise(self, monkeypatch):
        """Test that the series series_synthetic data produces identical outputs when in debug mode vs when not in debug mode."""
        df = pt.load_data('series_synthetic')
        np.random.seed(42) # Deterministic Testing
        df['gradual'] += np.random.normal(0, 10, size=len(df))
        results_debug, results_no_debug = compare_debug(monkeypatch, df)
        assert_segments_match(results_debug.segments, results_no_debug.segments)
        
# TODO Add more equivalency tests here