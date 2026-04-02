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


class TestDebug:
    """Test cases for data loader functionality."""

    @pytest.mark.core
    def test_debug_mode_equivalency(self, monkeypatch):
        """Test that the series series_synthetic data produces identical outputs when in debug mode vs when not in debug mode."""

        show_calls = []
        def fake_show(*args, **kwargs):
            show_calls.append((args, kwargs))
        monkeypatch.setattr(plt, 'show', fake_show)

        df = pt.load_data('series_synthetic')
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
        
        assert_segments_match(results_debug.segments, results_no_debug.segments)

    @pytest.mark.core
    def test_debug_mode_equivalency_noise(self, monkeypatch):
        """Test that the series series_synthetic data produces identical outputs when in debug mode vs when not in debug mode."""

        show_calls = []
        def fake_show(*args, **kwargs):
            show_calls.append((args, kwargs))
        monkeypatch.setattr(plt, 'show', fake_show)

        noise_std = 10
        np.random.seed(42)

        df = pt.load_data('series_synthetic')
        df['gradual'] += np.random.normal(0, noise_std, size=len(df))

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
        
        assert_segments_match(results_debug.segments, results_no_debug.segments)


# TODO Add more equivalency tests here