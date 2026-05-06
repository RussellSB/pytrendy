"""
Tests for debug functionality.

These tests verify that the debug process works as expected.
"""

import pytest
import pytrendy as pt
import matplotlib.pyplot as plt


class TestDebug:
    """Test cases for debug functionality."""

    @pytest.mark.core
    def test_debug_mode_plots(self, monkeypatch):
        """
        Test that the correct number of plots are created.
        We use monkeypatch to replace plt.show with a fake function that records calls.
        """
        
        show_calls = []
        def fake_show(*args, **kwargs):
            show_calls.append((args, kwargs))
            plt.close("all")
        monkeypatch.setattr(plt, 'show', fake_show)

        df = pt.load_data('series_synthetic')
        _ = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            debug=True,
            method_params=dict(is_abrupt_padded=False)
        )
        assert len(show_calls) == 7