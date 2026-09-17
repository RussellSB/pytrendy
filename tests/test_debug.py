"""
Tests for debug functionality.

These tests verify that the debug process works as expected.
"""

import pytest
import pytrendy as pt
import matplotlib
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
        # Force a GUI-capable backend so the guard in _show_plot() calls plt.show();
        # the suite otherwise runs on Agg, where show() is a no-op by design.
        monkeypatch.setattr(matplotlib, 'get_backend', lambda: 'tkagg')

        df = pt.load_data('series_synthetic')
        _ = pt.detect_trends(
            df,
            date_col='date',
            value_col='gradual',
            plot=False,
            debug=True
        )
        assert len(show_calls) == 7