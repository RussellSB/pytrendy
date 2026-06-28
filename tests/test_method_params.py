"""
Tests for method_params validation in detect_trends.
"""

import pytest
import pytrendy as pt


class TestMethodParams:
    """Tests for method_params behaviour in detect_trends."""

    @pytest.mark.core
    def test_is_abrupt_padded_deprecation_warning(self):
        """Test that passing is_abrupt_padded in method_params raises a DeprecationWarning."""
        df = pt.load_data('series_synthetic')
        with pytest.warns(DeprecationWarning, match="is_abrupt_padded.*deprecated.*abrupt_padding"):
            pt.detect_trends(
                df,
                date_col='date',
                value_col='abrupt',
                plot=False,
                method_params=dict(is_abrupt_padded=True)
            )
