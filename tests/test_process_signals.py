"""
Tests for process_signals's requirement that the input DataFrame use an integer index.

pytrendy's internal pipeline always converts the caller-facing index (dates, floats,
strings, etc.) into a positional integer index before calling `process_signals`, so
downstream arithmetic (previously date-based, now purely integer-based) is valid.
`process_signals` asserts this precondition explicitly.
"""

import pytest
import pandas as pd
import numpy as np
from pytrendy.process_signals import process_signals


class TestProcessSignalsIndexAssertion:
    """Tests for the integer-index precondition added to process_signals."""

    def test_datetime_index_raises_assertion_error(self):
        """A DatetimeIndex is not an integer index and should be rejected immediately."""
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        df = pd.DataFrame({'value': np.arange(30, dtype=float)}, index=dates)
        method_params = {'avoid_noise': True}

        with pytest.raises(AssertionError, match="Supplied Index has type"):
            process_signals(df, 'value', method_params)

    def test_float_index_raises_assertion_error(self):
        """A float index should also be rejected."""
        df = pd.DataFrame(
            {'value': np.arange(30, dtype=float)},
            index=np.linspace(0, 1, 30)
        )
        method_params = {'avoid_noise': True}

        with pytest.raises(AssertionError, match="Supplied Index has type"):
            process_signals(df, 'value', method_params)

    def test_integer_index_passes_assertion(self):
        """A plain integer (RangeIndex) should satisfy the precondition and proceed normally."""
        df = pd.DataFrame(
            {'value': np.linspace(0, 30, 30)},
            index=pd.RangeIndex(30)
        )
        method_params = {'avoid_noise': True}

        result = process_signals(df, 'value', method_params)

        assert 'trend_flag' in result.columns
        assert 'noise_flag' in result.columns
        assert 'flat_flag' in result.columns