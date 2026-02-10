"""
Tests for data loader functionality.

These tests verify that the data loader correctly loads built-in datasets
and handles error cases appropriately.
"""

import pytest
import pandas as pd
import pytrendy as pt


class TestDataLoader:
    """Test cases for data loader functionality."""

    @pytest.mark.core
    def test_load_series_synthetic_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame for series_synthetic."""
        df = pt.load_data('series_synthetic')
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.core
    def test_load_series_synthetic_has_data(self):
        """Test that series_synthetic dataset is not empty."""
        df = pt.load_data('series_synthetic')
        assert not df.empty

    @pytest.mark.core
    def test_load_series_synthetic_has_expected_columns(self):
        """Test that series_synthetic has all expected columns."""
        df = pt.load_data('series_synthetic')
        expected_columns = ['date', 'abrupt', 'gradual', 'gradual-noisy-20']
        assert all(col in df.columns for col in expected_columns)

    @pytest.mark.core
    def test_load_classes_signals_returns_dataframe(self):
        """Test that load_data returns a pandas DataFrame for classes_signals."""
        df = pt.load_data('classes_signals')
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.core
    def test_load_classes_signals_has_data(self):
        """Test that classes_signals dataset is not empty."""
        df = pt.load_data('classes_signals')
        assert not df.empty

    @pytest.mark.core
    def test_load_classes_signals_has_expected_columns(self):
        """Test that classes_signals has all expected columns."""
        df = pt.load_data('classes_signals')
        expected_columns = ['date', 'gradual_up', 'gradual_down', 'abrupt_up', 'abrupt_down', 'noise_up', 'noise_down']
        assert all(col in df.columns for col in expected_columns)

    @pytest.mark.core
    def test_load_default_dataset(self):
        """Test that load_data without arguments loads series_synthetic by default."""
        df = pt.load_data()
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.core
    def test_load_invalid_dataset_raises_error(self):
        """Test that invalid dataset name raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            pt.load_data('invalid_dataset')
