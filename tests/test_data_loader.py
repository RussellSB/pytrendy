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
        """Test that series_synthetic has expected columns."""
        df = pt.load_data('series_synthetic')
        assert 'date' in df.columns

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
        """Test that classes_signals has expected columns."""
        df = pt.load_data('classes_signals')
        assert 'date' in df.columns

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

    @pytest.mark.core
    def test_series_synthetic_column_count(self):
        """Test that series_synthetic has the expected number of columns."""
        df = pt.load_data('series_synthetic')
        assert len(df.columns) >= 3

    @pytest.mark.core
    def test_classes_signals_column_count(self):
        """Test that classes_signals has the expected number of columns."""
        df = pt.load_data('classes_signals')
        assert len(df.columns) >= 3
