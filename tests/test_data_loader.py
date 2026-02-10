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
    def test_load_series_synthetic_has_date_column(self):
        """Test that series_synthetic has date column."""
        df = pt.load_data('series_synthetic')
        assert 'date' in df.columns

    @pytest.mark.core
    def test_load_series_synthetic_has_abrupt_column(self):
        """Test that series_synthetic has abrupt column."""
        df = pt.load_data('series_synthetic')
        assert 'abrupt' in df.columns

    @pytest.mark.core
    def test_load_series_synthetic_has_gradual_column(self):
        """Test that series_synthetic has gradual column."""
        df = pt.load_data('series_synthetic')
        assert 'gradual' in df.columns

    @pytest.mark.core
    def test_load_series_synthetic_has_gradual_noisy_column(self):
        """Test that series_synthetic has gradual-noisy-20 column."""
        df = pt.load_data('series_synthetic')
        assert 'gradual-noisy-20' in df.columns

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
    def test_load_classes_signals_has_date_column(self):
        """Test that classes_signals has date column."""
        df = pt.load_data('classes_signals')
        assert 'date' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_gradual_up_column(self):
        """Test that classes_signals has gradual_up column."""
        df = pt.load_data('classes_signals')
        assert 'gradual_up' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_gradual_down_column(self):
        """Test that classes_signals has gradual_down column."""
        df = pt.load_data('classes_signals')
        assert 'gradual_down' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_abrupt_up_column(self):
        """Test that classes_signals has abrupt_up column."""
        df = pt.load_data('classes_signals')
        assert 'abrupt_up' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_abrupt_down_column(self):
        """Test that classes_signals has abrupt_down column."""
        df = pt.load_data('classes_signals')
        assert 'abrupt_down' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_noise_up_column(self):
        """Test that classes_signals has noise_up column."""
        df = pt.load_data('classes_signals')
        assert 'noise_up' in df.columns

    @pytest.mark.core
    def test_load_classes_signals_has_noise_down_column(self):
        """Test that classes_signals has noise_down column."""
        df = pt.load_data('classes_signals')
        assert 'noise_down' in df.columns

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
