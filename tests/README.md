# PyTrendy Test Suite

This directory contains the automated test suite for PyTrendy's trend detection functionality.

## Test Files

The tests are organized by functionality based on different aspects of trend detection:

### test_abrupts_and_spikes.py
Tests for abrupt trend detection and spike handling. Verifies:
- Detection of abrupt changes without padding
- Handling of single and multiple spikes
- Correct identification of abrupt segments vs noise
- Validation of segment boundaries and dates

### test_original_cases.py
Baseline tests for core functionality using the synthetic dataset. Validates:
- Gradual trend detection
- Abrupt trend detection (with and without padding)
- Segment properties (direction, days, change_rank, SNR)
- Chronological ordering of segments
- Classification of trend types (gradual vs abrupt)

### test_random_noise.py
Tests for robustness to random noise. Ensures:
- Detection works at various noise levels (0, 10, 15, 20, 50 std dev)
- Algorithm stability with high noise
- Consistent results across multiple runs
- Comparison between clean and noisy versions
- No crashes with extreme noise scenarios

### test_graduals_and_spikes.py
Tests for gradual trends with spike noise. Verifies:
- Detection of gradual trends with spikes at various positions
- Handling of single and multiple spikes
- Spikes at different magnitudes
- Consistency of detection across runs
- Proper continuation of trend detection after spikes

### test_crash_instances.py
Tests for previously problematic edge cases. Ensures:
- No crashes with extreme noise scenarios
- Handling of complex overlapping patterns
- Rapid transitions between levels
- Large value jumps and spikes
- Minimal/flat data scenarios
- Proper error handling and graceful degradation

## Running Tests

### Run all tests:
```bash
pytest tests/
```

### Run specific test file:
```bash
pytest tests/test_abrupts_and_spikes.py
```

### Run with verbose output:
```bash
pytest tests/ -v
```

### Run specific test:
```bash
pytest tests/test_original_cases.py::TestOriginalCases::test_gradual_trends
```

## Test Structure

All tests follow these patterns:
- Use pytest fixtures for common test data
- Include descriptive docstrings
- Make assertions on:
  - Number and types of detected segments
  - Segment start and end dates
  - Segment directions (Up/Down/Flat/Noise)
  - Temporal ordering and validity
  - Properties like days, change_rank, SNR
- Validate edge cases and error conditions

## Expected Results

The tests verify that PyTrendy:
1. Detects trends correctly in various scenarios
2. Handles noise robustly
3. Provides valid segment boundaries
4. Classifies trends appropriately
5. Doesn't crash on edge cases
6. Maintains consistency across runs
