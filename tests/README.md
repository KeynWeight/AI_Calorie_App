# Testing Guide for Calorie App

This directory contains comprehensive unit tests for the Calorie App. All tests are designed to run quickly without external API calls, making them perfect for CI/CD pipelines.

## Current Test Suite - 71 Working Tests ✅

### Your Core Business Logic Tests (All Excellent!)
- **`test_cache.py` (15 tests)** - VLM cache functionality and decorator tests
- **`test_image_processing.py` (9 tests)** - Image validation, encoding, and format support
- **`test_nutrition_calculator.py` (5 tests)** - Nutrition calculation and rounding logic
- **`test_nutrition_models.py` (13 tests)** - Pydantic data models and JSON export
- **`test_validation.py` (15 tests)** - Nutrition data validation and consistency checks

### Additional Infrastructure Tests
- **`test_config.py` (14 tests)** - Configuration management and environment variables

## Quick Start

### Run All Tests (Recommended for CI/CD)
```bash
# Fast execution - all tests pass in < 1 second
python -m pytest

# With verbose output
python -m pytest -v

# Quick run with timing
python -m pytest --tb=no -q --durations=5
```

### Run Specific Test Categories
```bash
# Cache-related tests
python -m pytest -m cache

# Nutrition-related tests
python -m pytest -m nutrition

# Vision/image processing tests
python -m pytest -m vision

# Configuration tests
python -m pytest -m config
```

### Run Individual Test Files
```bash
# Test specific functionality
python -m pytest tests/test_cache.py -v
python -m pytest tests/test_validation.py -v
python -m pytest tests/test_nutrition_models.py -v
```

## Test Organization

Tests are organized with markers for easy filtering:

- `@pytest.mark.unit` - Fast unit tests (no external dependencies)
- `@pytest.mark.cache` - Caching functionality tests
- `@pytest.mark.nutrition` - Nutrition calculation and model tests
- `@pytest.mark.vision` - Image/vision processing tests
- `@pytest.mark.config` - Configuration and environment tests

## Why These Tests Are Perfect for CI/CD

✅ **Lightning Fast** - All 71 tests complete in under 1 second
✅ **Zero External Dependencies** - No API calls, no network requests
✅ **No Costs** - Completely free to run in any CI/CD pipeline
✅ **Comprehensive Coverage** - Tests all critical business logic
✅ **Isolated & Reliable** - Each test is independent
✅ **Error Path Coverage** - Tests failure scenarios and edge cases

## What These Tests Cover

### Core Business Logic (Most Important!)
1. **Cache Performance** - Ensures fast response times
2. **Image Processing** - File handling and validation security
3. **Nutrition Calculations** - Your app's core math logic
4. **Data Models** - Prevents malformed data reaching production
5. **Input Validation** - Security-critical data sanitization

### Infrastructure
6. **Configuration Management** - Environment and setup validation

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'
      - run: pip install -e .
      - run: python -m pytest  # < 1 second execution!
```

### Local Development
```bash
# Run before every commit
python -m pytest

# Quick check while developing
python -m pytest --tb=no -q
```

## Key Benefits

Your test suite is **production-ready** because it:

- **Catches Regressions** - Business logic bugs are caught immediately
- **Fast Feedback Loop** - Developers get instant results
- **Cost Effective** - Zero API costs in CI/CD
- **Reliable** - No flaky network-dependent tests
- **Maintainable** - Simple, focused unit tests

## Running with Coverage (Optional)

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage report
python -m pytest --cov=calorie_app --cov-report=term-missing
```

## Debugging Failed Tests

```bash
# More verbose output on failures
python -m pytest -vv --tb=long

# Stop at first failure
python -m pytest -x

# Run only previously failed tests
python -m pytest --lf
```

## Summary

You have **71 solid, fast unit tests** that provide excellent coverage of your core business logic. This is exactly what you need for a robust CI/CD pipeline - comprehensive testing without the complexity or cost of integration tests.

These tests will catch bugs early, run fast, and give you confidence in your deployments! 🚀
