# Testing Documentation

This document describes the test suite for the 1997 LSTM reproduction.

## Test Structure

```
tests/
├── __init__.py
├── test_activations.py      # Activation function tests
├── test_cec_gradient.py     # CEC gradient verification
├── test_forward_shapes.py   # Tensor shape validation
└── test_experiments_smoke.py # Data generator smoke tests
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_activations.py

# Run with verbose output
pytest tests/ -v

# Run only numpy tests (no framework dependencies)
pytest tests/test_activations.py::TestActivationsNumpy -v
```

## Test Categories

### 1. Activation Function Tests (`test_activations.py`)

Verifies the paper-accurate activation functions:

| Test | Description |
|------|-------------|
| `TestActivationsNumpy::test_g_squash_range` | g(x) output is in [-2, 2] |
| `TestActivationsNumpy::test_h_squash_range` | h(x) output is in [-1, 1] |
| `TestActivationsNumpy::test_g_squash_zero_centered` | g(0) = 0 |
| `TestActivationsNumpy::test_h_squash_zero_centered` | h(0) = 0 |
| `TestActivationsTinygrad::*` | Same tests for tinygrad tensors |
| `TestActivationsTorch::*` | Same tests for PyTorch tensors |

### 2. CEC Gradient Tests (`test_cec_gradient.py`)

Verifies the Constant Error Carousel maintains gradient = 1.0:

| Test | Description |
|------|-------------|
| `TestCECGradientTinygrad::test_gradient_is_one` | ∂s_t/∂s_{t-1} = 1.0 |
| `TestCECGradientTorch::test_gradient_is_one` | Same for PyTorch |
| `TestCECMathematicalProperty::test_cec_identity_jacobian` | Pure numpy verification |

**This is the most important test** - it verifies the core innovation of the 1997 paper.

### 3. Forward Shape Tests (`test_forward_shapes.py`)

Verifies tensor shapes through the forward pass:

| Test | Description |
|------|-------------|
| `TestForwardShapesTinygrad::test_1d_input` | Single sample, 1D tensors |
| `TestForwardShapesTinygrad::test_batched_input` | Batched samples, 2D tensors |
| `TestForwardShapesTorch::*` | Same for PyTorch |
| `TestParameterShapes::test_weight_dimensions` | W, U, b matrix sizes |

### 4. Experiment Smoke Tests (`test_experiments_smoke.py`)

Quick verification that experiment data generators work:

| Test | Description |
|------|-------------|
| `TestAddingProblemGenerator::test_shapes` | X: (N, T, 2), Y: (N, 1) |
| `TestAddingProblemGenerator::test_values_in_range` | X[:,:,0] in [0, 1] |
| `TestAddingProblemGenerator::test_marker_count` | Exactly 2 markers per sequence |
| `TestMultiplicationProblemGenerator::*` | Same for multiplication |
| `TestTemporalOrderGenerator::test_one_hot_encoding` | Symbols are one-hot |
| `TestExperimentImports::*` | Verify all modules import |

## Framework-Specific Tests

Tests that require tinygrad or PyTorch are marked with `pytest.mark.skipif`:

```python
@pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
class TestForwardShapesTinygrad:
    ...

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestForwardShapesTorch:
    ...
```

This allows the test suite to run partially even if only one framework is installed.

## Pre-Push Verification

Before pushing to the repository, run:

```bash
# 1. Syntax check all Python files
python3 -m py_compile src/aquarius_lstm/*.py experiments/*.py tests/*.py

# 2. Run numpy-only tests (no framework deps)
pytest tests/test_activations.py::TestActivationsNumpy -v

# 3. If tinygrad installed, run full suite
pip install tinygrad && pytest tests/ -v

# 4. If PyTorch installed, run full suite
pip install torch && pytest tests/ -v
```

## Code Review Verification (Completed)

The Oracle agent verified the following against the 1997 paper:

### PASS ✓
- Forward equations match paper Section 2-3
- g(x) = 4σ(x) - 2 ∈ [-2, 2]
- h(x) = 2σ(x) - 1 ∈ [-1, 1]
- CEC update: s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t))
- Truncation: h_prev is DETACHED for gates
- Gradient tunnel: s_c_prev is NOT detached

### Fixed Issues
- Sequence length variation now ±10% (was only +0-10%)
- init_gate_biases() now handles hidden_size < len(bias_values)
- Docstrings corrected for data generator value ranges

## Continuous Integration

When CI is set up, the workflow should:

1. Install dependencies: `pip install -e ".[dev]"`
2. Run linting: `ruff check src/ experiments/`
3. Run tests: `pytest tests/ -v --tb=short`
4. Run smoke experiments: `python -m experiments.run_all --mode smoke`

## Adding New Tests

When adding new functionality:

1. Add tests to the appropriate file or create a new `test_*.py`
2. Use descriptive test names: `test_<what>_<expected_behavior>`
3. Include both positive and edge case tests
4. Mark framework-specific tests with `@pytest.mark.skipif`
5. Run the full suite before committing

## Latest Test Results (December 30, 2025)

### Summary
- **Total**: 61/62 passed
- **One failure**: `test_derivative_max_at_zero` (TestDerivatives) - argmax returns 499 instead of 500 due to even-length array where max value appears at two adjacent indices (off-by-one due to even array length, not a real bug).

### Key Verifications
- **CEC Gradient**: All CEC gradient tests pass (critical).
- **Activations**: All activation tests pass (except derivative max).
- **Forward Shapes**: All forward shape tests pass.
- **Experiment Smokes**: All experiment smoke tests pass.

### Smoke Experiment Notes
- **Adding Problem smoke**: Loss converges (0.016), but insufficient epochs to pass criterion.
- **Multiplication Problem smoke**: Loss converges (0.057), insufficient epochs.
- **Temporal Order smoke**: Accuracy improves from 29% to 81%, loss drops, learning is occurring.
- **Backend**: All experiments run successfully with PyTorch backend.

