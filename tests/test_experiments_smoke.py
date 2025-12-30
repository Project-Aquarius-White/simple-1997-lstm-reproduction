"""
Smoke tests for all experiments.

These tests verify that each experiment's data generator works correctly
without running the full training loop.

Reference: experiments/ directory
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAddingProblemGenerator:
    """Smoke tests for Adding Problem data generator."""

    def test_generate_adding_data_shapes(self):
        """Test output shapes of adding data generator."""
        from experiments.adding import generate_adding_data
        
        seq_len = 50
        num_samples = 10
        
        X, Y = generate_adding_data(seq_len=seq_len, num_samples=num_samples, seed=42)
        
        # X should be (num_samples, max_len, 2) where max_len ~= seq_len * 1.1
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[0] == num_samples, f"Expected {num_samples} samples, got {X.shape[0]}"
        assert X.shape[2] == 2, f"Expected 2 input channels, got {X.shape[2]}"
        assert X.shape[1] >= seq_len, f"Sequence length should be >= {seq_len}"
        
        # Y should be (num_samples, 1)
        assert Y.shape == (num_samples, 1), f"Expected Y shape ({num_samples}, 1), got {Y.shape}"

    def test_generate_adding_data_values(self):
        """Test value ranges in adding data."""
        from experiments.adding import generate_adding_data
        
        X, Y = generate_adding_data(seq_len=100, num_samples=100, seed=42)
        
        # Values (column 0) should be in [0, 1]
        values = X[:, :, 0]
        # Only check non-padded values (where there's any marker activity)
        non_padded = values[X[:, :, 1] != 0]
        if len(non_padded) > 0:
            assert non_padded.min() >= 0.0, f"Values should be >= 0, got {non_padded.min()}"
            assert non_padded.max() <= 1.0, f"Values should be <= 1, got {non_padded.max()}"
        
        # Markers (column 1) should be 0 or 1
        markers = X[:, :, 1]
        unique_markers = np.unique(markers)
        assert all(m in [0.0, 1.0] for m in unique_markers), \
            f"Markers should be 0 or 1, got {unique_markers}"
        
        # Each sample should have exactly 2 markers
        for i in range(X.shape[0]):
            num_markers = np.sum(X[i, :, 1] == 1.0)
            assert num_markers == 2, f"Sample {i} has {num_markers} markers, expected 2"
        
        # Target should be in [0.5, 1.0] (since Y = 0.5 + (X1+X2)/4)
        assert Y.min() >= 0.5, f"Y min should be >= 0.5, got {Y.min()}"
        assert Y.max() <= 1.0, f"Y max should be <= 1.0, got {Y.max()}"

    def test_generate_adding_data_reproducibility(self):
        """Test that seed produces reproducible results."""
        from experiments.adding import generate_adding_data
        
        X1, Y1 = generate_adding_data(seq_len=50, num_samples=10, seed=123)
        X2, Y2 = generate_adding_data(seq_len=50, num_samples=10, seed=123)
        
        assert np.allclose(X1, X2), "Same seed should produce identical X"
        assert np.allclose(Y1, Y2), "Same seed should produce identical Y"


class TestMultiplicationProblemGenerator:
    """Smoke tests for Multiplication Problem data generator."""

    def test_generate_multiplication_data_shapes(self):
        """Test output shapes of multiplication data generator."""
        from experiments.multiplication import generate_multiplication_data
        
        seq_len = 50
        num_samples = 10
        
        X, Y = generate_multiplication_data(seq_len=seq_len, num_samples=num_samples, seed=42)
        
        assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
        assert X.shape[0] == num_samples
        assert X.shape[2] == 2
        assert Y.shape == (num_samples, 1)

    def test_generate_multiplication_data_values(self):
        """Test value ranges in multiplication data."""
        from experiments.multiplication import generate_multiplication_data
        
        X, Y = generate_multiplication_data(seq_len=100, num_samples=100, seed=42)
        
        # Target should be in [0, 1] (product of two [0,1] values)
        assert Y.min() >= 0.0, f"Y min should be >= 0, got {Y.min()}"
        assert Y.max() <= 1.0, f"Y max should be <= 1, got {Y.max()}"
        
        # Each sample should have exactly 2 markers
        for i in range(X.shape[0]):
            num_markers = np.sum(X[i, :, 1] == 1.0)
            assert num_markers == 2, f"Sample {i} has {num_markers} markers, expected 2"

    def test_generate_multiplication_data_reproducibility(self):
        """Test reproducibility with seed."""
        from experiments.multiplication import generate_multiplication_data
        
        X1, Y1 = generate_multiplication_data(seq_len=50, num_samples=10, seed=456)
        X2, Y2 = generate_multiplication_data(seq_len=50, num_samples=10, seed=456)
        
        assert np.allclose(X1, X2)
        assert np.allclose(Y1, Y2)


class TestTemporalOrderGenerator:
    """Smoke tests for Temporal Order data generator."""

    def test_generate_temporal_order_data_shapes_2_symbols(self):
        """Test shapes for 2-symbol temporal order problem."""
        from experiments.temporal_order import generate_temporal_order_data
        
        seq_len = 50
        num_samples = 10
        num_relevant = 2
        
        X, Y = generate_temporal_order_data(
            seq_len=seq_len, 
            num_samples=num_samples, 
            num_relevant=num_relevant,
            seed=42
        )
        
        # X: (num_samples, max_len, 8) - 8 symbols (B, X, Y, E, d1-d4)
        assert X.ndim == 3
        assert X.shape[0] == num_samples
        assert X.shape[2] == 8  # 8 one-hot encoded symbols
        
        # Y: (num_samples, 4) - 4 classes for 2 symbols (XX, XY, YX, YY)
        num_classes = 2 ** num_relevant
        assert Y.shape == (num_samples, num_classes)

    def test_generate_temporal_order_data_shapes_3_symbols(self):
        """Test shapes for 3-symbol temporal order problem."""
        from experiments.temporal_order import generate_temporal_order_data
        
        seq_len = 50
        num_samples = 10
        num_relevant = 3
        
        X, Y = generate_temporal_order_data(
            seq_len=seq_len, 
            num_samples=num_samples, 
            num_relevant=num_relevant,
            seed=42
        )
        
        # Y: (num_samples, 8) - 8 classes for 3 symbols (XXX through YYY)
        num_classes = 2 ** num_relevant
        assert Y.shape == (num_samples, num_classes)

    def test_generate_temporal_order_data_one_hot(self):
        """Verify X is one-hot encoded."""
        from experiments.temporal_order import generate_temporal_order_data
        
        X, Y = generate_temporal_order_data(seq_len=50, num_samples=10, seed=42)
        
        # Each timestep should be one-hot (sum = 1 or 0 for padding)
        sums = X.sum(axis=2)
        valid_sums = np.logical_or(np.isclose(sums, 1.0), np.isclose(sums, 0.0))
        assert valid_sums.all(), "Each timestep should be one-hot or zero (padding)"

    def test_generate_temporal_order_data_labels_one_hot(self):
        """Verify Y is one-hot encoded."""
        from experiments.temporal_order import generate_temporal_order_data
        
        X, Y = generate_temporal_order_data(
            seq_len=50, num_samples=10, num_relevant=2, seed=42
        )
        
        # Each label should be one-hot (sum = 1)
        sums = Y.sum(axis=1)
        assert np.allclose(sums, 1.0), "Each label should be one-hot"

    def test_generate_temporal_order_data_reproducibility(self):
        """Test reproducibility with seed."""
        from experiments.temporal_order import generate_temporal_order_data
        
        X1, Y1 = generate_temporal_order_data(seq_len=50, num_samples=10, seed=789)
        X2, Y2 = generate_temporal_order_data(seq_len=50, num_samples=10, seed=789)
        
        assert np.allclose(X1, X2)
        assert np.allclose(Y1, Y2)


class TestDataGeneratorEdgeCases:
    """Test edge cases for data generators."""

    def test_adding_single_sample(self):
        """Test generating a single sample."""
        from experiments.adding import generate_adding_data
        
        X, Y = generate_adding_data(seq_len=50, num_samples=1, seed=42)
        assert X.shape[0] == 1
        assert Y.shape[0] == 1

    def test_adding_short_sequence(self):
        """Test with minimum sequence length."""
        from experiments.adding import generate_adding_data
        
        X, Y = generate_adding_data(seq_len=10, num_samples=5, seed=42)
        assert X.shape[1] >= 10

    def test_multiplication_single_sample(self):
        """Test generating a single sample."""
        from experiments.multiplication import generate_multiplication_data
        
        X, Y = generate_multiplication_data(seq_len=50, num_samples=1, seed=42)
        assert X.shape[0] == 1
        assert Y.shape[0] == 1

    def test_temporal_order_single_sample(self):
        """Test generating a single sample."""
        from experiments.temporal_order import generate_temporal_order_data
        
        X, Y = generate_temporal_order_data(seq_len=50, num_samples=1, seed=42)
        assert X.shape[0] == 1
        assert Y.shape[0] == 1

    def test_data_types(self):
        """Verify all generators return float32 data."""
        from experiments.adding import generate_adding_data
        from experiments.multiplication import generate_multiplication_data
        from experiments.temporal_order import generate_temporal_order_data
        
        X1, Y1 = generate_adding_data(seq_len=50, num_samples=5, seed=42)
        X2, Y2 = generate_multiplication_data(seq_len=50, num_samples=5, seed=42)
        X3, Y3 = generate_temporal_order_data(seq_len=50, num_samples=5, seed=42)
        
        assert X1.dtype == np.float32, f"Adding X dtype should be float32, got {X1.dtype}"
        assert Y1.dtype == np.float32, f"Adding Y dtype should be float32, got {Y1.dtype}"
        assert X2.dtype == np.float32, f"Multiplication X dtype should be float32"
        assert Y2.dtype == np.float32, f"Multiplication Y dtype should be float32"
        assert X3.dtype == np.float32, f"Temporal Order X dtype should be float32"
        assert Y3.dtype == np.float32, f"Temporal Order Y dtype should be float32"


class TestExperimentImports:
    """Test that experiment modules can be imported."""

    def test_import_adding(self):
        """Test adding experiment imports successfully."""
        from experiments import adding
        assert hasattr(adding, 'generate_adding_data')
        assert hasattr(adding, 'run_tinygrad')
        assert hasattr(adding, 'run_torch')

    def test_import_multiplication(self):
        """Test multiplication experiment imports successfully."""
        from experiments import multiplication
        assert hasattr(multiplication, 'generate_multiplication_data')
        assert hasattr(multiplication, 'run_tinygrad')
        assert hasattr(multiplication, 'run_torch')

    def test_import_temporal_order(self):
        """Test temporal order experiment imports successfully."""
        from experiments import temporal_order
        assert hasattr(temporal_order, 'generate_temporal_order_data')
        assert hasattr(temporal_order, 'run_tinygrad')
        assert hasattr(temporal_order, 'run_torch')
