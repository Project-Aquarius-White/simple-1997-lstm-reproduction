"""
Test activation functions g() and h().

The 1997 LSTM paper specifies:
- g(x) = 4 * sigmoid(x) - 2, range [-2, 2]  (input squashing)
- h(x) = 2 * sigmoid(x) - 1, range [-1, 1]  (output squashing)

Reference: Appendix A.2 and A.3 of Hochreiter & Schmidhuber (1997)
"""

import pytest
import numpy as np


# Check for optional dependencies
try:
    from tinygrad.tensor import Tensor as TinyTensor
    HAS_TINYGRAD = True
except ImportError:
    HAS_TINYGRAD = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestActivationsNumpy:
    """Test numpy activation function implementations."""

    def test_g_squash_range(self):
        """Verify g(x) output is in range [-2, 2]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_np
        
        # Test over wide input range
        x = np.linspace(-100, 100, 10000)
        y = g_squash_np(x)
        
        assert y.min() >= -2.0, f"g(x) min should be >= -2, got {y.min()}"
        assert y.max() <= 2.0, f"g(x) max should be <= 2, got {y.max()}"
        
        # At extreme values, should approach bounds
        assert np.isclose(g_squash_np(np.array([-100.0]))[0], -2.0, atol=1e-10)
        assert np.isclose(g_squash_np(np.array([100.0]))[0], 2.0, atol=1e-10)

    def test_h_squash_range(self):
        """Verify h(x) output is in range [-1, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_np
        
        # Test over wide input range
        x = np.linspace(-100, 100, 10000)
        y = h_squash_np(x)
        
        assert y.min() >= -1.0, f"h(x) min should be >= -1, got {y.min()}"
        assert y.max() <= 1.0, f"h(x) max should be <= 1, got {y.max()}"
        
        # At extreme values, should approach bounds
        assert np.isclose(h_squash_np(np.array([-100.0]))[0], -1.0, atol=1e-10)
        assert np.isclose(h_squash_np(np.array([100.0]))[0], 1.0, atol=1e-10)

    def test_g_squash_zero_centered(self):
        """Verify g(0) = 0 (zero-centered activation)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_np
        
        y = g_squash_np(np.array([0.0]))
        assert np.isclose(y[0], 0.0), f"g(0) should be 0, got {y[0]}"

    def test_h_squash_zero_centered(self):
        """Verify h(0) = 0 (zero-centered activation)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_np
        
        y = h_squash_np(np.array([0.0]))
        assert np.isclose(y[0], 0.0), f"h(0) should be 0, got {y[0]}"

    def test_sigmoid_range(self):
        """Verify sigmoid(x) output is in range [0, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import sigmoid_np
        
        x = np.linspace(-100, 100, 10000)
        y = sigmoid_np(x)
        
        assert y.min() >= 0.0, f"sigmoid min should be >= 0, got {y.min()}"
        assert y.max() <= 1.0, f"sigmoid max should be <= 1, got {y.max()}"

    def test_g_squash_formula(self):
        """Verify g(x) = 4 * sigmoid(x) - 2."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_np, sigmoid_np
        
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        g_result = g_squash_np(x)
        expected = 4.0 * sigmoid_np(x) - 2.0
        
        assert np.allclose(g_result, expected), \
            f"g(x) should equal 4*sigmoid(x)-2, got {g_result} vs {expected}"

    def test_h_squash_formula(self):
        """Verify h(x) = 2 * sigmoid(x) - 1."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_np, sigmoid_np
        
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        h_result = h_squash_np(x)
        expected = 2.0 * sigmoid_np(x) - 1.0
        
        assert np.allclose(h_result, expected), \
            f"h(x) should equal 2*sigmoid(x)-1, got {h_result} vs {expected}"

    def test_activations_monotonic(self):
        """Verify g(x) and h(x) are monotonically increasing."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_np, h_squash_np
        
        x = np.linspace(-10, 10, 1000)
        
        g_result = g_squash_np(x)
        h_result = h_squash_np(x)
        
        # Check monotonicity: all differences should be non-negative
        g_diff = np.diff(g_result)
        h_diff = np.diff(h_result)
        
        assert np.all(g_diff >= 0), "g(x) should be monotonically increasing"
        assert np.all(h_diff >= 0), "h(x) should be monotonically increasing"


class TestActivationsTinygrad:
    """Test tinygrad activation function implementations."""

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_g_squash_range(self):
        """Verify g(x) output is in range [-2, 2]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.activations import g_squash
        
        x = Tensor(np.linspace(-100, 100, 1000).astype(np.float32))
        y = g_squash(x).numpy()
        
        assert y.min() >= -2.0, f"g(x) min should be >= -2, got {y.min()}"
        assert y.max() <= 2.0, f"g(x) max should be <= 2, got {y.max()}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_h_squash_range(self):
        """Verify h(x) output is in range [-1, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.activations import h_squash
        
        x = Tensor(np.linspace(-100, 100, 1000).astype(np.float32))
        y = h_squash(x).numpy()
        
        assert y.min() >= -1.0, f"h(x) min should be >= -1, got {y.min()}"
        assert y.max() <= 1.0, f"h(x) max should be <= 1, got {y.max()}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_g_squash_zero_centered(self):
        """Verify g(0) = 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.activations import g_squash
        
        y = g_squash(Tensor([0.0])).numpy()[0]
        assert np.isclose(y, 0.0, atol=1e-6), f"g(0) should be 0, got {y}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_h_squash_zero_centered(self):
        """Verify h(0) = 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.activations import h_squash
        
        y = h_squash(Tensor([0.0])).numpy()[0]
        assert np.isclose(y, 0.0, atol=1e-6), f"h(0) should be 0, got {y}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_tinygrad_numpy_consistency(self):
        """Verify tinygrad and numpy implementations match."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.activations import g_squash, h_squash, g_squash_np, h_squash_np
        
        x_np = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        x_tg = Tensor(x_np)
        
        g_np = g_squash_np(x_np)
        g_tg = g_squash(x_tg).numpy()
        
        h_np = h_squash_np(x_np)
        h_tg = h_squash(x_tg).numpy()
        
        assert np.allclose(g_np, g_tg, atol=1e-5), \
            f"g_squash mismatch: numpy={g_np}, tinygrad={g_tg}"
        assert np.allclose(h_np, h_tg, atol=1e-5), \
            f"h_squash mismatch: numpy={h_np}, tinygrad={h_tg}"


class TestActivationsTorch:
    """Test PyTorch activation function implementations."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_g_squash_range(self):
        """Verify g(x) output is in range [-2, 2]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_torch
        
        x = torch.linspace(-100, 100, 1000)
        y = g_squash_torch(x).numpy()
        
        assert y.min() >= -2.0, f"g(x) min should be >= -2, got {y.min()}"
        assert y.max() <= 2.0, f"g(x) max should be <= 2, got {y.max()}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_h_squash_range(self):
        """Verify h(x) output is in range [-1, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_torch
        
        x = torch.linspace(-100, 100, 1000)
        y = h_squash_torch(x).numpy()
        
        assert y.min() >= -1.0, f"h(x) min should be >= -1, got {y.min()}"
        assert y.max() <= 1.0, f"h(x) max should be <= 1, got {y.max()}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_g_squash_zero_centered(self):
        """Verify g(0) = 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_torch
        
        y = g_squash_torch(torch.tensor([0.0])).numpy()[0]
        assert np.isclose(y, 0.0, atol=1e-6), f"g(0) should be 0, got {y}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_h_squash_zero_centered(self):
        """Verify h(0) = 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_torch
        
        y = h_squash_torch(torch.tensor([0.0])).numpy()[0]
        assert np.isclose(y, 0.0, atol=1e-6), f"h(0) should be 0, got {y}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_numpy_consistency(self):
        """Verify PyTorch and numpy implementations match."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import (
            g_squash_torch, h_squash_torch, 
            g_squash_np, h_squash_np
        )
        
        x_np = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        x_pt = torch.tensor(x_np)
        
        g_np = g_squash_np(x_np)
        g_pt = g_squash_torch(x_pt).numpy()
        
        h_np = h_squash_np(x_np)
        h_pt = h_squash_torch(x_pt).numpy()
        
        assert np.allclose(g_np, g_pt, atol=1e-5), \
            f"g_squash mismatch: numpy={g_np}, torch={g_pt}"
        assert np.allclose(h_np, h_pt, atol=1e-5), \
            f"h_squash mismatch: numpy={h_np}, torch={h_pt}"


class TestDerivatives:
    """Test activation function derivatives."""

    def test_g_derivative_formula(self):
        """Verify g'(x) = 4 * sigmoid(x) * (1 - sigmoid(x))."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import g_squash_derivative_np, sigmoid_np
        
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        derivative = g_squash_derivative_np(x)
        
        s = sigmoid_np(x)
        expected = 4.0 * s * (1.0 - s)
        
        assert np.allclose(derivative, expected), \
            f"g'(x) formula mismatch: got {derivative}, expected {expected}"

    def test_h_derivative_formula(self):
        """Verify h'(x) = 2 * sigmoid(x) * (1 - sigmoid(x))."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import h_squash_derivative_np, sigmoid_np
        
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        derivative = h_squash_derivative_np(x)
        
        s = sigmoid_np(x)
        expected = 2.0 * s * (1.0 - s)
        
        assert np.allclose(derivative, expected), \
            f"h'(x) formula mismatch: got {derivative}, expected {expected}"

    def test_derivatives_positive(self):
        """Verify derivatives are always positive (monotonic functions)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import (
            g_squash_derivative_np, h_squash_derivative_np, sigmoid_derivative_np
        )
        
        x = np.linspace(-10, 10, 1000)
        
        assert np.all(sigmoid_derivative_np(x) >= 0), "sigmoid' should be >= 0"
        assert np.all(g_squash_derivative_np(x) >= 0), "g' should be >= 0"
        assert np.all(h_squash_derivative_np(x) >= 0), "h' should be >= 0"

    def test_derivative_max_at_zero(self):
        """Verify derivatives are maximal at x=0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import (
            g_squash_derivative_np, h_squash_derivative_np, sigmoid_derivative_np
        )
        
        x = np.linspace(-10, 10, 1000)
        
        # sigmoid'(0) = 0.25, g'(0) = 1.0, h'(0) = 0.5
        assert np.argmax(sigmoid_derivative_np(x)) == len(x) // 2
        assert np.argmax(g_squash_derivative_np(x)) == len(x) // 2
        assert np.argmax(h_squash_derivative_np(x)) == len(x) // 2

    def test_specific_derivative_values(self):
        """Test known derivative values at x=0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.activations import (
            g_squash_derivative_np, h_squash_derivative_np, sigmoid_derivative_np
        )
        
        x = np.array([0.0])
        
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        assert np.isclose(sigmoid_derivative_np(x)[0], 0.25)
        
        # g'(0) = 4 * 0.25 = 1.0
        assert np.isclose(g_squash_derivative_np(x)[0], 1.0)
        
        # h'(0) = 2 * 0.25 = 0.5
        assert np.isclose(h_squash_derivative_np(x)[0], 0.5)
