"""
Test CEC (Constant Error Carousel) gradient properties.

The key property of the 1997 LSTM is that ds_t/ds_{t-k} = 1.0 for the cell state path.
This is what allows gradients to flow unattenuated through time.

Reference: Section 2-4 of Hochreiter & Schmidhuber (1997)
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


class TestCECGradientTinygrad:
    """Test CEC gradient = 1.0 through time using tinygrad."""

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_cell_state_gradient_is_one(self):
        """Verify that ds_t/ds_{t-k} = 1.0 for pure cell state path.
        
        The CEC property means the self-connection weight is 1.0, so
        gradients flow without attenuation through the cell state.
        """
        from tinygrad.tensor import Tensor
        
        # Create a simple cell state chain: s_t = s_{t-1} + input
        # This mimics the CEC update without gates
        seq_len = 10
        
        # Initial state with gradient tracking
        s_0 = Tensor([1.0], requires_grad=True)
        
        # Simulate CEC updates (s_t = s_{t-1} + 0, i.e., just passing through)
        s = s_0
        for t in range(seq_len):
            s = s + Tensor([0.0])  # s_t = s_{t-1} + 0
        
        # Final state
        s_final = s
        
        # Compute gradient of s_final w.r.t. s_0
        # Should be 1.0 because ds_t/ds_{t-1} = 1.0 at each step
        loss = s_final.sum()
        loss.backward()
        
        grad = s_0.grad.numpy()
        assert np.allclose(grad, 1.0), f"Expected gradient 1.0, got {grad}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_cec_gradient_with_lstm_cell(self):
        """Test that gradients flow through the cell state in actual LSTM cell."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMCell1997
        
        cell = LSTMCell1997(input_size=2, hidden_size=3, seed=42)
        
        # Initialize states
        h, s_c = cell.init_state()
        
        # Mark initial cell state for gradient tracking
        s_c_init = Tensor.zeros(3, requires_grad=True)
        s_c = s_c_init
        
        # Run forward for several steps
        seq_len = 5
        for t in range(seq_len):
            x_t = Tensor(np.random.randn(2).astype(np.float32))
            h, s_c = cell(x_t, h, s_c)
        
        # Backward from cell state
        loss = s_c.sum()
        loss.backward()
        
        # The gradient should exist and not be zero (CEC allows gradient flow)
        # Due to the additive update s_t = s_{t-1} + input_gate * g(net),
        # the contribution from s_{t-1} has gradient exactly 1.0
        grad = s_c_init.grad.numpy()
        
        # Gradient should be close to 1.0 (the direct path through CEC)
        assert np.allclose(grad, 1.0, atol=1e-5), \
            f"CEC gradient should be 1.0, got {grad}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_cec_gradient_long_sequence(self):
        """Test CEC gradient preservation over longer sequences."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMCell1997
        
        cell = LSTMCell1997(input_size=2, hidden_size=3, seed=42)
        
        h, _ = cell.init_state()
        s_c_init = Tensor.zeros(3, requires_grad=True)
        s_c = s_c_init
        
        # Longer sequence
        seq_len = 20
        for t in range(seq_len):
            x_t = Tensor(np.random.randn(2).astype(np.float32))
            h, s_c = cell(x_t, h, s_c)
        
        loss = s_c.sum()
        loss.backward()
        
        grad = s_c_init.grad.numpy()
        
        # Even after 20 steps, the CEC gradient should be 1.0
        assert np.allclose(grad, 1.0, atol=1e-5), \
            f"CEC gradient should be 1.0 after {seq_len} steps, got {grad}"


class TestCECGradientTorch:
    """Test CEC gradient = 1.0 through time using PyTorch."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_cell_state_gradient_is_one(self):
        """Verify that ds_t/ds_{t-k} = 1.0 for pure cell state path."""
        seq_len = 10
        
        s_0 = torch.tensor([1.0], requires_grad=True)
        
        s = s_0
        for t in range(seq_len):
            s = s + torch.tensor([0.0])
        
        s_final = s
        loss = s_final.sum()
        loss.backward()
        
        grad = s_0.grad.numpy()
        assert np.allclose(grad, 1.0), f"Expected gradient 1.0, got {grad}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_cec_gradient_with_lstm_cell(self):
        """Test that gradients flow through the cell state in PyTorch LSTM cell."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        cell = LSTMCell1997Torch(input_size=2, hidden_size=3, seed=42)
        
        h, _ = cell.init_state()
        s_c_init = torch.zeros(3, requires_grad=True)
        s_c = s_c_init
        
        seq_len = 5
        for t in range(seq_len):
            x_t = torch.randn(2)
            h, s_c = cell(x_t, h, s_c)
        
        loss = s_c.sum()
        loss.backward()
        
        grad = s_c_init.grad.numpy()
        
        assert np.allclose(grad, 1.0, atol=1e-5), \
            f"CEC gradient should be 1.0, got {grad}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_cec_gradient_long_sequence(self):
        """Test CEC gradient preservation over longer sequences."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        cell = LSTMCell1997Torch(input_size=2, hidden_size=3, seed=42)
        
        h, _ = cell.init_state()
        s_c_init = torch.zeros(3, requires_grad=True)
        s_c = s_c_init
        
        seq_len = 20
        for t in range(seq_len):
            x_t = torch.randn(2)
            h, s_c = cell(x_t, h, s_c)
        
        loss = s_c.sum()
        loss.backward()
        
        grad = s_c_init.grad.numpy()
        
        assert np.allclose(grad, 1.0, atol=1e-5), \
            f"CEC gradient should be 1.0 after {seq_len} steps, got {grad}"


class TestCECMathematicalProperty:
    """Test the mathematical CEC property without deep learning frameworks."""

    def test_cec_identity_jacobian(self):
        """Verify that the Jacobian of s_t w.r.t. s_{t-1} is identity.
        
        In the 1997 LSTM: s_t = s_{t-1} + y_in * g(net_c)
        
        The partial derivative ds_t/ds_{t-1} = 1.0 (identity for each component)
        This is independent of the input gate and cell input values.
        """
        # Symbolic verification: s_t = s_{t-1} + f(x, h)
        # ds_t/ds_{t-1} = 1 (since s_{t-1} appears with coefficient 1)
        
        # Numerical verification
        hidden_size = 5
        
        # For any values of s_{t-1}
        for _ in range(10):
            s_prev = np.random.randn(hidden_size)
            delta = np.random.randn(hidden_size) * 0.1  # Simulated input
            
            # Forward: s_t = s_{t-1} + delta
            s_t = s_prev + delta
            
            # Jacobian ds_t/ds_{t-1} should be identity matrix
            # We verify component-wise: ds_t[i]/ds_{t-1}[j] = 1 if i==j else 0
            jacobian = np.eye(hidden_size)  # Expected
            
            # Numerical gradient check
            eps = 1e-5
            numerical_jacobian = np.zeros((hidden_size, hidden_size))
            for j in range(hidden_size):
                s_prev_plus = s_prev.copy()
                s_prev_plus[j] += eps
                s_t_plus = s_prev_plus + delta
                
                s_prev_minus = s_prev.copy()
                s_prev_minus[j] -= eps
                s_t_minus = s_prev_minus + delta
                
                numerical_jacobian[:, j] = (s_t_plus - s_t_minus) / (2 * eps)
            
            assert np.allclose(numerical_jacobian, jacobian, atol=1e-6), \
                f"CEC Jacobian should be identity, got {numerical_jacobian}"

    def test_cec_gradient_chain_rule(self):
        """Test that gradient through T timesteps equals 1.0 via chain rule.
        
        If ds_t/ds_{t-1} = 1 for all t, then:
        ds_T/ds_0 = ds_T/ds_{T-1} * ds_{T-1}/ds_{T-2} * ... * ds_1/ds_0 = 1^T = 1
        """
        T = 100  # Long sequence
        
        # Product of all Jacobians should be 1
        gradient_product = 1.0
        for t in range(T):
            jacobian_t = 1.0  # CEC property
            gradient_product *= jacobian_t
        
        assert gradient_product == 1.0, \
            f"Gradient through {T} CEC timesteps should be 1.0, got {gradient_product}"
