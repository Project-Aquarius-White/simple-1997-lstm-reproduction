"""
Test tensor shapes through forward pass.

Verifies that the LSTM cell produces correct output shapes for both
1D (single sample) and 2D (batched) inputs.

Reference: cell.py forward() documentation
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


class TestForwardShapesTinygrad:
    """Test forward pass shapes using tinygrad."""

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_1d_input_shapes(self):
        """Test shapes for 1D (unbatched) input."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMCell1997
        
        input_size = 5
        hidden_size = 10
        
        cell = LSTMCell1997(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # 1D input: shape (input_size,)
        x_t = Tensor(np.random.randn(input_size).astype(np.float32))
        h_prev, s_c_prev = cell.init_state(batch_size=1)
        
        # Forward pass
        h_t, s_c_t = cell(x_t, h_prev, s_c_prev)
        
        # Verify shapes
        h_shape = h_t.shape
        s_c_shape = s_c_t.shape
        
        assert h_shape == (hidden_size,), \
            f"Expected h_t shape ({hidden_size},), got {h_shape}"
        assert s_c_shape == (hidden_size,), \
            f"Expected s_c_t shape ({hidden_size},), got {s_c_shape}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_batched_input_shapes(self):
        """Test shapes for 2D (batched) input."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMCell1997
        
        input_size = 5
        hidden_size = 10
        batch_size = 8
        
        cell = LSTMCell1997(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # 2D input: shape (batch_size, input_size)
        x_t = Tensor(np.random.randn(batch_size, input_size).astype(np.float32))
        h_prev, s_c_prev = cell.init_state(batch_size=batch_size)
        
        # Forward pass
        h_t, s_c_t = cell(x_t, h_prev, s_c_prev)
        
        # Verify shapes
        h_shape = h_t.shape
        s_c_shape = s_c_t.shape
        
        assert h_shape == (batch_size, hidden_size), \
            f"Expected h_t shape ({batch_size}, {hidden_size}), got {h_shape}"
        assert s_c_shape == (batch_size, hidden_size), \
            f"Expected s_c_t shape ({batch_size}, {hidden_size}), got {s_c_shape}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_sequence_processing_shapes(self):
        """Test shapes through a full sequence."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMModel1997
        
        input_size = 5
        hidden_size = 10
        output_size = 3
        seq_len = 15
        
        model = LSTMModel1997(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=42
        )
        
        # Sequence input: (seq_len, input_size)
        x_seq = Tensor(np.random.randn(seq_len, input_size).astype(np.float32))
        
        pred, h_final, s_c_final = model.forward_sequence(x_seq)
        
        # Verify shapes
        assert pred.shape == (output_size,), \
            f"Expected prediction shape ({output_size},), got {pred.shape}"
        assert h_final.shape == (hidden_size,), \
            f"Expected h_final shape ({hidden_size},), got {h_final.shape}"
        assert s_c_final.shape == (hidden_size,), \
            f"Expected s_c_final shape ({hidden_size},), got {s_c_final.shape}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_batched_sequence_shapes(self):
        """Test shapes for batched sequence processing."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from tinygrad.tensor import Tensor
        from aquarius_lstm.cell import LSTMModel1997
        
        input_size = 5
        hidden_size = 10
        output_size = 3
        seq_len = 15
        batch_size = 4
        
        model = LSTMModel1997(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=42
        )
        
        # Batched sequence: (seq_len, batch_size, input_size)
        x_seq = Tensor(np.random.randn(seq_len, batch_size, input_size).astype(np.float32))
        
        pred, h_final, s_c_final = model.forward_sequence(x_seq)
        
        # Verify shapes
        assert pred.shape == (batch_size, output_size), \
            f"Expected prediction shape ({batch_size}, {output_size}), got {pred.shape}"
        assert h_final.shape == (batch_size, hidden_size), \
            f"Expected h_final shape ({batch_size}, {hidden_size}), got {h_final.shape}"
        assert s_c_final.shape == (batch_size, hidden_size), \
            f"Expected s_c_final shape ({batch_size}, {hidden_size}), got {s_c_final.shape}"

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_init_state_shapes(self):
        """Test init_state produces correct shapes."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell import LSTMCell1997
        
        hidden_size = 10
        cell = LSTMCell1997(input_size=5, hidden_size=hidden_size, seed=42)
        
        # Single sample
        h, s_c = cell.init_state(batch_size=1)
        assert h.shape == (hidden_size,), f"Expected ({hidden_size},), got {h.shape}"
        assert s_c.shape == (hidden_size,), f"Expected ({hidden_size},), got {s_c.shape}"
        
        # Batched
        batch_size = 8
        h, s_c = cell.init_state(batch_size=batch_size)
        assert h.shape == (batch_size, hidden_size), \
            f"Expected ({batch_size}, {hidden_size}), got {h.shape}"
        assert s_c.shape == (batch_size, hidden_size), \
            f"Expected ({batch_size}, {hidden_size}), got {s_c.shape}"


class TestForwardShapesTorch:
    """Test forward pass shapes using PyTorch."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_1d_input_shapes(self):
        """Test shapes for 1D (unbatched) input."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        input_size = 5
        hidden_size = 10
        
        cell = LSTMCell1997Torch(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # 1D input: shape (input_size,)
        x_t = torch.randn(input_size)
        h_prev, s_c_prev = cell.init_state(batch_size=1)
        
        # Forward pass
        h_t, s_c_t = cell(x_t, h_prev, s_c_prev)
        
        # Verify shapes
        assert h_t.shape == (hidden_size,), \
            f"Expected h_t shape ({hidden_size},), got {tuple(h_t.shape)}"
        assert s_c_t.shape == (hidden_size,), \
            f"Expected s_c_t shape ({hidden_size},), got {tuple(s_c_t.shape)}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_batched_input_shapes(self):
        """Test shapes for 2D (batched) input."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        input_size = 5
        hidden_size = 10
        batch_size = 8
        
        cell = LSTMCell1997Torch(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # 2D input: shape (batch_size, input_size)
        x_t = torch.randn(batch_size, input_size)
        h_prev, s_c_prev = cell.init_state(batch_size=batch_size)
        
        # Forward pass
        h_t, s_c_t = cell(x_t, h_prev, s_c_prev)
        
        # Verify shapes
        assert h_t.shape == (batch_size, hidden_size), \
            f"Expected h_t shape ({batch_size}, {hidden_size}), got {tuple(h_t.shape)}"
        assert s_c_t.shape == (batch_size, hidden_size), \
            f"Expected s_c_t shape ({batch_size}, {hidden_size}), got {tuple(s_c_t.shape)}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_sequence_processing_shapes(self):
        """Test shapes through a full sequence."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMModel1997Torch
        
        input_size = 5
        hidden_size = 10
        output_size = 3
        seq_len = 15
        
        model = LSTMModel1997Torch(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=42
        )
        
        # Sequence input: (seq_len, input_size)
        x_seq = torch.randn(seq_len, input_size)
        
        pred, h_final, s_c_final = model.forward_sequence(x_seq)
        
        # Verify shapes
        assert pred.shape == (output_size,), \
            f"Expected prediction shape ({output_size},), got {tuple(pred.shape)}"
        assert h_final.shape == (hidden_size,), \
            f"Expected h_final shape ({hidden_size},), got {tuple(h_final.shape)}"
        assert s_c_final.shape == (hidden_size,), \
            f"Expected s_c_final shape ({hidden_size},), got {tuple(s_c_final.shape)}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_batched_sequence_shapes(self):
        """Test shapes for batched sequence processing."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMModel1997Torch
        
        input_size = 5
        hidden_size = 10
        output_size = 3
        seq_len = 15
        batch_size = 4
        
        model = LSTMModel1997Torch(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=42
        )
        
        # Batched sequence: (seq_len, batch_size, input_size)
        x_seq = torch.randn(seq_len, batch_size, input_size)
        
        pred, h_final, s_c_final = model.forward_sequence(x_seq)
        
        # Verify shapes
        assert pred.shape == (batch_size, output_size), \
            f"Expected prediction shape ({batch_size}, {output_size}), got {tuple(pred.shape)}"
        assert h_final.shape == (batch_size, hidden_size), \
            f"Expected h_final shape ({batch_size}, {hidden_size}), got {tuple(h_final.shape)}"
        assert s_c_final.shape == (batch_size, hidden_size), \
            f"Expected s_c_final shape ({batch_size}, {hidden_size}), got {tuple(s_c_final.shape)}"

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_init_state_shapes(self):
        """Test init_state produces correct shapes."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        hidden_size = 10
        cell = LSTMCell1997Torch(input_size=5, hidden_size=hidden_size, seed=42)
        
        # Single sample
        h, s_c = cell.init_state(batch_size=1)
        assert h.shape == (hidden_size,), f"Expected ({hidden_size},), got {tuple(h.shape)}"
        assert s_c.shape == (hidden_size,), f"Expected ({hidden_size},), got {tuple(s_c.shape)}"
        
        # Batched
        batch_size = 8
        h, s_c = cell.init_state(batch_size=batch_size)
        assert h.shape == (batch_size, hidden_size), \
            f"Expected ({batch_size}, {hidden_size}), got {tuple(h.shape)}"
        assert s_c.shape == (batch_size, hidden_size), \
            f"Expected ({batch_size}, {hidden_size}), got {tuple(s_c.shape)}"


class TestParameterShapes:
    """Test that model parameters have expected shapes."""

    @pytest.mark.skipif(not HAS_TINYGRAD, reason="tinygrad not installed")
    def test_tinygrad_parameter_shapes(self):
        """Verify parameter shapes in tinygrad cell."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell import LSTMCell1997
        
        input_size = 5
        hidden_size = 10
        
        cell = LSTMCell1997(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # Weight matrices: (hidden_size, input_size)
        assert cell.W_in.shape == (hidden_size, input_size)
        assert cell.W_out.shape == (hidden_size, input_size)
        assert cell.W_c.shape == (hidden_size, input_size)
        
        # Recurrent matrices: (hidden_size, hidden_size)
        assert cell.U_in.shape == (hidden_size, hidden_size)
        assert cell.U_out.shape == (hidden_size, hidden_size)
        assert cell.U_c.shape == (hidden_size, hidden_size)
        
        # Biases: (hidden_size,)
        assert cell.b_in.shape == (hidden_size,)
        assert cell.b_out.shape == (hidden_size,)
        assert cell.b_c.shape == (hidden_size,)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
    def test_torch_parameter_shapes(self):
        """Verify parameter shapes in PyTorch cell."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from aquarius_lstm.cell_torch import LSTMCell1997Torch
        
        input_size = 5
        hidden_size = 10
        
        cell = LSTMCell1997Torch(input_size=input_size, hidden_size=hidden_size, seed=42)
        
        # Weight matrices: (hidden_size, input_size)
        assert cell.W_in.shape == (hidden_size, input_size)
        assert cell.W_out.shape == (hidden_size, input_size)
        assert cell.W_c.shape == (hidden_size, input_size)
        
        # Recurrent matrices: (hidden_size, hidden_size)
        assert cell.U_in.shape == (hidden_size, hidden_size)
        assert cell.U_out.shape == (hidden_size, hidden_size)
        assert cell.U_c.shape == (hidden_size, hidden_size)
        
        # Biases: (hidden_size,)
        assert cell.b_in.shape == (hidden_size,)
        assert cell.b_out.shape == (hidden_size,)
        assert cell.b_c.shape == (hidden_size,)
