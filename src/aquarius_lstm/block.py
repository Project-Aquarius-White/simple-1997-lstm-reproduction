"""
LSTM Memory Cell Block implementation (1997 paper).

The original paper describes "memory cell blocks" where multiple memory cells
share the same input gate and output gate. This reduces parameter count and
is how the paper's experiments are structured.

From the paper (Section 2):
    "A memory cell block of size S has S memory cells sharing the same
    input gate and the same output gate."

Reference: Section 2 and Section 5 (experiment architectures)
"""

from typing import Tuple, Optional, List
from tinygrad.tensor import Tensor
import numpy as np

from .activations import g_squash, h_squash
from .initialization import init_weights_paper


class LSTMBlock1997:
    """
    A Memory Cell Block where S cells share gates.
    
    This is the architecture actually used in many of the paper's experiments.
    Multiple memory cells share the same input and output gates, which:
    1. Reduces parameter count
    2. Forces cells within a block to coordinate
    3. Allows different blocks to specialize for different purposes
    
    Attributes:
        input_size: Dimension of input vector
        block_size: Number of memory cells per block (S in the paper)
        num_blocks: Number of blocks
        hidden_size: Total hidden size = block_size * num_blocks
    """
    
    def __init__(
        self,
        input_size: int,
        block_size: int = 1,
        num_blocks: int = 4,
        init_range: float = 0.1,
        input_gate_biases: Optional[List[float]] = None,
        output_gate_biases: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LSTM block.
        
        Args:
            input_size: Size of input vector
            block_size: Number of cells per block (S)
            num_blocks: Number of blocks
            init_range: Weight initialization range
            input_gate_biases: Per-block input gate biases
            output_gate_biases: Per-block output gate biases
            seed: Random seed
        """
        self.input_size = input_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.hidden_size = block_size * num_blocks
        
        if seed is not None:
            np.random.seed(seed)
        
        # Default biases if not specified
        if input_gate_biases is None:
            input_gate_biases = [-2.0] * num_blocks
        if output_gate_biases is None:
            output_gate_biases = [float(-i-1) for i in range(num_blocks)]
        
        # Ensure we have enough bias values
        while len(input_gate_biases) < num_blocks:
            input_gate_biases.append(input_gate_biases[-1])
        while len(output_gate_biases) < num_blocks:
            output_gate_biases.append(output_gate_biases[-1])
        
        # ====================================================================
        # Shared Input Gates (one per block)
        # Shape: (num_blocks,) since gates are shared within each block
        # ====================================================================
        self.W_in = Tensor(
            init_weights_paper((num_blocks, input_size), init_range),
            requires_grad=True
        )
        self.U_in = Tensor(
            init_weights_paper((num_blocks, self.hidden_size), init_range),
            requires_grad=True
        )
        self.b_in = Tensor(
            np.array(input_gate_biases[:num_blocks], dtype=np.float32),
            requires_grad=True
        )
        
        # ====================================================================
        # Shared Output Gates (one per block)
        # ====================================================================
        self.W_out = Tensor(
            init_weights_paper((num_blocks, input_size), init_range),
            requires_grad=True
        )
        self.U_out = Tensor(
            init_weights_paper((num_blocks, self.hidden_size), init_range),
            requires_grad=True
        )
        self.b_out = Tensor(
            np.array(output_gate_biases[:num_blocks], dtype=np.float32),
            requires_grad=True
        )
        
        # ====================================================================
        # Cell Inputs (one weight set per cell, so hidden_size total)
        # ====================================================================
        self.W_c = Tensor(
            init_weights_paper((self.hidden_size, input_size), init_range),
            requires_grad=True
        )
        self.U_c = Tensor(
            init_weights_paper((self.hidden_size, self.hidden_size), init_range),
            requires_grad=True
        )
        self.b_c = Tensor(
            np.zeros(self.hidden_size, dtype=np.float32),
            requires_grad=True
        )
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        return [
            self.W_in, self.U_in, self.b_in,
            self.W_out, self.U_out, self.b_out,
            self.W_c, self.U_c, self.b_c,
        ]
    
    def forward(
        self,
        x_t: Tensor,
        h_prev: Tensor,
        s_c_prev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for one timestep.
        
        The key difference from LSTMCell1997 is that gates are computed
        per-block and then expanded to cover all cells in each block.
        
        Args:
            x_t: Input, shape (input_size,) or (batch, input_size)
            h_prev: Previous hidden state, shape (hidden_size,) or (batch, hidden_size)
            s_c_prev: Previous cell state, shape (hidden_size,) or (batch, hidden_size)
        
        Returns:
            h_t: Current hidden state
            s_c_t: Current cell state
        """
        # Truncated backprop: freeze recurrent signal for gates
        h_frozen = h_prev.detach()
        
        is_batched = len(x_t.shape) == 2
        
        if is_batched:
            def matmul(W, x):
                return x.matmul(W.T)
        else:
            def matmul(W, x):
                return W.dot(x)
        
        # ====================================================================
        # Compute per-block gate activations
        # Shape: (num_blocks,) or (batch, num_blocks)
        # ====================================================================
        net_in = matmul(self.W_in, x_t) + matmul(self.U_in, h_frozen) + self.b_in
        y_in_blocks = net_in.sigmoid()  # (num_blocks,) or (batch, num_blocks)
        
        net_out = matmul(self.W_out, x_t) + matmul(self.U_out, h_frozen) + self.b_out
        y_out_blocks = net_out.sigmoid()
        
        # ====================================================================
        # Expand gate activations to cover all cells in each block
        # Each gate value is repeated block_size times
        # ====================================================================
        # Reshape and repeat: (num_blocks,) -> (num_blocks, 1) -> (num_blocks, block_size) -> (hidden_size,)
        if is_batched:
            # (batch, num_blocks) -> (batch, num_blocks, block_size) -> (batch, hidden_size)
            y_in = y_in_blocks.reshape(-1, self.num_blocks, 1).expand(-1, -1, self.block_size).reshape(-1, self.hidden_size)
            y_out = y_out_blocks.reshape(-1, self.num_blocks, 1).expand(-1, -1, self.block_size).reshape(-1, self.hidden_size)
        else:
            # Simple repeat for 1D case
            y_in = y_in_blocks.reshape(self.num_blocks, 1).expand(self.num_blocks, self.block_size).reshape(self.hidden_size)
            y_out = y_out_blocks.reshape(self.num_blocks, 1).expand(self.num_blocks, self.block_size).reshape(self.hidden_size)
        
        # ====================================================================
        # Compute per-cell input values
        # Shape: (hidden_size,) or (batch, hidden_size)
        # ====================================================================
        net_c = matmul(self.W_c, x_t) + matmul(self.U_c, h_frozen) + self.b_c
        g_val = g_squash(net_c)
        
        # ====================================================================
        # CEC update and output
        # ====================================================================
        s_c_t = s_c_prev + (y_in * g_val)
        h_s_c = h_squash(s_c_t)
        h_t = y_out * h_s_c
        
        return h_t, s_c_t
    
    def init_state(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """Initialize hidden and cell state to zeros."""
        if batch_size == 1:
            h_0 = Tensor.zeros(self.hidden_size)
            s_c_0 = Tensor.zeros(self.hidden_size)
        else:
            h_0 = Tensor.zeros(batch_size, self.hidden_size)
            s_c_0 = Tensor.zeros(batch_size, self.hidden_size)
        return h_0, s_c_0
    
    def __call__(
        self,
        x_t: Tensor,
        h_prev: Tensor,
        s_c_prev: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Alias for forward()."""
        return self.forward(x_t, h_prev, s_c_prev)
