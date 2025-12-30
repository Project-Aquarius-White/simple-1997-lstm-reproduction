"""
Original 1997 LSTM Memory Cell implementation in tinygrad.

This implements the LSTM architecture exactly as described in:
"Long Short-Term Memory" - Hochreiter & Schmidhuber (1997)

Key characteristics of the 1997 LSTM:
1. NO forget gate (added in later variants like Gers et al. 2000)
2. Constant Error Carousel (CEC) with weight 1.0 self-connection
3. Input gate and Output gate only
4. Truncated backpropagation through gates (O(1) complexity per timestep)

The forward equations are:

    y_in(t) = sigmoid(W_in @ x(t) + U_in @ y(t-1) + b_in)     # Input gate
    y_out(t) = sigmoid(W_out @ x(t) + U_out @ y(t-1) + b_out) # Output gate
    net_c(t) = W_c @ x(t) + U_c @ y(t-1) + b_c                # Cell input
    s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t))                 # Cell state (CEC)
    y_c(t) = y_out(t) * h(s_c(t))                              # Cell output

Where:
    - g(x) = 4 * sigmoid(x) - 2  (range [-2, 2])
    - h(x) = 2 * sigmoid(x) - 1  (range [-1, 1])

Truncation strategy (for O(1) complexity):
    - The recurrent connection y(t-1) is DETACHED before entering gate computations
    - The cell state s_c path is NOT detached (gradient flows through the CEC)
    - This allows error to propagate through the "1.0 tunnel" while preventing
      explosion through the multiplicative gates

Reference: Paper Section 2-4, Appendix A
"""

from typing import Tuple, Optional, List
from tinygrad.tensor import Tensor
import numpy as np

from .activations import g_squash, h_squash
from .initialization import init_weights_paper, init_gate_biases, get_init_config


class LSTMCell1997:
    """
    A single 1997 LSTM Memory Cell (no forget gate).
    
    This class implements one memory cell with its own input gate and output gate.
    The paper also describes "blocks" where multiple cells share gates - see
    LSTMBlock1997 for that variant.
    
    Attributes:
        input_size: Dimension of input vector x(t)
        hidden_size: Dimension of hidden state / number of memory cells
        
    Paper notation mapping:
        - y^in_j → y_in (input gate activation)
        - y^out_j → y_out (output gate activation)  
        - s_c_j → s_c (internal cell state)
        - y^c_j → y_c (cell output / hidden state)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_range: float = 0.1,
        input_gate_bias: float = -2.0,
        output_gate_bias: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LSTM cell with paper-accurate settings.
        
        Args:
            input_size: Size of input vector
            hidden_size: Size of hidden state (number of memory units)
            init_range: Weight initialization range [-init_range, init_range]
            input_gate_bias: Initial bias for input gate (negative = starts closed)
            output_gate_bias: Initial bias for output gate
            seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # ====================================================================
        # Input Gate weights (y^in in paper)
        # Equation: y_in = sigmoid(W_in @ x + U_in @ h_prev + b_in)
        # ====================================================================
        self.W_in = Tensor(
            init_weights_paper((hidden_size, input_size), init_range),
            requires_grad=True
        )
        self.U_in = Tensor(
            init_weights_paper((hidden_size, hidden_size), init_range),
            requires_grad=True
        )
        # Negative bias to start "closed" (prevents abuse problem)
        self.b_in = Tensor(
            np.full(hidden_size, input_gate_bias, dtype=np.float32),
            requires_grad=True
        )
        
        # ====================================================================
        # Output Gate weights (y^out in paper)
        # Equation: y_out = sigmoid(W_out @ x + U_out @ h_prev + b_out)
        # ====================================================================
        self.W_out = Tensor(
            init_weights_paper((hidden_size, input_size), init_range),
            requires_grad=True
        )
        self.U_out = Tensor(
            init_weights_paper((hidden_size, hidden_size), init_range),
            requires_grad=True
        )
        self.b_out = Tensor(
            np.full(hidden_size, output_gate_bias, dtype=np.float32),
            requires_grad=True
        )
        
        # ====================================================================
        # Cell Input weights (net_c in paper, feeds into g())
        # Equation: net_c = W_c @ x + U_c @ h_prev + b_c
        # ====================================================================
        self.W_c = Tensor(
            init_weights_paper((hidden_size, input_size), init_range),
            requires_grad=True
        )
        self.U_c = Tensor(
            init_weights_paper((hidden_size, hidden_size), init_range),
            requires_grad=True
        )
        self.b_c = Tensor(
            np.zeros(hidden_size, dtype=np.float32),
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
        
        This implements the core LSTM equations with truncated backprop:
        - Gates receive DETACHED hidden state (truncation)
        - Cell state path is NOT detached (gradient tunnel)
        
        Args:
            x_t: Input at current timestep, shape (input_size,) or (batch, input_size)
            h_prev: Previous hidden state (cell output), shape (hidden_size,) or (batch, hidden_size)
            s_c_prev: Previous cell state, shape (hidden_size,) or (batch, hidden_size)
        
        Returns:
            h_t: Current hidden state (cell output)
            s_c_t: Current cell state
            
        Paper reference:
            Equations in Section 2-3, truncation strategy in Section 4
        """
        # ====================================================================
        # THE SCISSORS (Truncated Backpropagation)
        # ====================================================================
        # We DETACH the recurrent signal before it enters the gate computations.
        # This enforces O(1) complexity per timestep because we don't backprop
        # through the gate's recurrent connections across time.
        #
        # CRITICAL: We do NOT detach s_c_prev - the CEC gradient tunnel stays open!
        # This is the key insight of the 1997 paper.
        # ====================================================================
        h_frozen = h_prev.detach()
        
        # Handle both 1D (single sample) and 2D (batch) inputs
        is_batched = len(x_t.shape) == 2
        
        if is_batched:
            # Batched: (batch, input_size) @ (input_size, hidden_size).T
            # = (batch, hidden_size)
            def matmul(W, x):
                return x.matmul(W.T)
        else:
            # Single sample: use dot product
            def matmul(W, x):
                return W.dot(x)
        
        # ====================================================================
        # 1. Input Gate: y_in = sigmoid(net_in)
        # ====================================================================
        net_in = matmul(self.W_in, x_t) + matmul(self.U_in, h_frozen) + self.b_in
        y_in = net_in.sigmoid()
        
        # ====================================================================
        # 2. Output Gate: y_out = sigmoid(net_out)
        # ====================================================================
        net_out = matmul(self.W_out, x_t) + matmul(self.U_out, h_frozen) + self.b_out
        y_out = net_out.sigmoid()
        
        # ====================================================================
        # 3. Cell Input: g(net_c) where g ∈ [-2, 2]
        # ====================================================================
        net_c = matmul(self.W_c, x_t) + matmul(self.U_c, h_frozen) + self.b_c
        g_val = g_squash(net_c)  # 4 * sigmoid(net_c) - 2
        
        # ====================================================================
        # 4. Cell State Update (Constant Error Carousel)
        # ====================================================================
        # s_c(t) = s_c(t-1) + y_in(t) * g(net_c(t))
        #
        # NOTE: s_c_prev is NOT detached! This is the "1.0 tunnel" that allows
        # gradients to flow back through time without vanishing/exploding.
        # The self-connection weight is implicitly 1.0 (just addition).
        # ====================================================================
        s_c_t = s_c_prev + (y_in * g_val)
        
        # ====================================================================
        # 5. Cell Output: y_c = y_out * h(s_c) where h ∈ [-1, 1]
        # ====================================================================
        h_s_c = h_squash(s_c_t)  # 2 * sigmoid(s_c) - 1
        h_t = y_out * h_s_c
        
        return h_t, s_c_t
    
    def init_state(self, batch_size: int = 1) -> Tuple[Tensor, Tensor]:
        """
        Initialize hidden state and cell state to zeros.
        
        Paper reference: "s_c(0) = 0" (Section 2)
        
        Args:
            batch_size: Number of sequences in batch
        
        Returns:
            h_0: Initial hidden state (zeros)
            s_c_0: Initial cell state (zeros)
        """
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
        """Alias for forward() to allow cell(x, h, s) syntax."""
        return self.forward(x_t, h_prev, s_c_prev)


class LSTMModel1997:
    """
    A simple LSTM model wrapper for sequence tasks.
    
    Wraps an LSTMCell1997 and adds an output projection layer
    for prediction tasks.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        **cell_kwargs,
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden state
            output_size: Size of output (prediction dimension)
            **cell_kwargs: Additional arguments passed to LSTMCell1997
        """
        self.cell = LSTMCell1997(input_size, hidden_size, **cell_kwargs)
        
        # Output projection layer
        self.W_out_proj = Tensor(
            init_weights_paper((output_size, hidden_size), 0.1),
            requires_grad=True
        )
        self.b_out_proj = Tensor(
            np.zeros(output_size, dtype=np.float32),
            requires_grad=True
        )
    
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        return self.cell.parameters() + [self.W_out_proj, self.b_out_proj]
    
    def forward_sequence(
        self,
        x_seq: Tensor,
        realize_every: int = 10,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Process a full sequence and return the final prediction.
        
        Args:
            x_seq: Input sequence, shape (seq_len, input_size) or (seq_len, batch, input_size)
            realize_every: Call .realize() every N steps to prevent graph explosion
        
        Returns:
            prediction: Final output prediction
            h_final: Final hidden state
            s_c_final: Final cell state
        """
        seq_len = x_seq.shape[0]
        
        # Detect batch dimension
        if len(x_seq.shape) == 3:
            batch_size = x_seq.shape[1]
        else:
            batch_size = 1
        
        h, s_c = self.cell.init_state(batch_size)
        
        for t in range(seq_len):
            x_t = x_seq[t]
            h, s_c = self.cell(x_t, h, s_c)
            
            # Periodic realization to prevent computational graph explosion
            # This is a tinygrad optimization - does NOT break gradient flow
            if t > 0 and t % realize_every == 0:
                h = h.realize()
                s_c = s_c.realize()
        
        # Final prediction
        if batch_size == 1:
            pred = self.W_out_proj.dot(h) + self.b_out_proj
        else:
            pred = h.matmul(self.W_out_proj.T) + self.b_out_proj
        
        return pred, h, s_c
