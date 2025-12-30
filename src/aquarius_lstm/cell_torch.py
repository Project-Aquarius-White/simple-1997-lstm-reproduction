"""
Original 1997 LSTM Memory Cell implementation in PyTorch.

This is a verification implementation that mirrors the tinygrad version
exactly. Use this to cross-check results and for compatibility with
PyTorch-based workflows.

See cell.py for the canonical implementation and detailed documentation.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np

from .activations import g_squash_torch, h_squash_torch
from .initialization import init_weights_paper, init_gate_biases


class LSTMCell1997Torch(nn.Module):
    """
    PyTorch implementation of the 1997 LSTM Memory Cell.
    
    Implements the exact same equations as the tinygrad version for
    cross-validation. See cell.py for detailed documentation.
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
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Input Gate
        self.W_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_in = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_in = nn.Parameter(torch.full(
            (hidden_size,), input_gate_bias, dtype=torch.float32
        ))
        
        # Output Gate
        self.W_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_out = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_out = nn.Parameter(torch.full(
            (hidden_size,), output_gate_bias, dtype=torch.float32
        ))
        
        # Cell Input
        self.W_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, input_size), init_range)
        ))
        self.U_c = nn.Parameter(torch.tensor(
            init_weights_paper((hidden_size, hidden_size), init_range)
        ))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
    
    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        s_c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one timestep with truncated backprop.
        
        Args:
            x_t: Input, shape (input_size,) or (batch, input_size)
            h_prev: Previous hidden state
            s_c_prev: Previous cell state
        
        Returns:
            h_t: Current hidden state
            s_c_t: Current cell state
        """
        # THE SCISSORS: Truncate gradient through gates' recurrent path
        h_frozen = h_prev.detach()
        
        # Handle both 1D and 2D inputs
        is_batched = x_t.dim() == 2
        
        if is_batched:
            # (batch, input_size) @ (input_size, hidden_size) -> (batch, hidden_size)
            net_in = x_t @ self.W_in.T + h_frozen @ self.U_in.T + self.b_in
            net_out = x_t @ self.W_out.T + h_frozen @ self.U_out.T + self.b_out
            net_c = x_t @ self.W_c.T + h_frozen @ self.U_c.T + self.b_c
        else:
            # 1D case
            net_in = self.W_in @ x_t + self.U_in @ h_frozen + self.b_in
            net_out = self.W_out @ x_t + self.U_out @ h_frozen + self.b_out
            net_c = self.W_c @ x_t + self.U_c @ h_frozen + self.b_c
        
        # Gates and activations
        y_in = torch.sigmoid(net_in)
        y_out = torch.sigmoid(net_out)
        g_val = g_squash_torch(net_c)
        
        # CEC update (NO detach on s_c_prev!)
        s_c_t = s_c_prev + (y_in * g_val)
        
        # Output
        h_s_c = h_squash_torch(s_c_t)
        h_t = y_out * h_s_c
        
        return h_t, s_c_t
    
    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell state to zeros."""
        device = device or self.W_in.device
        
        if batch_size == 1:
            h_0 = torch.zeros(self.hidden_size, device=device)
            s_c_0 = torch.zeros(self.hidden_size, device=device)
        else:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            s_c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        
        return h_0, s_c_0


class LSTMModel1997Torch(nn.Module):
    """
    PyTorch LSTM model wrapper for sequence tasks.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        **cell_kwargs,
    ):
        super().__init__()
        
        self.cell = LSTMCell1997Torch(input_size, hidden_size, **cell_kwargs)
        self.output_proj = nn.Linear(hidden_size, output_size, bias=True)
        
        # Initialize output projection with paper-accurate range
        with torch.no_grad():
            self.output_proj.weight.copy_(torch.tensor(
                init_weights_paper((output_size, hidden_size), 0.1)
            ))
            self.output_proj.bias.zero_()
    
    def forward_sequence(
        self,
        x_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a full sequence and return the final prediction.
        
        Args:
            x_seq: Input sequence, shape (seq_len, input_size) or (seq_len, batch, input_size)
        
        Returns:
            prediction: Final output
            h_final: Final hidden state
            s_c_final: Final cell state
        """
        seq_len = x_seq.shape[0]
        batch_size = x_seq.shape[1] if x_seq.dim() == 3 else 1
        
        h, s_c = self.cell.init_state(batch_size, device=x_seq.device)
        
        for t in range(seq_len):
            x_t = x_seq[t]
            h, s_c = self.cell(x_t, h, s_c)
        
        pred = self.output_proj(h)
        
        return pred, h, s_c
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Standard forward that returns just the prediction."""
        pred, _, _ = self.forward_sequence(x_seq)
        return pred
