"""
Weight initialization schemes as specified in the 1997 LSTM paper.

The paper uses specific initialization strategies for different experiments:

1. Weight initialization ranges:
   - Experiments 1 & 2: [-0.2, 0.2]
   - Other experiments: [-0.1, 0.1]

2. Gate bias initialization (to solve the "Abuse Problem"):
   - Output gates: Negative biases (e.g., -1, -2, -3) to start closed
   - Input gates: Often negative to prevent internal state drift
   
The "Abuse Problem" (Section 4) refers to the risk of:
- Input gates always open → cell state drifts
- Output gates always open → cell outputs noise

By initializing gate biases to negative values, gates start mostly closed
and must learn to open for relevant inputs.

Reference: Sections 4 and 5 of the paper.
"""

from typing import Tuple, List, Optional, Union
import numpy as np


def init_weights_paper(
    shape: Tuple[int, ...],
    init_range: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """Initialize weights uniformly in [-init_range, init_range].
    
    Args:
        shape: Shape of the weight matrix
        init_range: Half-width of uniform distribution (default 0.1)
                   Paper uses 0.2 for experiments 1-2, 0.1 for others
        seed: Random seed for reproducibility
    
    Returns:
        Weight matrix as numpy array
    
    Paper reference:
        "weights initialized in [-0.1, 0.1]" (Section 5.4)
        "weights initialized in [-0.2, 0.2]" (Section 5.1, 5.2)
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-init_range, init_range, shape).astype(np.float32)


def init_gate_biases(
    hidden_size: int,
    num_cells: int = 1,
    gate_type: str = "output",
    bias_values: Optional[List[float]] = None
) -> np.ndarray:
    """Initialize gate biases according to the paper's recommendations.
    
    The paper recommends negative biases for gates to prevent the
    "Abuse Problem" where gates are always open.
    
    Args:
        hidden_size: Total hidden state size
        num_cells: Number of memory cell blocks (for distributing biases)
        gate_type: "input" or "output"
        bias_values: Specific bias values to use (e.g., [-1, -2, -3])
                    If None, uses paper defaults
    
    Returns:
        Bias vector as numpy array
    
    Paper reference:
        "Output gate biases: -1, -2, -3, -4" (Section 5.1)
        "Input gate biases: -3.0, -6.0" (Section 5.4)
    """
    if bias_values is None:
        if gate_type == "output":
            # Default output gate biases from paper
            bias_values = [-1.0, -2.0, -3.0, -4.0]
        else:  # input gate
            # Default input gate biases (more conservative)
            bias_values = [-2.0, -4.0, -6.0]
    
    biases = np.zeros(hidden_size, dtype=np.float32)
    
    # Handle edge case: hidden_size < len(bias_values)
    # Use round-robin assignment in this case
    if hidden_size <= len(bias_values):
        for i in range(hidden_size):
            biases[i] = bias_values[i % len(bias_values)]
    else:
        # Distribute bias values across cells/blocks
        cells_per_bias = hidden_size // len(bias_values)
        for i, bias in enumerate(bias_values):
            start = i * cells_per_bias
            end = start + cells_per_bias if i < len(bias_values) - 1 else hidden_size
            biases[start:end] = bias
    
    return biases


def init_cell_state(batch_size: int, hidden_size: int) -> np.ndarray:
    """Initialize cell state to zeros.
    
    Paper reference:
        "s_c(0) = 0" (Section 2, Equation for initial state)
    """
    return np.zeros((batch_size, hidden_size), dtype=np.float32)


def init_hidden_state(batch_size: int, hidden_size: int) -> np.ndarray:
    """Initialize hidden state to zeros.
    
    The paper doesn't explicitly state initial hidden state values,
    but zero initialization is the standard approach.
    """
    return np.zeros((batch_size, hidden_size), dtype=np.float32)


# ============================================================================
# Initialization configs for specific experiments
# ============================================================================

class InitConfig:
    """Configuration class for paper-accurate initialization."""
    
    def __init__(
        self,
        weight_range: float = 0.1,
        input_gate_biases: Optional[List[float]] = None,
        output_gate_biases: Optional[List[float]] = None,
        cell_bias: float = 0.0,
    ):
        self.weight_range = weight_range
        self.input_gate_biases = input_gate_biases or [-2.0]
        self.output_gate_biases = output_gate_biases or [-1.0, -2.0, -3.0]
        self.cell_bias = cell_bias


# Pre-defined configs matching paper experiments
INIT_CONFIGS = {
    # Experiment 1: Embedded Reber Grammar (Section 5.1)
    "reber": InitConfig(
        weight_range=0.2,
        input_gate_biases=[0.0],  # Not specified, using 0
        output_gate_biases=[-1.0, -2.0, -3.0, -4.0],
    ),
    
    # Experiment 2: Noise-free/noisy sequences (Section 5.2)
    "long_lag": InitConfig(
        weight_range=0.2,
        input_gate_biases=[0.0],
        output_gate_biases=[0.0],  # No output gates used in some variants
    ),
    
    # Experiment 3: Two-sequence problem (Section 5.3)
    "two_sequence": InitConfig(
        weight_range=0.1,
        input_gate_biases=[-1.0, -3.0, -5.0],
        output_gate_biases=[-2.0, -4.0, -6.0],
    ),
    
    # Experiment 4: Adding problem (Section 5.4)
    "adding": InitConfig(
        weight_range=0.1,
        input_gate_biases=[-3.0, -6.0],
        output_gate_biases=None,  # Random initialization
    ),
    
    # Experiment 5: Multiplication problem (Section 5.5)
    "multiplication": InitConfig(
        weight_range=0.1,
        input_gate_biases=[-3.0, -6.0],
        output_gate_biases=None,
    ),
    
    # Experiment 6: Temporal order (Section 5.6)
    "temporal_order": InitConfig(
        weight_range=0.1,
        input_gate_biases=[-2.0, -4.0],
        output_gate_biases=None,
    ),
}


def get_init_config(experiment: str) -> InitConfig:
    """Get initialization config for a specific experiment."""
    if experiment not in INIT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment}. "
                        f"Available: {list(INIT_CONFIGS.keys())}")
    return INIT_CONFIGS[experiment]
