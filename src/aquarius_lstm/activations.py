"""
Activation functions as specified in the 1997 LSTM paper.

The paper uses specific activation function ranges for different components:
- Gates use standard logistic sigmoid: f(x) = 1/(1+e^-x) in [0, 1]
- Input squashing g(x) = 4*sigmoid(x) - 2 in [-2, 2]  (Appendix A.2)
- Output squashing h(x) = 2*sigmoid(x) - 1 in [-1, 1]  (Appendix A.3)

These ranges are critical for the Constant Error Carousel (CEC) to work properly.
The paper specifically notes that g and h are centered around 0 to allow both
positive and negative values in the cell state.

Reference: Section 4 and Appendix A of the paper.
"""

from typing import Union, TYPE_CHECKING
import numpy as np

# Type hints for both frameworks
if TYPE_CHECKING:
    from tinygrad.tensor import Tensor as TinyTensor
    import torch
    TensorType = Union[TinyTensor, torch.Tensor, np.ndarray]
else:
    TensorType = Union[np.ndarray, "TinyTensor", "torch.Tensor"]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid using numpy.
    
    f(x) = 1 / (1 + e^(-x))
    Range: [0, 1]
    
    Used for gate activations (input gate, output gate).
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def g_squash_np(x: np.ndarray) -> np.ndarray:
    """Input squashing function g(x) as per Appendix A.2.
    
    g(x) = 4 * sigmoid(x) - 2
    Range: [-2, 2]
    
    This function squashes the input to the memory cell before it is
    gated by the input gate. The range [-2, 2] allows for both positive
    and negative contributions to the cell state.
    
    Paper quote: "g is a logistic sigmoid in [-2, 2]"
    """
    return 4.0 * sigmoid_np(x) - 2.0


def h_squash_np(x: np.ndarray) -> np.ndarray:
    """Output squashing function h(x) as per Appendix A.3.
    
    h(x) = 2 * sigmoid(x) - 1
    Range: [-1, 1]
    
    This function squashes the cell state before it is gated by the
    output gate. The range [-1, 1] normalizes the output.
    
    Paper quote: "h is a logistic sigmoid in [-1, 1]"
    """
    return 2.0 * sigmoid_np(x) - 1.0


# ============================================================================
# Tinygrad implementations
# ============================================================================

def sigmoid(x: "TinyTensor") -> "TinyTensor":
    """Standard logistic sigmoid for tinygrad tensors.
    
    Uses tinygrad's built-in sigmoid for efficiency.
    """
    return x.sigmoid()


def g_squash(x: "TinyTensor") -> "TinyTensor":
    """Input squashing function g(x) for tinygrad.
    
    g(x) = 4 * sigmoid(x) - 2
    Range: [-2, 2]
    """
    return x.sigmoid() * 4.0 - 2.0


def h_squash(x: "TinyTensor") -> "TinyTensor":
    """Output squashing function h(x) for tinygrad.
    
    h(x) = 2 * sigmoid(x) - 1
    Range: [-1, 1]
    """
    return x.sigmoid() * 2.0 - 1.0


# ============================================================================
# PyTorch implementations
# ============================================================================

def sigmoid_torch(x: "torch.Tensor") -> "torch.Tensor":
    """Standard logistic sigmoid for PyTorch tensors."""
    import torch
    return torch.sigmoid(x)


def g_squash_torch(x: "torch.Tensor") -> "torch.Tensor":
    """Input squashing function g(x) for PyTorch.
    
    g(x) = 4 * sigmoid(x) - 2
    Range: [-2, 2]
    """
    import torch
    return torch.sigmoid(x) * 4.0 - 2.0


def h_squash_torch(x: "torch.Tensor") -> "torch.Tensor":
    """Output squashing function h(x) for PyTorch.
    
    h(x) = 2 * sigmoid(x) - 1
    Range: [-1, 1]
    """
    import torch
    return torch.sigmoid(x) * 2.0 - 1.0


# ============================================================================
# Derivatives (for reference / manual backprop if needed)
# ============================================================================

def sigmoid_derivative_np(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))"""
    s = sigmoid_np(x)
    return s * (1.0 - s)


def g_squash_derivative_np(x: np.ndarray) -> np.ndarray:
    """Derivative of g: g'(x) = 4 * sigmoid'(x) = 4 * sigmoid(x) * (1 - sigmoid(x))"""
    return 4.0 * sigmoid_derivative_np(x)


def h_squash_derivative_np(x: np.ndarray) -> np.ndarray:
    """Derivative of h: h'(x) = 2 * sigmoid'(x) = 2 * sigmoid(x) * (1 - sigmoid(x))"""
    return 2.0 * sigmoid_derivative_np(x)
