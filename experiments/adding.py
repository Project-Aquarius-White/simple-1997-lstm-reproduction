#!/usr/bin/env python3
"""
Adding Problem - Experiment 4 (Section 5.4)

The LSTM must learn to add two marked numbers in a long sequence of random noise.

Task Description:
    - Input: Sequence of pairs (value, marker)
    - Two random positions have marker = 1.0, rest have marker = 0.0 or -1.0
    - Values are uniform random in [0, 1]
    - Target: 0.5 + (X1 + X2) / 4.0  (scaled to [0.5, 1.0])
    
Paper Criterion:
    "Absolute error at sequence end below 0.04"

Paper Hyperparameters (Section 5.4):
    - Sequence length T: 100, 500, or 1000 (random +/- 10%)
    - Learning rate: 0.5
    - Weight init: [-0.1, 0.1]
    - Input gate biases: -3.0, -6.0
    - Output gate biases: random

Reference: Section 5.4 of Hochreiter & Schmidhuber (1997)
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquarius_lstm.metrics import paper_accuracy_criterion, ExperimentResult


def generate_adding_data(
    seq_len: int = 100,
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Adding Problem dataset as specified in Section 5.4.
    
    Args:
        seq_len: Base sequence length (actual length varies +/- 10%)
        num_samples: Number of sequences to generate
        seed: Random seed for reproducibility
    
    Returns:
        X: Input sequences, shape (num_samples, max_len, 2)
           - Column 0: Values in [0, 1] (uniform random)
           - Column 1: Markers (two 1.0s at marked positions, 0.0 elsewhere)
        Y: Targets, shape (num_samples, 1)
           - Target = 0.5 + (X1 + X2) / 4.0
    
    Paper reference (Section 5.4):
        "Each input sequence is of length T randomly chosen between T and
        T + T/10... target at sequence end is 0.5 + (X1 + X2)/4.0"
        
    Note: The paper's adding problem uses values in [0, 1] and marker
    baseline of 0.0. The scaled target ensures output is in [0.5, 1.0].
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Actual sequence lengths vary by +/- 10% as per paper
    min_len = max(1, int(seq_len * 0.9))
    max_len = int(seq_len * 1.1) + 1
    actual_lengths = np.random.randint(min_len, max_len, num_samples)
    
    X = np.zeros((num_samples, max_len, 2), dtype=np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(num_samples):
        actual_len = actual_lengths[i]
        
        # Stream 1: Random values in [0, 1]
        X[i, :actual_len, 0] = np.random.uniform(0, 1, actual_len)
        
        # Stream 2: Markers (two 1.0s at random positions)
        # According to paper: first marker in first half, second in second half
        first_half = actual_len // 2
        pos1 = np.random.randint(0, first_half)
        pos2 = np.random.randint(first_half, actual_len)
        
        X[i, pos1, 1] = 1.0
        X[i, pos2, 1] = 1.0
        
        # Target: scaled sum as per paper
        # Y = 0.5 + (X1 + X2) / 4.0
        X1 = X[i, pos1, 0]
        X2 = X[i, pos2, 0]
        Y[i] = 0.5 + (X1 + X2) / 4.0
    
    return X, Y


def run_tinygrad(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 10,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    realize_every: int = 10,
    log_every: int = 10,
) -> ExperimentResult:
    """Run the Adding Problem experiment with tinygrad."""
    from tinygrad.tensor import Tensor
    from aquarius_lstm.cell import LSTMCell1997
    from aquarius_lstm.initialization import init_weights_paper
    
    print(f"\n{'='*60}")
    print("ADDING PROBLEM (Section 5.4) - tinygrad")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Generate data
    X, Y = generate_adding_data(seq_len, num_samples, seed)
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y)
    
    # Initialize model with paper-accurate settings
    cell = LSTMCell1997(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=-3.0,  # Paper: -3.0, -6.0 for different cells
        output_gate_bias=0.0,
        seed=seed,
    )
    
    # Output projection
    W_out = Tensor(
        init_weights_paper((1, hidden_size), 0.1, seed),
        requires_grad=True
    )
    b_out = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    
    params = cell.parameters() + [W_out, b_out]
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for sample_idx in range(num_samples):
            # Initialize state
            h, s_c = cell.init_state()
            
            # Forward through sequence
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
                
                # Realize periodically to prevent graph explosion
                if t % realize_every == 0:
                    h = h.realize()
                    s_c = s_c.realize()
            
            # Compute prediction and loss
            pred = W_out.dot(h) + b_out
            target = Y_tensor[sample_idx]
            loss = (pred - target).pow(2).sum()
            
            # Backward and update
            loss.backward()
            
            for p in params:
                if p.grad is not None:
                    p.assign(p - p.grad * learning_rate).realize()
                    p.grad = None
            
            epoch_loss += float(loss.numpy())
        
        avg_loss = epoch_loss / num_samples
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Evaluate on all samples
    predictions = []
    for sample_idx in range(num_samples):
        h, s_c = cell.init_state()
        for t in range(X.shape[1]):
            x_t = X_tensor[sample_idx, t]
            h, s_c = cell(x_t, h, s_c)
        pred = W_out.dot(h) + b_out
        predictions.append(float(pred.numpy()))
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Evaluate against paper criterion
    result = paper_accuracy_criterion("adding", predictions, Y)
    print(f"\n{result}")
    
    return result


def run_torch(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 10,
    learning_rate: float = 0.5,
    seed: Optional[int] = 42,
    log_every: int = 10,
) -> ExperimentResult:
    """Run the Adding Problem experiment with PyTorch."""
    import torch
    from aquarius_lstm.cell_torch import LSTMCell1997Torch
    
    print(f"\n{'='*60}")
    print("ADDING PROBLEM (Section 5.4) - PyTorch")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate data
    X, Y = generate_adding_data(seq_len, num_samples, seed)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    
    # Initialize model
    cell = LSTMCell1997Torch(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=-3.0,
        seed=seed,
    )
    
    W_out = torch.nn.Parameter(torch.randn(1, hidden_size) * 0.1)
    b_out = torch.nn.Parameter(torch.zeros(1))
    
    params = list(cell.parameters()) + [W_out, b_out]
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for sample_idx in range(num_samples):
            optimizer.zero_grad()
            
            h, s_c = cell.init_state()
            
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            
            pred = h @ W_out.T + b_out
            target = Y_tensor[sample_idx]
            loss = ((pred - target) ** 2).sum()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_samples
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    # Evaluate
    predictions = []
    with torch.no_grad():
        for sample_idx in range(num_samples):
            h, s_c = cell.init_state()
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
            pred = h @ W_out.T + b_out
            predictions.append(pred.item())
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    result = paper_accuracy_criterion("adding", predictions, Y)
    print(f"\n{result}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Adding Problem (Section 5.4)")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke",
                       help="smoke: quick test, paper: full reproduction")
    parser.add_argument("--backend", choices=["tinygrad", "torch", "both"], 
                       default="tinygrad")
    parser.add_argument("--seq-len", type=int, default=None,
                       help="Override sequence length")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.mode == "smoke":
        params = {
            "seq_len": args.seq_len or 50,
            "num_samples": 100,
            "num_epochs": 30,
            "hidden_size": 10,
            "learning_rate": 0.5,
        }
    else:  # paper mode
        params = {
            "seq_len": args.seq_len or 100,  # Paper uses 100, 500, 1000
            "num_samples": 1000,
            "num_epochs": 100,
            "hidden_size": 10,
            "learning_rate": 0.5,
        }
    
    params["seed"] = args.seed
    
    results = []
    
    if args.backend in ["tinygrad", "both"]:
        try:
            result = run_tinygrad(**params)
            results.append(result)
        except ImportError as e:
            print(f"tinygrad not available: {e}")
    
    if args.backend in ["torch", "both"]:
        try:
            result = run_torch(**params)
            results.append(result)
        except ImportError as e:
            print(f"PyTorch not available: {e}")
    
    # Return exit code based on pass/fail
    passed = all(r.passed for r in results) if results else False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
