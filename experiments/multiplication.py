#!/usr/bin/env python3
"""
Multiplication Problem - Experiment 5 (Section 5.5)

Similar to the Adding Problem, but the target is the product X1 * X2.

Task Description:
    - Input: Same as Adding Problem (value, marker pairs)
    - Target: X1 * X2 (product of marked values)
    
Paper Criterion:
    "Absolute error below 0.04"

Paper Hyperparameters (Section 5.5):
    - Sequence length T: 100
    - Learning rate: 0.1
    - Other settings same as Adding Problem

Reference: Section 5.5 of Hochreiter & Schmidhuber (1997)
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aquarius_lstm.metrics import paper_accuracy_criterion, ExperimentResult


def generate_multiplication_data(
    seq_len: int = 100,
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Multiplication Problem dataset (Section 5.5).
    
    Same as Adding Problem, but target is X1 * X2 instead of sum.
    Values are in [0, 1], so product is also in [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Actual sequence lengths vary by +/- 10%
    min_len = max(1, int(seq_len * 0.9))
    max_len = int(seq_len * 1.1) + 1
    actual_lengths = np.random.randint(min_len, max_len, num_samples)
    
    X = np.zeros((num_samples, max_len, 2), dtype=np.float32)
    Y = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(num_samples):
        actual_len = actual_lengths[i]
        
        X[i, :actual_len, 0] = np.random.uniform(0, 1, actual_len)
        
        first_half = actual_len // 2
        pos1 = np.random.randint(0, first_half)
        pos2 = np.random.randint(first_half, actual_len)
        
        X[i, pos1, 1] = 1.0
        X[i, pos2, 1] = 1.0
        
        # Target: product
        X1 = X[i, pos1, 0]
        X2 = X[i, pos2, 0]
        Y[i] = X1 * X2
    
    return X, Y


def run_tinygrad(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 10,
    learning_rate: float = 0.1,  # Paper uses 0.1 for multiplication
    seed: Optional[int] = 42,
    realize_every: int = 10,
    log_every: int = 10,
) -> ExperimentResult:
    """Run Multiplication Problem with tinygrad."""
    from tinygrad.tensor import Tensor
    from aquarius_lstm.cell import LSTMCell1997
    from aquarius_lstm.initialization import init_weights_paper
    
    print(f"\n{'='*60}")
    print("MULTIPLICATION PROBLEM (Section 5.5) - tinygrad")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Samples: {num_samples}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    X, Y = generate_multiplication_data(seq_len, num_samples, seed)
    X_tensor = Tensor(X)
    Y_tensor = Tensor(Y)
    
    cell = LSTMCell1997(
        input_size=2,
        hidden_size=hidden_size,
        init_range=0.1,
        input_gate_bias=-3.0,
        seed=seed,
    )
    
    W_out = Tensor(
        init_weights_paper((1, hidden_size), 0.1, seed),
        requires_grad=True
    )
    b_out = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    
    params = cell.parameters() + [W_out, b_out]
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for sample_idx in range(num_samples):
            h, s_c = cell.init_state()
            
            for t in range(X.shape[1]):
                x_t = X_tensor[sample_idx, t]
                h, s_c = cell(x_t, h, s_c)
                if t % realize_every == 0:
                    h = h.realize()
                    s_c = s_c.realize()
            
            pred = W_out.dot(h) + b_out
            target = Y_tensor[sample_idx]
            loss = (pred - target).pow(2).sum()
            
            loss.backward()
            
            for p in params:
                if p.grad is not None:
                    p.assign(p - p.grad * learning_rate).realize()
                    p.grad = None
            
            epoch_loss += float(loss.numpy())
        
        avg_loss = epoch_loss / num_samples
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
    
    predictions = []
    for sample_idx in range(num_samples):
        h, s_c = cell.init_state()
        for t in range(X.shape[1]):
            x_t = X_tensor[sample_idx, t]
            h, s_c = cell(x_t, h, s_c)
        pred = W_out.dot(h) + b_out
        predictions.append(float(pred.numpy()))
    
    predictions = np.array(predictions).reshape(-1, 1)
    
    result = paper_accuracy_criterion("multiplication", predictions, Y)
    print(f"\n{result}")
    
    return result


def run_torch(
    seq_len: int = 100,
    num_samples: int = 1000,
    num_epochs: int = 100,
    hidden_size: int = 10,
    learning_rate: float = 0.1,
    seed: Optional[int] = 42,
    log_every: int = 10,
) -> ExperimentResult:
    """Run Multiplication Problem with PyTorch."""
    import torch
    from aquarius_lstm.cell_torch import LSTMCell1997Torch
    
    print(f"\n{'='*60}")
    print("MULTIPLICATION PROBLEM (Section 5.5) - PyTorch")
    print(f"{'='*60}")
    
    if seed is not None:
        torch.manual_seed(seed)
    
    X, Y = generate_multiplication_data(seq_len, num_samples, seed)
    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    
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
        
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {epoch_loss/num_samples:.6f}")
    
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
    
    result = paper_accuracy_criterion("multiplication", predictions, Y)
    print(f"\n{result}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Multiplication Problem (Section 5.5)")
    parser.add_argument("--mode", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--backend", choices=["tinygrad", "torch", "both"], default="tinygrad")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.mode == "smoke":
        params = {
            "seq_len": 50,
            "num_samples": 100,
            "num_epochs": 30,
            "hidden_size": 10,
            "learning_rate": 0.1,
        }
    else:
        params = {
            "seq_len": 100,
            "num_samples": 1000,
            "num_epochs": 100,
            "hidden_size": 10,
            "learning_rate": 0.1,
        }
    
    params["seed"] = args.seed
    
    results = []
    
    if args.backend in ["tinygrad", "both"]:
        try:
            results.append(run_tinygrad(**params))
        except ImportError as e:
            print(f"tinygrad not available: {e}")
    
    if args.backend in ["torch", "both"]:
        try:
            results.append(run_torch(**params))
        except ImportError as e:
            print(f"PyTorch not available: {e}")
    
    passed = all(r.passed for r in results) if results else False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
