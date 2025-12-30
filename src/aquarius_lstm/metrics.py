"""
Evaluation metrics and success criteria from the 1997 LSTM paper.

Each experiment in the paper has specific success criteria. This module
provides functions to evaluate model performance against those criteria.

Reference: Section 5 of the paper describes all experiments and their
success conditions.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ExperimentResult:
    """Container for experiment evaluation results."""
    experiment_name: str
    metric_name: str
    achieved_value: float
    threshold: float
    passed: bool
    details: Optional[str] = None
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"[{status}] {self.experiment_name}: "
                f"{self.metric_name} = {self.achieved_value:.6f} "
                f"(threshold: {self.threshold})")


def paper_accuracy_criterion(
    experiment: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    **kwargs
) -> ExperimentResult:
    """Evaluate predictions against paper-specified criteria.
    
    Args:
        experiment: Name of the experiment (e.g., "adding", "reber")
        predictions: Model predictions
        targets: Ground truth targets
        **kwargs: Additional experiment-specific parameters
    
    Returns:
        ExperimentResult with pass/fail status
    """
    evaluators = {
        "adding": _evaluate_adding,
        "multiplication": _evaluate_multiplication,
        "temporal_order": _evaluate_temporal_order,
        "temporal_order_3": _evaluate_temporal_order_3,
        "reber": _evaluate_reber,
        "long_lag": _evaluate_long_lag,
        "two_sequence": _evaluate_two_sequence,
    }
    
    if experiment not in evaluators:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    return evaluators[experiment](predictions, targets, **kwargs)


def _evaluate_adding(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.04,
    **kwargs
) -> ExperimentResult:
    """Evaluate Adding Problem (Section 5.4).
    
    Paper criterion: "absolute error at sequence end below 0.04"
    
    Note: Paper uses scaled target Y = 0.5 + (X1 + X2) / 4.0
    so the raw sum values are in [0, 2] mapped to [0.5, 1.0]
    """
    # Compute absolute errors at sequence end
    abs_errors = np.abs(predictions.flatten() - targets.flatten())
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    
    passed = max_error < threshold
    
    return ExperimentResult(
        experiment_name="Adding Problem",
        metric_name="max_absolute_error",
        achieved_value=max_error,
        threshold=threshold,
        passed=passed,
        details=f"mean_error={mean_error:.6f}, max_error={max_error:.6f}"
    )


def _evaluate_multiplication(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.04,
    **kwargs
) -> ExperimentResult:
    """Evaluate Multiplication Problem (Section 5.5).
    
    Paper criterion: "absolute error below 0.04"
    Same as adding problem but target is X1 * X2.
    """
    abs_errors = np.abs(predictions.flatten() - targets.flatten())
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    
    passed = max_error < threshold
    
    return ExperimentResult(
        experiment_name="Multiplication Problem",
        metric_name="max_absolute_error",
        achieved_value=max_error,
        threshold=threshold,
        passed=passed,
        details=f"mean_error={mean_error:.6f}"
    )


def _evaluate_temporal_order(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.3,
    **kwargs
) -> ExperimentResult:
    """Evaluate Temporal Order - 2 symbols (Section 5.6.1).
    
    Paper criterion: "final absolute error of all output units below 0.3"
    
    This is a 4-class classification (XX, XY, YX, YY).
    """
    # For classification, compute max absolute error across all output units
    abs_errors = np.abs(predictions - targets)
    max_error = np.max(abs_errors)
    
    # Also compute classification accuracy
    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(targets, axis=-1)
    accuracy = np.mean(pred_classes == true_classes)
    
    passed = max_error < threshold
    
    return ExperimentResult(
        experiment_name="Temporal Order (2 symbols)",
        metric_name="max_absolute_error",
        achieved_value=max_error,
        threshold=threshold,
        passed=passed,
        details=f"accuracy={accuracy:.2%}"
    )


def _evaluate_temporal_order_3(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.3,
    **kwargs
) -> ExperimentResult:
    """Evaluate Temporal Order - 3 symbols (Section 5.6.2).
    
    Same as 2-symbol but with 8 classes (XXX through YYY).
    """
    abs_errors = np.abs(predictions - targets)
    max_error = np.max(abs_errors)
    
    pred_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(targets, axis=-1)
    accuracy = np.mean(pred_classes == true_classes)
    
    passed = max_error < threshold
    
    return ExperimentResult(
        experiment_name="Temporal Order (3 symbols)",
        metric_name="max_absolute_error",
        achieved_value=max_error,
        threshold=threshold,
        passed=passed,
        details=f"accuracy={accuracy:.2%}"
    )


def _evaluate_reber(
    predictions: np.ndarray,
    targets: np.ndarray,
    **kwargs
) -> ExperimentResult:
    """Evaluate Embedded Reber Grammar (Section 5.1).
    
    Paper criterion: "all string symbols in both test and training sets
    are predicted correctly (most active output unit corresponds to the
    possible next symbol)"
    """
    # For each timestep, the most active output should be a valid next symbol
    pred_symbols = np.argmax(predictions, axis=-1)
    true_symbols = np.argmax(targets, axis=-1)
    
    # The paper allows for multiple valid next symbols at each position
    # For simplicity, we check if prediction matches the actual next symbol
    correct = (pred_symbols == true_symbols)
    accuracy = np.mean(correct)
    
    # Paper requires 100% accuracy
    passed = accuracy == 1.0
    
    return ExperimentResult(
        experiment_name="Embedded Reber Grammar",
        metric_name="symbol_accuracy",
        achieved_value=accuracy,
        threshold=1.0,
        passed=passed,
        details=f"Correct predictions: {np.sum(correct)}/{len(correct)}"
    )


def _evaluate_long_lag(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.25,
    num_sequences: int = 10000,
    **kwargs
) -> ExperimentResult:
    """Evaluate Long Time Lag tasks (Section 5.2).
    
    Paper criterion: "maximal absolute error of all output units below 0.25
    over 10,000 successive sequences"
    """
    abs_errors = np.abs(predictions - targets)
    max_error = np.max(abs_errors)
    
    passed = max_error < threshold
    
    return ExperimentResult(
        experiment_name="Long Time Lag",
        metric_name="max_absolute_error",
        achieved_value=float(max_error),
        threshold=threshold,
        passed=bool(passed),
        details=f"Over {len(predictions)} sequences"
    )


def _evaluate_two_sequence(
    predictions: np.ndarray,
    targets: np.ndarray,
    criterion: str = "ST1",
    **kwargs
) -> ExperimentResult:
    """Evaluate Two-Sequence Problem (Section 5.3).
    
    Paper criteria:
    - ST1: "none of 256 test sequences misclassified"
    - ST2: "mean absolute test error below 0.01"
    """
    if criterion == "ST1":
        # Binary classification - check for misclassifications
        pred_class = (predictions > 0.5).astype(int)
        true_class = (targets > 0.5).astype(int)
        misclassified = np.sum(pred_class != true_class)
        
        passed = misclassified == 0
        
        return ExperimentResult(
            experiment_name="Two-Sequence (ST1)",
            metric_name="misclassifications",
            achieved_value=float(misclassified),
            threshold=0.0,
            passed=passed,
            details=f"{misclassified}/{len(predictions)} misclassified"
        )
    else:  # ST2
        mean_abs_error = np.mean(np.abs(predictions - targets))
        threshold = 0.01
        passed = mean_abs_error < threshold
        
        return ExperimentResult(
            experiment_name="Two-Sequence (ST2)",
            metric_name="mean_absolute_error",
            achieved_value=float(mean_abs_error),
            threshold=threshold,
            passed=bool(passed)
        )


# ============================================================================
# Loss functions matching paper specifications
# ============================================================================

def mse_loss_np(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error loss."""
    return float(np.mean((predictions - targets) ** 2))


def cross_entropy_loss_np(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-7
) -> float:
    """Cross-entropy loss for classification tasks."""
    # Clip predictions to avoid log(0)
    preds_clipped = np.clip(predictions, eps, 1 - eps)
    return -float(np.mean(targets * np.log(preds_clipped)))


# ============================================================================
# Summary utilities
# ============================================================================

def print_results_table(results: List[ExperimentResult]) -> None:
    """Print a formatted table of experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS (vs. 1997 Paper Criteria)")
    print("=" * 70)
    
    for result in results:
        print(result)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("-" * 70)
    print(f"Total: {passed}/{total} experiments passed")
    print("=" * 70 + "\n")
