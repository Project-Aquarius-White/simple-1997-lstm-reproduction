#!/usr/bin/env python3
"""
Experiment Orchestrator - Run all 1997 LSTM paper experiments.

This script orchestrates running all six experiments from the paper:
    1. Adding Problem (Section 5.4)
    2. Multiplication Problem (Section 5.5)
    3. Temporal Order (Section 5.6)
    4. Embedded Reber Grammar (Section 5.1)
    5. Long Time Lag (Section 5.2)
    6. Two-Sequence Problem (Section 5.3)

Usage:
    # Quick smoke test (all experiments)
    python -m experiments.run_all --mode smoke

    # Full paper reproduction
    python -m experiments.run_all --mode paper

    # Run specific experiments only
    python -m experiments.run_all --experiments adding,multiplication

    # Run with both backends
    python -m experiments.run_all --backend both
"""

import sys
import argparse
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import metrics directly to avoid loading backends via __init__.py
try:
    from aquarius_lstm.metrics import print_results_table, ExperimentResult
except ImportError:
    # If package import fails due to missing backends, import module directly
    import importlib.util
    metrics_path = Path(__file__).parent.parent / "src" / "aquarius_lstm" / "metrics.py"
    spec = importlib.util.spec_from_file_location("metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics from {metrics_path}")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    print_results_table = metrics.print_results_table
    ExperimentResult = metrics.ExperimentResult


# All available experiments
AVAILABLE_EXPERIMENTS = [
    "adding",
    "multiplication",
    "temporal_order",
    "reber",
    "long_lag",
    "two_sequence",
]


@dataclass
class ExperimentRunResult:
    """Result from running an experiment module."""
    name: str
    exit_code: int
    duration_seconds: float
    backend: str
    error: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and self.error is None
    
    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "PASS" if self.passed else "FAIL"


def run_experiment(
    name: str,
    mode: str,
    backend: str,
    seed: int = 42,
) -> ExperimentRunResult:
    """
    Run a single experiment module.
    
    Args:
        name: Experiment name (e.g., "adding", "reber")
        mode: "smoke" for quick test, "paper" for full reproduction
        backend: "tinygrad", "torch", or "both"
        seed: Random seed for reproducibility
    
    Returns:
        ExperimentRunResult with exit code and timing info
    """
    start_time = time.time()
    
    try:
        # Import the experiment module
        module = importlib.import_module(f"experiments.{name}")
    except ImportError as e:
        duration = time.time() - start_time
        return ExperimentRunResult(
            name=name,
            exit_code=1,
            duration_seconds=duration,
            backend=backend,
            error=f"ImportError: {e}",
        )
    
    # Check if module has main function
    if not hasattr(module, "main"):
        duration = time.time() - start_time
        return ExperimentRunResult(
            name=name,
            exit_code=1,
            duration_seconds=duration,
            backend=backend,
            error="Module missing main() function",
        )
    
    # Temporarily override sys.argv to pass arguments to the experiment
    original_argv = sys.argv
    sys.argv = [
        f"experiments/{name}.py",
        "--mode", mode,
        "--backend", backend,
        "--seed", str(seed),
    ]
    
    try:
        exit_code = module.main()
        duration = time.time() - start_time
        return ExperimentRunResult(
            name=name,
            exit_code=exit_code if exit_code is not None else 0,
            duration_seconds=duration,
            backend=backend,
        )
    except ImportError as e:
        # Backend not available
        duration = time.time() - start_time
        return ExperimentRunResult(
            name=name,
            exit_code=1,
            duration_seconds=duration,
            backend=backend,
            error=f"Backend not available: {e}",
        )
    except Exception as e:
        duration = time.time() - start_time
        return ExperimentRunResult(
            name=name,
            exit_code=1,
            duration_seconds=duration,
            backend=backend,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        sys.argv = original_argv


def print_summary_table(results: List[ExperimentRunResult]) -> None:
    """Print a formatted summary table of all experiment results."""
    print("\n" + "=" * 78)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 78)
    
    # Header
    print(f"{'Experiment':<20} {'Backend':<10} {'Status':<8} {'Duration':<12} {'Details'}")
    print("-" * 78)
    
    # Results
    for result in results:
        duration_str = f"{result.duration_seconds:.2f}s"
        details = result.error if result.error else ""
        if len(details) > 30:
            details = details[:27] + "..."
        print(f"{result.name:<20} {result.backend:<10} {result.status:<8} {duration_str:<12} {details}")
    
    print("-" * 78)
    
    # Summary counts
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and not r.error)
    errors = sum(1 for r in results if r.error)
    total = len(results)
    total_time = sum(r.duration_seconds for r in results)
    
    print(f"Total: {passed} passed, {failed} failed, {errors} errors (out of {total})")
    print(f"Total duration: {total_time:.2f}s")
    print("=" * 78 + "\n")


def parse_experiments(experiments_str: Optional[str]) -> List[str]:
    """Parse comma-separated experiment list, returning all if None."""
    if experiments_str is None:
        return AVAILABLE_EXPERIMENTS.copy()
    
    experiments = [e.strip() for e in experiments_str.split(",")]
    
    # Validate experiment names
    invalid = [e for e in experiments if e not in AVAILABLE_EXPERIMENTS]
    if invalid:
        print(f"Warning: Unknown experiments will be skipped: {invalid}")
        experiments = [e for e in experiments if e in AVAILABLE_EXPERIMENTS]
    
    return experiments


def main() -> int:
    """
    Main entry point for the experiment orchestrator.
    
    Returns:
        0 if all experiments passed, 1 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Run all 1997 LSTM paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick smoke test
    python -m experiments.run_all --mode smoke

    # Full paper reproduction
    python -m experiments.run_all --mode paper

    # Run specific experiments
    python -m experiments.run_all --experiments adding,multiplication

    # Test both backends
    python -m experiments.run_all --backend both
        """,
    )
    
    parser.add_argument(
        "--mode",
        choices=["smoke", "paper"],
        default="smoke",
        help="smoke: quick test, paper: full reproduction (default: smoke)",
    )
    
    parser.add_argument(
        "--backend",
        choices=["tinygrad", "torch", "both"],
        default="tinygrad",
        help="Backend to use for experiments (default: tinygrad)",
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of experiments to run (default: all)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Parse experiment list
    experiments = parse_experiments(args.experiments)
    
    if not experiments:
        print("Error: No valid experiments specified.")
        return 1
    
    # Determine backends to run
    if args.backend == "both":
        backends = ["tinygrad", "torch"]
    else:
        backends = [args.backend]
    
    # Print run configuration
    print("\n" + "=" * 78)
    print("LSTM 1997 PAPER EXPERIMENTS")
    print("=" * 78)
    print(f"Mode: {args.mode}")
    print(f"Backend(s): {', '.join(backends)}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Seed: {args.seed}")
    print("=" * 78 + "\n")
    
    # Run all experiments
    all_results: List[ExperimentRunResult] = []
    
    for experiment in experiments:
        for backend in backends:
            print(f"\n{'#' * 78}")
            print(f"# Running: {experiment} ({backend})")
            print(f"{'#' * 78}\n")
            
            result = run_experiment(
                name=experiment,
                mode=args.mode,
                backend=backend,
                seed=args.seed,
            )
            all_results.append(result)
            
            # Print immediate result
            print(f"\n>>> {experiment} ({backend}): {result.status}")
            if result.error:
                print(f"    Error: {result.error}")
    
    # Print summary table
    print_summary_table(all_results)
    
    # Determine exit code
    all_passed = all(r.passed for r in all_results)
    
    if all_passed:
        print("All experiments PASSED!")
        return 0
    else:
        failed_experiments = [r.name for r in all_results if not r.passed]
        print(f"Some experiments FAILED: {', '.join(set(failed_experiments))}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
