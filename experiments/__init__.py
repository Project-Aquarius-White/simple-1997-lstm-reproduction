"""
Experiments package initialization.

Each experiment module can be run standalone:
    python -m experiments.adding --mode paper
    python -m experiments.adding --mode smoke

Or use run_all.py to run all experiments.
"""

__all__ = [
    "adding",
    "multiplication", 
    "temporal_order",
    "reber",
    "long_lag",
    "two_sequence",
    "run_all",
]
