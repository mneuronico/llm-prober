"""
Master script: run all figure generation for the paper.
Creates a timestamped results directory under paper/ and generates all panels.

Usage:
    python run_all_figures.py
"""

import os
import sys
import time
from datetime import datetime

# Ensure the script directory is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from figure_1 import generate_figure_1
from figure_2 import generate_figure_2
from figure_3 import generate_figure_3
from figure_4 import generate_figure_4
from figure_5 import generate_figure_5


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(SCRIPT_DIR, '..', f'results_{timestamp}')
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"  Paper Figure Generation")
    print(f"  Output: {results_dir}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)

    t0 = time.time()

    figures = [
        ("Figure 1", generate_figure_1),
        ("Figure 2", generate_figure_2),
        ("Figure 3", generate_figure_3),
        ("Figure 4", generate_figure_4),
        ("Figure 5", generate_figure_5),
    ]

    for name, func in figures:
        tf = time.time()
        try:
            func(results_dir)
            elapsed = time.time() - tf
            print(f"  ✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - tf
            print(f"  ✗ {name} FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  All figures generated in {total:.1f}s")
    print(f"  Output: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
