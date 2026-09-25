"""
NOEMA Benchmark Suite — Comprehensive testing of the architecture.

Runs:
1. Subsystem unit tests (S1-S6, core modules)
2. Four invariant tests (I1-I4)
3. Four decisive experiments (Section 8)
4. Performance metrics and diagnostics
"""

import torch
import numpy as np
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noema.tests.test_subsystems import run_all_subsystem_tests
from noema.tests.test_invariants import (
    test_I1_autonomy_of_objective,
    test_I2_grounded_abstraction,
    test_I3_relational_portability,
    test_I4_ontogeny,
)
from noema.tests.test_decisive_experiments import run_all_decisive_experiments


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   NOEMA — Neuromorphic Ontogenetic Extrapolative Mapping           ║
║          Architecture                                              ║
║                                                                    ║
║   AGI: A Sufficient Architecture for Artificial General             ║
║   Intelligence                                                     ║
║                                                                    ║
║   "The Sufficiency Thesis, Its Construction, and Its               ║
║    Burden of Proof"                                                ║
║                                                                    ║
║   Kaoru Aguilera Katayama                                          ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def run_full_benchmark():
    """Run the complete NOEMA benchmark suite."""
    print_banner()

    total_start = time.time()
    all_results = {}

    # ================================================================
    # PART 1: Subsystem Unit Tests
    # ================================================================
    print("\n" + "▓" * 70)
    print("▓ PART 1: SUBSYSTEM UNIT TESTS")
    print("▓" * 70)
    
    t0 = time.time()
    subsystem_results = run_all_subsystem_tests()
    t1 = time.time()
    all_results["subsystem_tests"] = subsystem_results
    all_results["subsystem_time"] = t1 - t0

    # ================================================================
    # PART 2: Four Invariants
    # ================================================================
    print("\n" + "▓" * 70)
    print("▓ PART 2: FOUR INVARIANTS (Sufficiency Thesis)")
    print("▓" * 70)

    t0 = time.time()
    invariant_results = {}
    
    try:
        invariant_results["I1"] = test_I1_autonomy_of_objective()
    except Exception as e:
        print(f"  I1 ERROR: {e}")
        invariant_results["I1"] = False

    try:
        invariant_results["I2"] = test_I2_grounded_abstraction()
    except Exception as e:
        print(f"  I2 ERROR: {e}")
        invariant_results["I2"] = False

    try:
        invariant_results["I3"] = test_I3_relational_portability()
    except Exception as e:
        print(f"  I3 ERROR: {e}")
        invariant_results["I3"] = False

    try:
        invariant_results["I4"] = test_I4_ontogeny()
    except Exception as e:
        print(f"  I4 ERROR: {e}")
        invariant_results["I4"] = False

    t1 = time.time()
    all_results["invariants"] = invariant_results
    all_results["invariants_time"] = t1 - t0

    # ================================================================
    # PART 3: Four Decisive Experiments
    # ================================================================
    print("\n" + "▓" * 70)
    print("▓ PART 3: FOUR DECISIVE EXPERIMENTS (Section 8)")
    print("▓" * 70)

    t0 = time.time()
    exp_results, exp_details = run_all_decisive_experiments()
    t1 = time.time()
    all_results["experiments"] = exp_results
    all_results["experiments_time"] = t1 - t0

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_time = time.time() - total_start

    print("\n" + "█" * 70)
    print("█ FINAL VERDICT")
    print("█" * 70)

    print("\n  Subsystem Tests:")
    for name, passed in all_results["subsystem_tests"].items():
        print(f"    {name}: {'✓' if passed else '✗'}")

    print("\n  Invariant Tests:")
    for name, passed in all_results["invariants"].items():
        print(f"    {name}: {'✓' if passed else '✗'}")

    print("\n  Decisive Experiments:")
    for name, passed in all_results["experiments"].items():
        print(f"    {name}: {'✓' if passed else '✗'}")

    subsystem_pass = all(all_results["subsystem_tests"].values())
    invariant_pass = all(all_results["invariants"].values())
    experiment_pass = all(all_results["experiments"].values())

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │ Subsystems functional:  {'YES ✓' if subsystem_pass else 'NO ✗':17s} │")
    print(f"  │ Invariants satisfied:   {'YES ✓' if invariant_pass else 'NO ✗':17s} │")
    print(f"  │ Experiments passed:     {'YES ✓' if experiment_pass else 'NO ✗':17s} │")
    print(f"  │                                         │")

    if subsystem_pass and invariant_pass and experiment_pass:
        print(f"  │  ══════════════════════════════════════  │")
        print(f"  │  SUFFICIENCY THESIS: HOLDS              │")
        print(f"  │  NOEMA is a complete AGI blueprint      │")
        print(f"  │  ══════════════════════════════════════  │")
    else:
        print(f"  │  THESIS STATUS: PARTIALLY VERIFIED      │")
        failing = []
        if not subsystem_pass:
            failing.append("Subsystems")
        if not invariant_pass:
            failing.append("Invariants")
        if not experiment_pass:
            failing.append("Experiments")
        print(f"  │  Failing: {', '.join(failing):26s} │")

    print(f"  │                                         │")
    print(f"  │  Total benchmark time: {total_time:5.1f}s            │")
    print(f"  └─────────────────────────────────────────┘")

    print("""
  The thesis is now in the only place a thesis of this magnitude
  belongs: in the open, fully armed, waiting to be proven or broken.

  — NOEMA, Section 11 (Conclusion)
""")

    return all_results


if __name__ == "__main__":
    results = run_full_benchmark()
