"""Validate the deterministic numerical claims reported by the paper.

This script intentionally avoids machine-dependent runtime thresholds.  It checks
exact identities, approximation guarantees, row counts, and the generated LaTeX
macros so that CI fails if the experiment code and manuscript drift apart.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"expected {expected}, got {actual}")


def row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    summary_path = RESULTS / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Closed-form theorem instances.
    close(float(summary["public_private_n10_public"]), 0.9)
    close(float(summary["public_private_n10_private"]), 0.5)
    close(float(summary["public_private_n10_premium"]), 0.4)
    close(float(summary["decoy_mi_gain_ratio"]), 0.0)
    close(float(summary["greedy_delta_001_ratio"]), 0.01 / 0.99)

    # The separable-revelation suite must obey the proved greedy bound.
    guarantee = 1.0 - 1.0 / math.e
    if float(summary["coverage_greedy_min"]) + 1e-9 < guarantee:
        raise AssertionError("observed revelation ratio violates the 1-1/e theorem")
    if int(summary["coverage_instances"]) != 250:
        raise AssertionError("unexpected revelation instance count")

    # In the operational emergency suite every state variable affects reward.
    # Value-aware greedy is exact here; information-only selectors remain poor.
    close(float(summary["emergency_budget1_greedy_mean"]), 1.0, tol=1e-8)
    if float(summary["emergency_budget1_mi_mean"]) >= 0.01:
        raise AssertionError("MI baseline no longer exhibits the reported separation")
    if float(summary["emergency_budget1_cmi_mean"]) >= 0.01:
        raise AssertionError("conditional-MI baseline no longer exhibits the separation")
    close(float(summary["emergency_nuisance_mi_mean"]), 0.0)
    if int(summary["emergency_instances"]) != 40:
        raise AssertionError("unexpected emergency instance count")

    # Runtime experiment combinatorics are deterministic; wall-clock ratios are not.
    if int(summary["runtime_enumerated_subsets"]) != 155_382:
        raise AssertionError("unexpected subset enumeration count")
    if float(summary["runtime_joint_seconds"]) <= 0.0:
        raise AssertionError("invalid joint-MILP timing")

    expected_rows = {
        "public_private.csv": 20,
        "information_decoy.csv": 24,
        "greedy_trap.csv": 18,
        "revelation_coverage.csv": 250,
        "emergency_response.csv": 80,
        "runtime_scaling.csv": 21,
    }
    for filename, expected in expected_rows.items():
        actual = row_count(RESULTS / filename)
        if actual != expected:
            raise AssertionError(f"{filename}: expected {expected} rows, got {actual}")

    generated = (PAPER / "generated_results.tex").read_text(encoding="utf-8")
    required_macros = (
        r"\newcommand{\CoverageGreedyMean}",
        r"\newcommand{\EmergencyGreedyMean}",
        r"\newcommand{\EmergencyMIMean}",
        r"\newcommand{\RuntimeSpeedup}",
    )
    missing = [macro for macro in required_macros if macro not in generated]
    if missing:
        raise AssertionError(f"missing generated LaTeX macros: {missing}")

    print("Validated theorem instances, benchmark aggregates, CSV sizes, and LaTeX macros.")


if __name__ == "__main__":
    main()
