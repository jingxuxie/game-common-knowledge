#!/usr/bin/env python3
"""Check the deterministic correlated-certificate outputs used in the paper."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
frame = pd.read_csv(ROOT / "results" / "correlated_certificates.csv")
summary = json.loads((ROOT / "results" / "correlated_certificates_summary.json").read_text())
macros = (ROOT / "paper" / "generated_correlated_results.tex").read_text()

assert len(frame) == summary["correlated_instances"] == 250
assert abs(frame.true_greedy_ratio.mean() - summary["correlated_true_greedy_mean"]) < 1e-12
assert abs(frame.independence_greedy_ratio.mean() - summary["correlated_independence_mean"]) < 1e-12
assert frame.true_greedy_ratio.min() >= 1 - 1 / 2.718281828459045 - 1e-12
assert frame.independence_greedy_ratio.min() < 0.7
assert frame.true_greedy_domains.mean() > frame.independence_greedy_domains.mean() + 0.8
assert f"{{{summary['correlated_true_greedy_mean']:.3f}}}" in macros
assert f"{{{summary['correlated_independence_mean']:.3f}}}" in macros
print("correlated-certificate outputs validated")
