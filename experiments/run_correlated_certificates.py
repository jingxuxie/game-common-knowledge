#!/usr/bin/env python3
"""Exact correlated-certificate experiment for the manuscript extension."""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_knowledge.certificates import random_shared_failure_instance

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
for directory in (RESULTS, FIGURES, PAPER):
    directory.mkdir(parents=True, exist_ok=True)


def _ratio(value: float, optimum: float) -> float:
    if optimum <= 1e-15:
        return 1.0
    return value / optimum


def run(num_instances: int = 250, budget: int = 4) -> dict[str, float | int]:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(271828)

    for seed in range(num_instances):
        instance = random_shared_failure_instance(seed)
        optimum_subset, optimum_gain = instance.optimum(budget)
        true_greedy = instance.greedy(budget)
        independent_greedy = instance.greedy(budget, assume_independent=True)
        singleton = instance.singleton_ranking(budget)

        feasible = list(combinations(range(instance.num_features), budget))
        sampled = [feasible[int(rng.integers(len(feasible)))] for _ in range(30)]
        random_gain = float(np.mean([instance.gain(subset) for subset in sampled]))

        true_domains = len(set(instance.sensor_domains[list(true_greedy)]))
        independent_domains = len(set(instance.sensor_domains[list(independent_greedy)]))
        optimal_domains = len(set(instance.sensor_domains[list(optimum_subset)]))

        rows.append(
            {
                "seed": seed,
                "optimal_gain": optimum_gain,
                "true_greedy_ratio": _ratio(instance.gain(true_greedy), optimum_gain),
                "independence_greedy_ratio": _ratio(
                    instance.gain(independent_greedy), optimum_gain
                ),
                "singleton_ratio": _ratio(instance.gain(singleton), optimum_gain),
                "random_ratio": _ratio(random_gain, optimum_gain),
                "true_greedy_domains": true_domains,
                "independence_greedy_domains": independent_domains,
                "optimal_domains": optimal_domains,
                "true_greedy_subset": str(true_greedy),
                "independence_greedy_subset": str(independent_greedy),
                "optimal_subset": str(optimum_subset),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "correlated_certificates.csv", index=False)

    labels = [
        "Correlation-aware\ngreedy",
        "Independence-model\ngreedy",
        "Singleton\nranking",
        "Random",
    ]
    ratio_columns = [
        "true_greedy_ratio",
        "independence_greedy_ratio",
        "singleton_ratio",
        "random_ratio",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.15))
    axes[0].boxplot(
        [frame[column] for column in ratio_columns],
        tick_labels=labels,
        showfliers=False,
    )
    axes[0].axhline(1.0 - 1.0 / math.e, linestyle="--", linewidth=1, label=r"$1-1/e$")
    axes[0].set_ylabel("Fraction of optimal coordination gain")
    axes[0].set_ylim(0.55, 1.03)
    axes[0].tick_params(axis="x", rotation=8)
    axes[0].grid(alpha=0.2, axis="y")
    axes[0].legend(frameon=False, loc="lower right")

    domain_means = [
        frame["true_greedy_domains"].mean(),
        frame["independence_greedy_domains"].mean(),
        frame["optimal_domains"].mean(),
    ]
    domain_errors = [
        frame["true_greedy_domains"].std(ddof=1) / math.sqrt(len(frame)),
        frame["independence_greedy_domains"].std(ddof=1) / math.sqrt(len(frame)),
        frame["optimal_domains"].std(ddof=1) / math.sqrt(len(frame)),
    ]
    domain_labels = ["Aware greedy", "Independence model", "Optimum"]
    axes[1].bar(range(3), domain_means, yerr=domain_errors, capsize=3)
    axes[1].set_xticks(range(3), domain_labels, rotation=10, ha="right")
    axes[1].set_ylabel("Failure domains represented")
    axes[1].set_ylim(0, 3.2)
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "correlated_certificates.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "correlated_certificates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, float | int] = {
        "correlated_instances": num_instances,
        "correlated_budget": budget,
        "correlated_true_greedy_mean": float(frame.true_greedy_ratio.mean()),
        "correlated_true_greedy_min": float(frame.true_greedy_ratio.min()),
        "correlated_independence_mean": float(frame.independence_greedy_ratio.mean()),
        "correlated_independence_min": float(frame.independence_greedy_ratio.min()),
        "correlated_singleton_mean": float(frame.singleton_ratio.mean()),
        "correlated_random_mean": float(frame.random_ratio.mean()),
        "correlated_true_domains_mean": float(frame.true_greedy_domains.mean()),
        "correlated_independence_domains_mean": float(
            frame.independence_greedy_domains.mean()
        ),
        "correlated_optimal_domains_mean": float(frame.optimal_domains.mean()),
    }
    (RESULTS / "correlated_certificates_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    macros = "\n".join(
        [
            "% Auto-generated by experiments/run_correlated_certificates.py; do not edit.",
            rf"\newcommand{{\NumCorrelatedInstances}}{{{num_instances}}}",
            rf"\newcommand{{\CorrelatedAwareMean}}{{{summary['correlated_true_greedy_mean']:.3f}}}",
            rf"\newcommand{{\CorrelatedAwareMin}}{{{summary['correlated_true_greedy_min']:.3f}}}",
            rf"\newcommand{{\CorrelatedIndependenceMean}}{{{summary['correlated_independence_mean']:.3f}}}",
            rf"\newcommand{{\CorrelatedIndependenceMin}}{{{summary['correlated_independence_min']:.3f}}}",
            rf"\newcommand{{\CorrelatedSingletonMean}}{{{summary['correlated_singleton_mean']:.3f}}}",
            rf"\newcommand{{\CorrelatedRandomMean}}{{{summary['correlated_random_mean']:.3f}}}",
            rf"\newcommand{{\CorrelatedAwareDomains}}{{{summary['correlated_true_domains_mean']:.2f}}}",
            rf"\newcommand{{\CorrelatedIndependenceDomains}}{{{summary['correlated_independence_domains_mean']:.2f}}}",
            "",
        ]
    )
    (PAPER / "generated_correlated_results.tex").write_text(macros, encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
