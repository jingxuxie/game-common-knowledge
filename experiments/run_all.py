#!/usr/bin/env python3
"""Reproduce every experiment and figure in the paper.

The experiments are deliberately small and exact.  They require no GPU and are
expected to finish in a few minutes on a laptop.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_knowledge.benchmarks import (
    emergency_response_game,
    greedy_trap_game,
    information_decoy_game,
    maximum_coverage_game,
    operational_emergency_response_game,
    random_revelation_instance,
)
from common_knowledge.information import mutual_information_greedy
from common_knowledge.optimization import (
    evaluate_subset,
    exact_subset_search,
    solve_design_milp,
    value_greedy,
)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
for directory in (RESULTS, FIGURES, PAPER):
    directory.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def public_private_experiment() -> dict[str, float]:
    epsilon = 0.1
    rows = []
    for n in range(1, 21):
        public = 1 - epsilon
        private = max(0.5, (1 - epsilon) ** n)
        rows.append({"agents": n, "public": public, "private": private, "premium": public - private})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "public_private.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(frame["agents"], frame["public"], marker="o", label="One public channel")
    ax.plot(frame["agents"], frame["private"], marker="s", label="Independent private channels")
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Optimal probability all agents are correct")
    ax.set_ylim(0.45, 0.95)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "public_private_gap")

    return {
        "public_private_epsilon": epsilon,
        "public_private_n10_public": float(frame.loc[frame.agents == 10, "public"].iloc[0]),
        "public_private_n10_private": float(frame.loc[frame.agents == 10, "private"].iloc[0]),
        "public_private_n10_premium": float(frame.loc[frame.agents == 10, "premium"].iloc[0]),
        "public_private_limit_premium": 0.5 - epsilon,
    }


def structural_counterexamples() -> dict[str, float]:
    decoy_rows = []
    for p in np.linspace(0.02, 0.48, 24):
        game = information_decoy_game(float(p))
        baseline = evaluate_subset(game, ()).value
        mi_subset = mutual_information_greedy(game, budget=1)
        optimum = solve_design_milp(game, budget=1)
        mi_value = evaluate_subset(game, mi_subset).value
        decoy_rows.append(
            {
                "target_probability": p,
                "baseline": baseline,
                "mi_value": mi_value,
                "optimal_value": optimum.value,
                "mi_gain_ratio": 0.0 if optimum.value <= baseline + 1e-12 else (mi_value - baseline) / (optimum.value - baseline),
            }
        )
    decoy = pd.DataFrame(decoy_rows)
    decoy.to_csv(RESULTS / "information_decoy.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.plot(decoy["target_probability"], decoy["optimal_value"] - decoy["baseline"], label="Value-aware selection")
    ax.plot(decoy["target_probability"], decoy["mi_value"] - decoy["baseline"], label="Mutual-information selection")
    ax.set_xlabel("Probability of the rare target state")
    ax.set_ylabel("Coordination gain")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "information_decoy")

    trap_rows = []
    for delta in np.geomspace(0.005, 0.3, 18):
        game = greedy_trap_game(float(delta))
        baseline = evaluate_subset(game, ()).value
        greedy = value_greedy(game, budget=2)
        optimum = solve_design_milp(game, budget=2)
        ratio = (greedy.value - baseline) / (optimum.value - baseline)
        trap_rows.append(
            {
                "delta": delta,
                "greedy_ratio": ratio,
                "theory_ratio": delta / (1 - delta),
                "greedy_subset": str(greedy.subset),
                "optimal_subset": str(optimum.subset),
            }
        )
    trap = pd.DataFrame(trap_rows)
    trap.to_csv(RESULTS / "greedy_trap.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.loglog(trap["delta"], trap["greedy_ratio"], marker="o", label="Exact experiment")
    ax.loglog(trap["delta"], trap["theory_ratio"], linestyle="--", label=r"$\delta/(1-\delta)$")
    ax.set_xlabel(r"Distractor mass $\delta$")
    ax.set_ylabel("Greedy / optimal coordination gain")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    save_figure(fig, "greedy_failure")

    p = 0.1
    game = information_decoy_game(p)
    baseline = evaluate_subset(game, ()).value
    mi_value = evaluate_subset(game, mutual_information_greedy(game, 1)).value
    optimal = solve_design_milp(game, 1).value
    return {
        "decoy_p": p,
        "decoy_baseline": baseline,
        "decoy_mi_value": mi_value,
        "decoy_optimal_value": optimal,
        "decoy_mi_gain_ratio": (mi_value - baseline) / (optimal - baseline),
        "greedy_delta_001_ratio": 0.01 / 0.99,
    }


def revelation_experiment(num_instances: int = 250) -> dict[str, float]:
    rows = []
    rng = np.random.default_rng(2027)
    for seed in range(num_instances):
        instance = random_revelation_instance(seed, num_contexts=8, num_features=12)
        budget = 4
        optimum = instance.optimum(budget)
        greedy = instance.greedy(budget)
        base = instance.value(())
        opt_gain = instance.value(optimum) - base
        greedy_gain = instance.value(greedy) - base

        # Non-adaptive singleton ranking: select the four individually strongest sensors.
        singleton_order = sorted(
            range(instance.num_features),
            key=lambda j: instance.gain((j,)),
            reverse=True,
        )
        singleton_set = tuple(sorted(singleton_order[:budget]))
        singleton_gain = instance.value(singleton_set) - base

        feasible = list(combinations(range(instance.num_features), budget))
        random_sets = [feasible[int(rng.integers(len(feasible)))] for _ in range(30)]
        random_gain = float(np.mean([instance.value(s) - base for s in random_sets]))

        rows.append(
            {
                "seed": seed,
                "baseline": base,
                "optimal_gain": opt_gain,
                "greedy_gain": greedy_gain,
                "singleton_gain": singleton_gain,
                "random_gain": random_gain,
                "greedy_ratio": greedy_gain / opt_gain,
                "singleton_ratio": singleton_gain / opt_gain,
                "random_ratio": random_gain / opt_gain,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "revelation_coverage.csv", index=False)

    labels = ["Value greedy", "Singleton ranking", "Random"]
    data = [frame["greedy_ratio"], frame["singleton_ratio"], frame["random_ratio"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(1 - 1 / math.e, linestyle="--", linewidth=1, label=r"$1-1/e$")
    ax.set_ylabel("Normalized coordination gain")
    ax.set_ylim(0, 1.04)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.2, axis="y")
    save_figure(fig, "revelation_coverage")

    return {
        "coverage_instances": num_instances,
        "coverage_greedy_mean": float(frame.greedy_ratio.mean()),
        "coverage_greedy_min": float(frame.greedy_ratio.min()),
        "coverage_greedy_q05": float(frame.greedy_ratio.quantile(0.05)),
        "coverage_singleton_mean": float(frame.singleton_ratio.mean()),
        "coverage_random_mean": float(frame.random_ratio.mean()),
    }


def emergency_experiment(num_instances: int = 40) -> dict[str, float | str]:
    """Evaluate utility-aware versus information-only alarm selection.

    The primary benchmark contains only operational features derived from fire
    severity.  A secondary variant adds two high-entropy, payoff-irrelevant
    weather bits as a stress test; both are saved in the output CSV.
    """
    rows = []
    selected_opt = Counter()
    for variant in ("operational-only", "with-nuisance-telemetry"):
        for seed in range(num_instances):
            game = (
                operational_emergency_response_game(seed)
                if variant == "operational-only"
                else emergency_response_game(seed)
            )
            for budget in [1]:
                baseline = evaluate_subset(game, ()).value
                optimum = solve_design_milp(game, budget, time_limit=60)
                greedy = value_greedy(game, budget)
                mi_set = mutual_information_greedy(game, budget, private_conditioned=False)
                cmi_set = mutual_information_greedy(game, budget, private_conditioned=True)
                mi = evaluate_subset(game, mi_set)
                cmi = evaluate_subset(game, cmi_set)

                feasible_singletons = [j for j, cost in enumerate(game.feature_costs) if cost <= budget]
                random_values = [evaluate_subset(game, (j,)).value for j in feasible_singletons]
                random_value = float(np.mean(random_values))
                opt_gain = optimum.value - baseline

                def ratio(value: float) -> float:
                    return 1.0 if opt_gain <= 1e-12 else max(0.0, (value - baseline) / opt_gain)

                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "budget": budget,
                        "baseline": baseline,
                        "optimal_value": optimum.value,
                        "value_greedy": greedy.value,
                        "mi_value": mi.value,
                        "conditional_mi_value": cmi.value,
                        "random_singleton_value": random_value,
                        "greedy_ratio": ratio(greedy.value),
                        "mi_ratio": ratio(mi.value),
                        "conditional_mi_ratio": ratio(cmi.value),
                        "random_ratio": ratio(random_value),
                        "optimal_subset": str(optimum.subset),
                        "greedy_subset": str(greedy.subset),
                        "mi_subset": str(mi_set),
                        "conditional_mi_subset": str(cmi_set),
                        "joint_milp_seconds": optimum.runtime_seconds,
                    }
                )
                if variant == "operational-only":
                    selected_opt.update([tuple(game.feature_names[j] for j in optimum.subset)])

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "emergency_response.csv", index=False)

    primary = frame[(frame.variant == "operational-only") & (frame.budget == 1)]
    stress = frame[(frame.variant == "with-nuisance-telemetry") & (frame.budget == 1)]
    labels = ["Value greedy", "MI", "Private-conditioned MI", "Random singleton"]
    means = [
        primary.greedy_ratio.mean(),
        primary.mi_ratio.mean(),
        primary.conditional_mi_ratio.mean(),
        primary.random_ratio.mean(),
    ]
    errors = [
        primary.greedy_ratio.std(ddof=1) / math.sqrt(len(primary)),
        primary.mi_ratio.std(ddof=1) / math.sqrt(len(primary)),
        primary.conditional_mi_ratio.std(ddof=1) / math.sqrt(len(primary)),
        primary.random_ratio.std(ddof=1) / math.sqrt(len(primary)),
    ]
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    ax.bar(range(len(labels)), means, yerr=errors, capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction of optimal coordination gain")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.2, axis="y")
    save_figure(fig, "emergency_response")

    return {
        "emergency_instances": num_instances,
        "emergency_budget1_greedy_mean": float(primary.greedy_ratio.mean()),
        "emergency_budget1_mi_mean": float(primary.mi_ratio.mean()),
        "emergency_budget1_cmi_mean": float(primary.conditional_mi_ratio.mean()),
        "emergency_budget1_random_mean": float(primary.random_ratio.mean()),
        "emergency_nuisance_mi_mean": float(stress.mi_ratio.mean()),
        "emergency_joint_milp_median_seconds": float(frame.joint_milp_seconds.median()),
        "emergency_most_common_optimal": str(selected_opt.most_common(3)),
    }


def runtime_experiment() -> dict[str, float]:
    """Compare the joint MILP with direct subset enumeration on coverage games.

    The coverage family has a closed-form value for a selected sensor set, so the
    exhaustive baseline need not solve a second policy MILP for every subset.
    This keeps the scaling experiment stable after the other suites have invoked
    many exact solves in the same Python process.
    """
    rows = []
    rng = np.random.default_rng(44)
    for m in range(6, 19, 2):
        for repeat in range(3):
            contexts = 12
            cover_sets = []
            for _ in range(m):
                covered = {k for k in range(contexts) if rng.random() < 0.34}
                if not covered:
                    covered.add(int(rng.integers(contexts)))
                cover_sets.append(covered)
            weights = rng.dirichlet(np.ones(contexts))
            game = maximum_coverage_game(cover_sets, weights)
            budget = m // 2

            start = perf_counter()
            joint = solve_design_milp(game, budget, time_limit=60)
            joint_wall = perf_counter() - start
            if not joint.success:
                raise RuntimeError(f"joint MILP failed: {joint.message}")

            start = perf_counter()
            best_covered_mass = 0.0
            for r in range(budget + 1):
                for subset in combinations(range(m), r):
                    covered = set().union(*(cover_sets[j] for j in subset)) if subset else set()
                    best_covered_mass = max(best_covered_mass, float(weights[list(covered)].sum()) if covered else 0.0)
            exhaustive_value = 0.5 + 0.5 * best_covered_mass
            exhaustive_wall = perf_counter() - start
            if not math.isclose(joint.value, exhaustive_value, abs_tol=1e-8):
                raise AssertionError("joint MILP and analytical exhaustive search disagree")
            rows.append(
                {
                    "features": m,
                    "repeat": repeat,
                    "joint_seconds": joint_wall,
                    "enumeration_seconds": exhaustive_wall,
                    "enumerated_subsets": sum(math.comb(m, r) for r in range(budget + 1)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "runtime_scaling.csv", index=False)
    med = frame.groupby("features", as_index=False).median(numeric_only=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.semilogy(med["features"], med["joint_seconds"], marker="o", label="Joint MILP")
    ax.semilogy(med["features"], med["enumeration_seconds"], marker="s", label="Subset enumeration")
    ax.set_xlabel("Number of candidate public features")
    ax.set_ylabel("Wall-clock seconds")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both")
    save_figure(fig, "runtime_scaling")

    last = med.loc[med.features == med.features.max()].iloc[0]
    return {
        "runtime_max_features": int(last.features),
        "runtime_joint_seconds": float(last.joint_seconds),
        "runtime_enumeration_seconds": float(last.enumeration_seconds),
        "runtime_enumerated_subsets": int(last.enumerated_subsets),
        "runtime_speedup": float(last.enumeration_seconds / max(last.joint_seconds, 1e-12)),
    }


def write_latex_macros(summary: dict[str, object]) -> None:
    def fmt(value: object, digits: int = 3) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = ["% Auto-generated by experiments/run_all.py; do not edit manually."]
    mapping = {
        "NumCoverageInstances": ("coverage_instances", 0),
        "CoverageGreedyMean": ("coverage_greedy_mean", 3),
        "CoverageGreedyMin": ("coverage_greedy_min", 3),
        "CoverageSingletonMean": ("coverage_singleton_mean", 3),
        "CoverageRandomMean": ("coverage_random_mean", 3),
        "NumEmergencyInstances": ("emergency_instances", 0),
        "EmergencyGreedyMean": ("emergency_budget1_greedy_mean", 3),
        "EmergencyMIMean": ("emergency_budget1_mi_mean", 3),
        "EmergencyCMIMean": ("emergency_budget1_cmi_mean", 3),
        "EmergencyRandomMean": ("emergency_budget1_random_mean", 3),
        "PublicPrivatePremium": ("public_private_n10_premium", 3),
        "RuntimeSpeedup": ("runtime_speedup", 1),
    }
    for macro, (key, digits) in mapping.items():
        lines.append(f"\\newcommand{{\\{macro}}}{{{fmt(summary[key], digits)}}}")
    (PAPER / "generated_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary: dict[str, object] = {}
    summary.update(public_private_experiment())
    summary.update(structural_counterexamples())
    summary.update(revelation_experiment())
    # Run the scaling suite before the many small emergency-response MILPs.
    # Some HiGHS builds retain branch-and-bound state across hundreds of solves.
    summary.update(runtime_experiment())
    summary.update(emergency_experiment())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_latex_macros(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
