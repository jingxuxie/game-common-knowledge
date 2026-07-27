"""Information-theoretic baselines and diagnostics."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import log2
from typing import Hashable, Iterable, Mapping

import numpy as np

from .core import TeamGame


def entropy_of_labels(labels: Iterable[Hashable], weights: np.ndarray) -> float:
    """Weighted Shannon entropy in bits."""
    masses: dict[Hashable, float] = defaultdict(float)
    for label, weight in zip(labels, weights, strict=True):
        masses[label] += float(weight)
    return -sum(p * log2(p) for p in masses.values() if p > 0)


def joint_public_entropy(game: TeamGame, subset: Iterable[int]) -> float:
    selected = tuple(sorted(set(subset)))
    labels = [tuple(game.public_features[j, s] for j in selected) for s in range(game.num_states)]
    return entropy_of_labels(labels, game.prior)


def conditional_feature_entropy(
    game: TeamGame,
    feature: int,
    selected: Iterable[int] = (),
    agent: int | None = None,
) -> float:
    """Compute H(phi_feature | selected public signal[, private observation]).

    Because all features are deterministic functions of the world state, this is
    exactly the incremental mutual information with the state.
    """
    selected_tuple = tuple(sorted(set(selected)))
    groups: dict[tuple[Hashable, ...], list[int]] = defaultdict(list)
    for s in range(game.num_states):
        key: tuple[Hashable, ...] = tuple(game.public_features[j, s] for j in selected_tuple)
        if agent is not None:
            key = (game.private_observations[agent][s],) + key
        groups[key].append(s)

    total = 0.0
    for states in groups.values():
        mass = float(game.prior[states].sum())
        if mass <= 0:
            continue
        local_weights = game.prior[states] / mass
        labels = [game.public_features[feature, s] for s in states]
        total += mass * entropy_of_labels(labels, local_weights)
    return total


def mean_private_conditioned_increment(
    game: TeamGame, feature: int, selected: Iterable[int] = ()
) -> float:
    return float(
        np.mean(
            [
                conditional_feature_entropy(game, feature, selected, agent=i)
                for i in range(game.num_agents)
            ]
        )
    )


def mutual_information_greedy(
    game: TeamGame,
    budget: float,
    private_conditioned: bool = False,
) -> tuple[int, ...]:
    """Utility-agnostic greedy feature selection by incremental information."""
    selected: list[int] = []
    spent = 0.0
    while True:
        candidates = [
            j
            for j in range(game.num_features)
            if j not in selected and spent + game.feature_costs[j] <= budget + 1e-9
        ]
        if not candidates:
            break
        if private_conditioned:
            score = lambda j: mean_private_conditioned_increment(game, j, selected)
        else:
            score = lambda j: conditional_feature_entropy(game, j, selected)
        best = max(candidates, key=lambda j: (score(j) / game.feature_costs[j], score(j), -j))
        selected.append(best)
        spent += float(game.feature_costs[best])
    return tuple(sorted(selected))


def submodularity_ratio(
    values: Mapping[frozenset[int], float],
    ground_size: int,
    max_added: int,
    tol: float = 1e-12,
) -> float:
    """Exact finite-set submodularity ratio for a cached monotone value table.

    The returned ratio is clipped to [0, 1].  ``values`` must contain every set
    queried by the definition.  This routine is exponential and is intended for
    small diagnostic instances only.
    """
    universe = set(range(ground_size))
    gamma = 1.0
    for l_key, base in values.items():
        remaining = sorted(universe - set(l_key))
        for r in range(1, min(max_added, len(remaining)) + 1):
            for added_tuple in combinations(remaining, r):
                added = frozenset(added_tuple)
                union = frozenset(set(l_key) | set(added))
                if union not in values:
                    continue
                joint_gain = values[union] - base
                if joint_gain <= tol:
                    continue
                singleton_gain = 0.0
                complete = True
                for j in added:
                    one = frozenset(set(l_key) | {j})
                    if one not in values:
                        complete = False
                        break
                    singleton_gain += max(0.0, values[one] - base)
                if complete:
                    gamma = min(gamma, singleton_gain / joint_gain)
    return float(min(1.0, max(0.0, gamma)))
