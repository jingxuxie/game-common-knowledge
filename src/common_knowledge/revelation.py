"""Analytic separable-revelation subclass."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RevelationInstance:
    """Independent public sensors for context-dependent hidden targets.

    Context k occurs with weight ``weights[k]``.  If the target is not publicly
    revealed, the team succeeds with baseline probability ``baseline[k]``.
    Sensor j reveals the target in context k independently with probability
    ``reveal_prob[k, j]``.
    """

    weights: np.ndarray
    baseline: np.ndarray
    reveal_prob: np.ndarray
    costs: np.ndarray | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        b = np.asarray(self.baseline, dtype=float)
        p = np.asarray(self.reveal_prob, dtype=float)
        if w.ndim != 1 or b.shape != w.shape or p.ndim != 2 or p.shape[0] != w.size:
            raise ValueError("incompatible revelation-instance shapes")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum")
        if np.any((b < 0) | (b > 1)) or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must lie in [0, 1]")
        object.__setattr__(self, "weights", w / w.sum())
        object.__setattr__(self, "baseline", b)
        object.__setattr__(self, "reveal_prob", p)
        costs = np.ones(p.shape[1]) if self.costs is None else np.asarray(self.costs, dtype=float)
        if costs.shape != (p.shape[1],) or np.any(costs <= 0):
            raise ValueError("invalid sensor costs")
        object.__setattr__(self, "costs", costs)

    @property
    def num_features(self) -> int:
        return int(self.reveal_prob.shape[1])

    def value(self, subset: Iterable[int]) -> float:
        selected = tuple(sorted(set(subset)))
        if selected:
            miss = np.prod(1.0 - self.reveal_prob[:, selected], axis=1)
        else:
            miss = np.ones_like(self.weights)
        success = 1.0 - (1.0 - self.baseline) * miss
        return float(np.dot(self.weights, success))

    def gain(self, subset: Iterable[int]) -> float:
        return self.value(subset) - self.value(())

    def greedy(self, budget: float) -> tuple[int, ...]:
        selected: list[int] = []
        spent = 0.0
        while True:
            candidates = [
                j
                for j in range(self.num_features)
                if j not in selected and spent + self.costs[j] <= budget + 1e-9
            ]
            if not candidates:
                break
            current = self.value(selected)
            best = max(
                candidates,
                key=lambda j: ((self.value(selected + [j]) - current) / self.costs[j], -j),
            )
            selected.append(best)
            spent += float(self.costs[best])
        return tuple(sorted(selected))

    def optimum(self, budget: float) -> tuple[int, ...]:
        best_set: tuple[int, ...] = ()
        best_value = self.value(())
        for r in range(self.num_features + 1):
            for subset in combinations(range(self.num_features), r):
                if self.costs[list(subset)].sum() > budget + 1e-9:
                    continue
                value = self.value(subset)
                if value > best_value + 1e-12:
                    best_value, best_set = value, subset
        return best_set
