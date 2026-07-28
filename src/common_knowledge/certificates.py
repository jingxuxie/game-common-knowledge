"""Correlated public-certificate instances.

The class in this module models public sensors that share failure domains.  A
sensor can emit a decision-sufficient certificate only when its domain is up
and its local detector succeeds.  Sensors in the same domain are therefore
correlated, but the expected coordination gain remains a coverage function.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


Subset = tuple[int, ...]


@dataclass(frozen=True)
class SharedFailureCertificateInstance:
    """Finite correlated-certificate design instance.

    Parameters
    ----------
    context_weights:
        Probability of each context, shape ``(R,)`` and summing to one.
    baseline_success:
        Team success before any decisive public certificate, shape ``(R,)``.
    domain_uptime:
        Probability that each shared failure domain is available in each
        context, shape ``(R, D)``.
    local_success:
        Conditional certificate probability of every sensor given that its
        domain is available, shape ``(R, M)``.
    sensor_domains:
        Integer domain assignment for every sensor, shape ``(M,)``.
    costs:
        Positive feature costs.  The experiments use unit costs, while the
        paper's theorem permits a general knapsack budget.
    """

    context_weights: np.ndarray
    baseline_success: np.ndarray
    domain_uptime: np.ndarray
    local_success: np.ndarray
    sensor_domains: np.ndarray
    costs: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.context_weights, dtype=float)
        q = np.asarray(self.baseline_success, dtype=float)
        a = np.asarray(self.domain_uptime, dtype=float)
        p = np.asarray(self.local_success, dtype=float)
        d = np.asarray(self.sensor_domains, dtype=int)
        c = np.asarray(self.costs, dtype=float)

        if w.ndim != 1 or q.shape != w.shape:
            raise ValueError("context_weights and baseline_success must have shape (R,)")
        if a.ndim != 2 or a.shape[0] != w.size:
            raise ValueError("domain_uptime must have shape (R, D)")
        if p.ndim != 2 or p.shape[0] != w.size:
            raise ValueError("local_success must have shape (R, M)")
        if d.shape != (p.shape[1],) or c.shape != (p.shape[1],):
            raise ValueError("sensor_domains and costs must have shape (M,)")
        if np.any(d < 0) or np.any(d >= a.shape[1]):
            raise ValueError("sensor domain index out of range")
        if not np.isclose(w.sum(), 1.0, atol=1e-10):
            raise ValueError("context_weights must sum to one")
        if np.any(w < 0) or np.any((q < 0) | (q > 1)):
            raise ValueError("invalid context weights or baseline probabilities")
        if np.any((a < 0) | (a > 1)) or np.any((p < 0) | (p > 1)):
            raise ValueError("uptime and local-success probabilities must lie in [0, 1]")
        if np.any(c <= 0):
            raise ValueError("feature costs must be positive")

        # Store defensive copies in the frozen dataclass.
        object.__setattr__(self, "context_weights", w.copy())
        object.__setattr__(self, "baseline_success", q.copy())
        object.__setattr__(self, "domain_uptime", a.copy())
        object.__setattr__(self, "local_success", p.copy())
        object.__setattr__(self, "sensor_domains", d.copy())
        object.__setattr__(self, "costs", c.copy())

    @property
    def num_contexts(self) -> int:
        return int(self.context_weights.size)

    @property
    def num_domains(self) -> int:
        return int(self.domain_uptime.shape[1])

    @property
    def num_features(self) -> int:
        return int(self.local_success.shape[1])

    @property
    def baseline_value(self) -> float:
        return float(np.dot(self.context_weights, self.baseline_success))

    def _canonical_subset(self, subset: Iterable[int]) -> Subset:
        selected = tuple(sorted(set(int(j) for j in subset)))
        if any(j < 0 or j >= self.num_features for j in selected):
            raise IndexError("feature index out of range")
        return selected

    def gain(self, subset: Iterable[int], *, assume_independent: bool = False) -> float:
        """Expected coordination gain for a selected feature set.

        ``assume_independent=True`` evaluates the misspecified surrogate that
        replaces shared-domain failures by independent sensors with the same
        marginal certificate probabilities.  The returned value is therefore
        a *surrogate objective*, not the true value of the selected set.
        """

        selected = self._canonical_subset(subset)
        if not selected:
            return 0.0

        no_certificate = np.ones(self.num_contexts, dtype=float)
        if assume_independent:
            for j in selected:
                domain = self.sensor_domains[j]
                marginal = self.domain_uptime[:, domain] * self.local_success[:, j]
                no_certificate *= 1.0 - marginal
        else:
            for domain in range(self.num_domains):
                local_indices = [j for j in selected if self.sensor_domains[j] == domain]
                if not local_indices:
                    continue
                local_miss = np.prod(1.0 - self.local_success[:, local_indices], axis=1)
                # Either the shared domain is down, or it is up and every local
                # sensor misses.  Different domains are independent in this
                # benchmark; within-domain sensors are correlated through uptime.
                no_certificate *= (
                    1.0 - self.domain_uptime[:, domain]
                    + self.domain_uptime[:, domain] * local_miss
                )

        context_gain = (1.0 - self.baseline_success) * (1.0 - no_certificate)
        return float(np.dot(self.context_weights, context_gain))

    def value(self, subset: Iterable[int]) -> float:
        return self.baseline_value + self.gain(subset)

    def greedy(self, budget: int, *, assume_independent: bool = False) -> Subset:
        """Cardinality-budget greedy under the true or misspecified objective."""

        if budget < 0:
            raise ValueError("budget must be nonnegative")
        selected: list[int] = []
        for _ in range(min(budget, self.num_features)):
            current = self.gain(selected, assume_independent=assume_independent)
            best_j: int | None = None
            best_marginal = -np.inf
            for j in range(self.num_features):
                if j in selected:
                    continue
                candidate = self.gain((*selected, j), assume_independent=assume_independent)
                marginal = candidate - current
                if marginal > best_marginal + 1e-15:
                    best_marginal = marginal
                    best_j = j
            if best_j is None:
                break
            selected.append(best_j)
        return tuple(sorted(selected))

    def singleton_ranking(self, budget: int) -> Subset:
        order = sorted(
            range(self.num_features),
            key=lambda j: self.gain((j,)),
            reverse=True,
        )
        return tuple(sorted(order[: min(budget, self.num_features)]))

    def optimum(self, budget: int) -> tuple[Subset, float]:
        """Exact cardinality-budget optimum by enumeration."""

        if budget < 0:
            raise ValueError("budget must be nonnegative")
        best_subset: Subset = ()
        best_gain = 0.0
        max_size = min(budget, self.num_features)
        for size in range(max_size + 1):
            for subset in combinations(range(self.num_features), size):
                candidate = self.gain(subset)
                if candidate > best_gain + 1e-15:
                    best_subset = subset
                    best_gain = candidate
        return best_subset, best_gain


def random_shared_failure_instance(
    seed: int,
    *,
    num_contexts: int = 8,
    num_domains: int = 3,
    sensors_per_domain: int = 4,
) -> SharedFailureCertificateInstance:
    """Generate a correlated benchmark with one attractive but fragile domain.

    Sensors in the fragile domain have excellent local detectors but share a
    substantial common-mode outage probability.  Other domains have more
    moderate local detectors but high uptime.  A marginal-independence model
    tends to over-select the fragile domain, whereas the true objective rewards
    diversification across failure domains.
    """

    if num_domains < 2 or sensors_per_domain < 1:
        raise ValueError("need at least two domains and one sensor per domain")

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(num_contexts))
    baseline = rng.uniform(0.45, 0.72, size=num_contexts)
    domain_uptime = np.empty((num_contexts, num_domains), dtype=float)
    num_features = num_domains * sensors_per_domain
    local_success = np.empty((num_contexts, num_features), dtype=float)
    sensor_domains = np.repeat(np.arange(num_domains), sensors_per_domain)

    fragile_domain = int(rng.integers(num_domains))
    for domain in range(num_domains):
        if domain == fragile_domain:
            domain_uptime[:, domain] = rng.uniform(0.52, 0.68, size=num_contexts)
            local_low, local_high = 0.91, 0.995
        else:
            domain_uptime[:, domain] = rng.uniform(0.88, 0.98, size=num_contexts)
            local_low, local_high = 0.48, 0.68
        for local_index in range(sensors_per_domain):
            j = domain * sensors_per_domain + local_index
            local_success[:, j] = rng.uniform(local_low, local_high, size=num_contexts)

    return SharedFailureCertificateInstance(
        context_weights=weights,
        baseline_success=baseline,
        domain_uptime=domain_uptime,
        local_success=local_success,
        sensor_domains=sensor_domains,
        costs=np.ones(num_features, dtype=float),
    )
