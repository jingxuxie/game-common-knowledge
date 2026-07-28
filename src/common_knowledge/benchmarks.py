"""Small exact benchmark families used in the paper."""
from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np

from .core import TeamGame
from .revelation import RevelationInstance


def parity_game() -> TeamGame:
    states = list(product([0, 1], repeat=2))
    prior = np.full(len(states), 1.0 / len(states))
    private = (np.zeros(len(states), dtype=int), np.zeros(len(states), dtype=int))
    features = np.asarray([[b1 for b1, _ in states], [b2 for _, b2 in states]], dtype=object)
    payoffs = np.zeros((len(states), 2, 2))
    for s, (b1, b2) in enumerate(states):
        target = b1 ^ b2
        payoffs[s, target, target] = 1.0
    return TeamGame(
        prior,
        private,
        features,
        payoffs,
        feature_names=("first parity bit", "second parity bit"),
        state_names=tuple(map(str, states)),
    )


def information_decoy_game(target_one_probability: float = 0.1) -> TeamGame:
    if not 0 < target_one_probability < 0.5:
        raise ValueError("target_one_probability must lie in (0, 1/2)")
    states = list(product([0, 1], repeat=2))  # target, nuisance
    prior = np.asarray(
        [
            (target_one_probability if target else 1 - target_one_probability) * 0.5
            for target, nuisance in states
        ],
        dtype=float,
    )
    private = (np.zeros(len(states), dtype=int), np.zeros(len(states), dtype=int))
    features = np.asarray(
        [[nuisance for _, nuisance in states], [target for target, _ in states]],
        dtype=object,
    )
    payoffs = np.zeros((len(states), 2, 2))
    for s, (target, _) in enumerate(states):
        payoffs[s, target, target] = 1.0
    return TeamGame(
        prior,
        private,
        features,
        payoffs,
        feature_names=("fair nuisance", "biased target"),
        state_names=tuple(map(str, states)),
    )


def greedy_trap_game(delta: float = 0.05) -> TeamGame:
    """Parity complementarity plus a small positive distractor.

    With budget two, marginal-value greedy first chooses the distractor and can
    no longer acquire both parity shares.  Its gain ratio is delta/(1-delta).
    """
    if not 0 < delta < 0.5:
        raise ValueError("delta must lie in (0, 1/2)")
    states: list[tuple[str, int, int]] = []
    prior: list[float] = []
    for b1, b2 in product([0, 1], repeat=2):
        states.append(("core", b1, b2))
        prior.append((1 - delta) / 4)
    for d in [0, 1]:
        states.append(("decoy", d, 0))
        prior.append(delta / 2)

    private_labels = np.asarray([state[0] for state in states], dtype=object)
    private = (private_labels.copy(), private_labels.copy())
    feature_a = [b1 if kind == "core" else "e" for kind, b1, b2 in states]
    feature_b = [b2 if kind == "core" else "e" for kind, b1, b2 in states]
    distractor = [b1 if kind == "decoy" else "e" for kind, b1, b2 in states]
    features = np.asarray([feature_a, feature_b, distractor], dtype=object)

    payoffs = np.zeros((len(states), 2, 2))
    for s, (kind, x, y) in enumerate(states):
        target = x ^ y if kind == "core" else x
        payoffs[s, target, target] = 1.0
    return TeamGame(
        np.asarray(prior),
        private,
        features,
        payoffs,
        feature_names=("parity share A", "parity share B", "small distractor"),
        state_names=tuple(map(str, states)),
    )


def maximum_coverage_game(
    cover_sets: Sequence[Iterable[int]],
    weights: Sequence[float],
) -> TeamGame:
    weights_array = np.asarray(weights, dtype=float)
    if np.any(weights_array < 0) or weights_array.sum() <= 0:
        raise ValueError("weights must be nonnegative and nonzero")
    weights_array = weights_array / weights_array.sum()
    contexts = range(len(weights_array))
    normalized_sets = [set(x) for x in cover_sets]

    states = [(k, bit) for k in contexts for bit in [0, 1]]
    prior = np.asarray([weights_array[k] / 2 for k, bit in states])
    private_labels = np.asarray([k for k, bit in states], dtype=object)
    private = (private_labels.copy(), private_labels.copy())
    features = np.asarray(
        [
            [bit if k in covered else "erasure" for k, bit in states]
            for covered in normalized_sets
        ],
        dtype=object,
    )
    payoffs = np.zeros((len(states), 2, 2))
    for s, (k, bit) in enumerate(states):
        payoffs[s, bit, bit] = 1.0
    return TeamGame(
        prior,
        private,
        features,
        payoffs,
        feature_names=tuple(f"sensor {j}" for j in range(len(normalized_sets))),
        state_names=tuple(map(str, states)),
    )


def random_revelation_instance(
    seed: int,
    num_contexts: int = 8,
    num_features: int = 12,
) -> RevelationInstance:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(num_contexts))
    baseline = rng.uniform(0.45, 0.72, size=num_contexts)
    # Sparse-but-overlapping context coverage with heterogeneous reliability.
    mask = rng.random((num_contexts, num_features)) < rng.uniform(0.25, 0.55)
    reliability = rng.uniform(0.45, 0.98, size=(num_contexts, num_features))
    reveal = mask * reliability
    # Ensure every context is covered by at least one candidate sensor.
    for k in range(num_contexts):
        if not np.any(reveal[k] > 0):
            j = int(rng.integers(num_features))
            reveal[k, j] = rng.uniform(0.5, 0.95)
    return RevelationInstance(weights, baseline, reveal)


def emergency_response_game(
    seed: int = 0,
    severe_probability_scale: float = 1.0,
) -> TeamGame:
    """One-step two-site emergency response with asymmetric local observations.

    Each responder privately observes the severity at one site.  One responder
    can extinguish a mild incident, while both are required for a severe one.
    Public alarms can reveal coordination-critical global conditions.  Two fair
    weather bits are payoff-irrelevant but information-rich decoys.
    """
    rng = np.random.default_rng(seed)
    # Most incidents are absent or mild; severe incidents are rarer but carry a
    # much larger reward.  Small random perturbations create a family of priors.
    base_a = np.asarray([0.50, 0.39, 0.11 * severe_probability_scale])
    base_b = np.asarray([0.52, 0.37, 0.11 * severe_probability_scale])
    base_a /= base_a.sum()
    base_b /= base_b.sum()
    p_a = rng.dirichlet(35 * base_a + 0.5)
    p_b = rng.dirichlet(35 * base_b + 0.5)

    states = list(product(range(3), range(3), [0, 1], [0, 1]))
    prior = np.asarray([p_a[a] * p_b[b] * 0.25 for a, b, w1, w2 in states])
    private_a = np.asarray([a for a, b, w1, w2 in states], dtype=object)
    private_b = np.asarray([b for a, b, w1, w2 in states], dtype=object)

    features = np.asarray(
        [
            [int(a == 2) for a, b, w1, w2 in states],
            [int(b == 2) for a, b, w1, w2 in states],
            [int(a == 1 and b == 1) for a, b, w1, w2 in states],
            [a for a, b, w1, w2 in states],
            [b for a, b, w1, w2 in states],
            [w1 for a, b, w1, w2 in states],
            [w2 for a, b, w1, w2 in states],
            [(a + b) % 2 for a, b, w1, w2 in states],
            [int(a > 0) for a, b, w1, w2 in states],
            [int(b > 0) for a, b, w1, w2 in states],
            [np.sign(a - b).item() for a, b, w1, w2 in states],
        ],
        dtype=object,
    )

    # Actions: 0 -> site A, 1 -> site B.  Severe incidents require both agents.
    # Severe value is high enough that the rare coordination alarm matters.
    mild_value = rng.uniform(0.9, 1.1, size=2)
    severe_value = rng.uniform(3.7, 4.5, size=2)
    payoffs = np.zeros((len(states), 2, 2))
    for s, (a, b, w1, w2) in enumerate(states):
        for action_0, action_1 in product([0, 1], repeat=2):
            count_a = int(action_0 == 0) + int(action_1 == 0)
            count_b = 2 - count_a
            reward = 0.0
            if a == 1 and count_a >= 1:
                reward += mild_value[0]
            elif a == 2 and count_a >= 2:
                reward += severe_value[0]
            if b == 1 and count_b >= 1:
                reward += mild_value[1]
            elif b == 2 and count_b >= 2:
                reward += severe_value[1]
            payoffs[s, action_0, action_1] = reward

    return TeamGame(
        prior,
        (private_a, private_b),
        features,
        payoffs,
        feature_costs=np.asarray([1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2], dtype=float),
        feature_names=(
            "severe alarm A",
            "severe alarm B",
            "both mild",
            "exact severity A",
            "exact severity B",
            "weather bit 1",
            "weather bit 2",
            "severity parity",
            "any incident A",
            "any incident B",
            "severity comparison",
        ),
        state_names=tuple(map(str, states)),
    )


def operational_emergency_response_game(seed: int = 0) -> TeamGame:
    """Emergency-response benchmark containing only payoff-relevant state.

    The stress-test generator includes two independent telemetry bits.  Here we
    analytically marginalize those bits out and retain only the incident
    severities ``(a, b)``.  Every candidate public feature is therefore a
    deterministic statistic of payoff-relevant state.
    """
    full = emergency_response_game(seed)
    feature_indices = np.asarray([0, 1, 2, 7, 8, 9], dtype=int)

    # ``emergency_response_game`` enumerates (a, b, w1, w2), so each severity
    # pair occupies four consecutive states.  All operational observations,
    # features, and payoffs are constant across those four telemetry states.
    representatives = np.arange(0, full.num_states, 4, dtype=int)
    collapsed_prior = full.prior.reshape(-1, 4).sum(axis=1)
    severity_states = list(product(range(3), range(3)))
    private_a = np.asarray([a for a, _ in severity_states], dtype=object)
    private_b = np.asarray([b for _, b in severity_states], dtype=object)

    return TeamGame(
        collapsed_prior,
        (private_a, private_b),
        full.public_features[feature_indices][:, representatives],
        full.payoffs[representatives],
        full.feature_costs[feature_indices],
        tuple(full.feature_names[int(j)] for j in feature_indices),
        tuple(map(str, severity_states)),
    )


def monotone_access_game(minimal_authorized_sets: Sequence[Sequence[int]]) -> TeamGame:
    """Secret-sharing realization of a monotone access structure.

    Selecting a feature set reveals the hidden target bit exactly iff it contains
    one of ``minimal_authorized_sets``.  Otherwise the public signal is independent
    of the target.  This finite construction underpins the expressivity theorem.
    """
    hyperedges = [tuple(sorted(set(edge))) for edge in minimal_authorized_sets]
    if not hyperedges or any(not edge for edge in hyperedges):
        raise ValueError("minimal authorized sets must be nonempty")
    m = 1 + max(j for edge in hyperedges for j in edge)

    random_bit_count = sum(max(0, len(edge) - 1) for edge in hyperedges)
    states: list[tuple[int, tuple[int, ...]]] = []
    for secret in [0, 1]:
        for random_bits in product([0, 1], repeat=random_bit_count):
            states.append((secret, tuple(random_bits)))
    prior = np.full(len(states), 1.0 / len(states))

    feature_values: list[list[tuple[tuple[int, int], ...]]] = [[] for _ in range(m)]
    for secret, random_bits in states:
        per_feature: list[list[tuple[int, int]]] = [[] for _ in range(m)]
        cursor = 0
        for block, edge in enumerate(hyperedges):
            shares: list[int] = []
            if len(edge) > 1:
                shares.extend(random_bits[cursor : cursor + len(edge) - 1])
                cursor += len(edge) - 1
            last = secret
            for share in shares:
                last ^= share
            shares.append(last)
            for j, share in zip(edge, shares, strict=True):
                per_feature[j].append((block, int(share)))
        for j in range(m):
            feature_values[j].append(tuple(per_feature[j]))

    private = (np.zeros(len(states), dtype=int), np.zeros(len(states), dtype=int))
    payoffs = np.zeros((len(states), 2, 2))
    for s, (secret, _) in enumerate(states):
        payoffs[s, secret, secret] = 1.0
    feature_matrix = np.empty((m, len(states)), dtype=object)
    for j in range(m):
        for s, value in enumerate(feature_values[j]):
            feature_matrix[j, s] = value
    return TeamGame(
        prior,
        private,
        feature_matrix,
        payoffs,
        feature_names=tuple(f"share holder {j}" for j in range(m)),
    )
