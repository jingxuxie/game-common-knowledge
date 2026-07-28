from __future__ import annotations

from itertools import combinations

import numpy as np

from common_knowledge.certificates import random_shared_failure_instance


def test_shared_failure_gain_is_monotone_and_submodular() -> None:
    instance = random_shared_failure_instance(
        3, num_contexts=4, num_domains=3, sensors_per_domain=2
    )
    features = range(instance.num_features)
    subsets = [
        subset
        for size in range(instance.num_features + 1)
        for subset in combinations(features, size)
    ]
    values = {subset: instance.gain(subset) for subset in subsets}

    for a in subsets:
        set_a = set(a)
        for b in subsets:
            set_b = set(b)
            if not set_a.issubset(set_b):
                continue
            assert values[a] <= values[b] + 1e-12
            for j in features:
                if j in set_b:
                    continue
                a_plus = tuple(sorted((*a, j)))
                b_plus = tuple(sorted((*b, j)))
                marginal_a = values[a_plus] - values[a]
                marginal_b = values[b_plus] - values[b]
                assert marginal_a + 1e-12 >= marginal_b


def test_true_greedy_obeys_cardinality_guarantee() -> None:
    threshold = 1.0 - 1.0 / np.e
    for seed in range(10):
        instance = random_shared_failure_instance(seed)
        optimum, optimum_gain = instance.optimum(4)
        greedy = instance.greedy(4)
        assert optimum
        assert instance.gain(greedy) / optimum_gain >= threshold - 1e-12


def test_independence_surrogate_can_overselect_one_domain() -> None:
    instance = random_shared_failure_instance(8)
    optimum, optimum_gain = instance.optimum(4)
    surrogate = instance.greedy(4, assume_independent=True)
    true_greedy = instance.greedy(4)

    assert len(set(instance.sensor_domains[list(true_greedy)])) == 3
    assert instance.gain(true_greedy) >= instance.gain(surrogate)
    assert instance.gain(surrogate) / optimum_gain < 0.95
    assert optimum_gain > 0
