from __future__ import annotations

import itertools

import numpy as np

from common_knowledge.benchmarks import random_revelation_instance


def test_revelation_value_is_monotone_and_submodular() -> None:
    instance = random_revelation_instance(3, num_contexts=5, num_features=7)
    values = {}
    for mask in range(1 << instance.num_features):
        subset = frozenset(j for j in range(instance.num_features) if mask & (1 << j))
        values[subset] = instance.value(subset)

    for left, left_value in values.items():
        for right, right_value in values.items():
            if left <= right:
                assert left_value <= right_value + 1e-12

    universe = set(range(instance.num_features))
    for left in values:
        for right in values:
            if not left <= right:
                continue
            for j in universe - right:
                left_gain = values[left | {j}] - values[left]
                right_gain = values[right | {j}] - values[right]
                assert left_gain + 1e-12 >= right_gain


def test_revelation_greedy_respects_classical_bound() -> None:
    for seed in range(20):
        instance = random_revelation_instance(seed, num_contexts=6, num_features=9)
        budget = 3
        greedy = instance.greedy(budget)
        optimum = instance.optimum(budget)
        base = instance.value(())
        denominator = instance.value(optimum) - base
        if denominator > 1e-12:
            ratio = (instance.value(greedy) - base) / denominator
            assert ratio >= 1 - 1 / np.e - 1e-10
