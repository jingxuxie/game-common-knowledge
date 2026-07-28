from __future__ import annotations

import math

import numpy as np

from common_knowledge.benchmarks import (
    greedy_trap_game,
    information_decoy_game,
    maximum_coverage_game,
    monotone_access_game,
    parity_game,
)
from common_knowledge.information import mutual_information_greedy
from common_knowledge.optimization import evaluate_subset, exact_subset_search, solve_design_milp, value_greedy


def test_parity_is_complementary() -> None:
    game = parity_game()
    values = {
        subset: evaluate_subset(game, subset).value
        for subset in [(), (0,), (1,), (0, 1)]
    }
    assert math.isclose(values[()], 0.5, abs_tol=1e-8)
    assert math.isclose(values[(0,)], 0.5, abs_tol=1e-8)
    assert math.isclose(values[(1,)], 0.5, abs_tol=1e-8)
    assert math.isclose(values[(0, 1)], 1.0, abs_tol=1e-8)
    assert values[(0, 1)] - values[(0,)] > values[(1,)] - values[()]


def test_information_decoy_mi_selects_wrong_feature() -> None:
    game = information_decoy_game(0.1)
    mi = mutual_information_greedy(game, budget=1)
    optimal = solve_design_milp(game, budget=1)
    assert mi == (0,)
    assert optimal.subset == (1,)
    baseline = evaluate_subset(game, ()).value
    mi_value = evaluate_subset(game, mi).value
    assert math.isclose(mi_value, baseline, abs_tol=1e-8)
    assert optimal.value > mi_value + 0.09


def test_greedy_trap_has_predicted_ratio() -> None:
    delta = 0.05
    game = greedy_trap_game(delta)
    greedy = value_greedy(game, budget=2)
    optimum = solve_design_milp(game, budget=2)
    baseline = evaluate_subset(game, ()).value
    ratio = (greedy.value - baseline) / (optimum.value - baseline)
    assert greedy.subset == (2,)
    assert optimum.subset == (0, 1)
    assert math.isclose(ratio, delta / (1 - delta), rel_tol=1e-6, abs_tol=1e-8)


def test_maximum_coverage_value_formula() -> None:
    cover_sets = [{0, 1}, {1, 2}, {3}]
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    game = maximum_coverage_game(cover_sets, weights)
    subset = (0, 2)
    value = evaluate_subset(game, subset).value
    covered_weight = weights[[0, 1, 3]].sum()
    expected = 0.5 + 0.5 * covered_weight
    assert math.isclose(value, expected, abs_tol=1e-8)


def test_joint_milp_matches_subset_enumeration() -> None:
    game = maximum_coverage_game([{0, 1}, {1, 2}, {2, 3}, {0, 3}], [1, 2, 3, 4])
    joint = solve_design_milp(game, budget=2)
    exhaustive = exact_subset_search(game, budget=2)
    assert joint.success and exhaustive.success
    assert math.isclose(joint.value, exhaustive.value, abs_tol=1e-8)


def test_monotonicity_on_all_subsets() -> None:
    game = greedy_trap_game(0.08)
    values = {}
    for mask in range(1 << game.num_features):
        subset = tuple(j for j in range(game.num_features) if mask & (1 << j))
        values[frozenset(subset)] = evaluate_subset(game, subset).value
    for left, left_value in values.items():
        for right, right_value in values.items():
            if left <= right:
                assert left_value <= right_value + 1e-8


def test_access_structure_realization() -> None:
    game = monotone_access_game([(0, 1), (2, 3, 4)])
    unauthorized = [(), (0,), (1,), (2, 3), (0, 2, 4)]
    authorized = [(0, 1), (2, 3, 4), (0, 1, 4), (0, 2, 3, 4)]
    for subset in unauthorized:
        assert math.isclose(evaluate_subset(game, subset).value, 0.5, abs_tol=1e-8)
    for subset in authorized:
        assert math.isclose(evaluate_subset(game, subset).value, 1.0, abs_tol=1e-8)


def test_joint_milp_matches_exhaustive_on_random_general_games() -> None:
    from common_knowledge.core import TeamGame

    for seed in range(5):
        rng = np.random.default_rng(seed)
        num_states = 5
        prior = rng.dirichlet(np.ones(num_states))
        private = (
            rng.integers(0, 2, size=num_states).astype(object),
            rng.integers(0, 3, size=num_states).astype(object),
        )
        features = rng.integers(0, 2, size=(4, num_states)).astype(object)
        payoffs = rng.normal(size=(num_states, 2, 2))
        game = TeamGame(prior, private, features, payoffs)

        joint = solve_design_milp(game, budget=2)
        exhaustive = exact_subset_search(game, budget=2)
        assert joint.success and exhaustive.success
        assert math.isclose(joint.value, exhaustive.value, abs_tol=1e-8)

        selected = joint.subset
        for agent in range(game.num_agents):
            for left in range(game.num_states):
                for right in range(game.num_states):
                    if game.information_key(agent, left, selected) == game.information_key(
                        agent, right, selected
                    ):
                        assert joint.actions_by_state[left][agent] == joint.actions_by_state[right][agent]
