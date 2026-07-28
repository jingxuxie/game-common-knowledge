from __future__ import annotations

import math

from common_knowledge.benchmarks import emergency_response_game
from common_knowledge.information import mutual_information_greedy
from common_knowledge.optimization import evaluate_subset, solve_design_milp, value_greedy


def test_emergency_joint_solver_and_greedy_are_feasible() -> None:
    game = emergency_response_game(2)
    optimum = solve_design_milp(game, budget=2)
    greedy = value_greedy(game, budget=2)
    assert optimum.success and greedy.success
    assert len(optimum.subset) <= 2
    assert len(greedy.subset) <= 2
    assert greedy.value <= optimum.value + 1e-8


def test_emergency_information_baselines_evaluate() -> None:
    game = emergency_response_game(4)
    for conditioned in [False, True]:
        subset = mutual_information_greedy(game, budget=1, private_conditioned=conditioned)
        result = evaluate_subset(game, subset)
        assert result.success
        assert math.isfinite(result.value)


def test_operational_emergency_has_no_nuisance_features() -> None:
    from common_knowledge.benchmarks import operational_emergency_response_game

    game = operational_emergency_response_game(0)
    assert all("weather" not in name for name in game.feature_names)
    assert game.num_features == 6
    assert game.num_states == 9
    assert all(len(name.strip("()").split(",")) == 2 for name in game.state_names)
