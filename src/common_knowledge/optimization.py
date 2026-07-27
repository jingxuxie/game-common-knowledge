"""Exact MILP and feature-selection algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .core import TeamGame


@dataclass(frozen=True)
class DesignResult:
    value: float
    subset: tuple[int, ...]
    actions_by_state: tuple[tuple[int, ...], ...]
    success: bool
    message: str
    runtime_seconds: float
    mip_gap: float | None = None
    mip_nodes: int | None = None


class _ConstraintBuilder:
    def __init__(self, nvars: int) -> None:
        self.nvars = nvars
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []

    def add(self, coeffs: dict[int, float], lower: float = -np.inf, upper: float = np.inf) -> None:
        row = len(self.lb)
        for col, value in coeffs.items():
            if abs(value) > 0:
                self.rows.append(row)
                self.cols.append(col)
                self.data.append(float(value))
        self.lb.append(float(lower))
        self.ub.append(float(upper))

    def build(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.lb), self.nvars),
            dtype=float,
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lb), np.asarray(self.ub))



def _solve_fixed_subset_milp(
    game: TeamGame,
    subset: tuple[int, ...],
    time_limit: float | None,
    mip_rel_gap: float,
) -> DesignResult:
    """Compact exact MILP for one fixed public feature set."""
    s_count = game.num_states
    joint_actions = game.joint_actions
    j_count = len(joint_actions)

    cell_of_state: dict[tuple[int, int], int] = {}
    cells_per_agent: list[list[tuple[object, ...]]] = []
    for i in range(game.num_agents):
        keys: list[tuple[object, ...]] = []
        key_to_cell: dict[tuple[object, ...], int] = {}
        for state in range(s_count):
            key = game.information_key(i, state, subset)
            if key not in key_to_cell:
                key_to_cell[key] = len(keys)
                keys.append(key)
            cell_of_state[i, state] = key_to_cell[key]
        cells_per_agent.append(keys)

    cursor = 0
    y_index: dict[tuple[int, int, int], int] = {}
    for i, action_size in enumerate(game.action_sizes):
        for cell in range(len(cells_per_agent[i])):
            for action in range(action_size):
                y_index[i, cell, action] = cursor
                cursor += 1
    q_index: dict[tuple[int, int], int] = {}
    for state in range(s_count):
        for ja in range(j_count):
            q_index[state, ja] = cursor
            cursor += 1
    nvars = cursor

    objective = np.zeros(nvars, dtype=float)
    for state in range(s_count):
        for ja, actions in enumerate(joint_actions):
            objective[q_index[state, ja]] = -game.prior[state] * game.payoffs[(state,) + actions]

    builder = _ConstraintBuilder(nvars)
    for i, action_size in enumerate(game.action_sizes):
        for cell in range(len(cells_per_agent[i])):
            builder.add(
                {y_index[i, cell, action]: 1.0 for action in range(action_size)},
                lower=1.0,
                upper=1.0,
            )

    for state in range(s_count):
        builder.add(
            {q_index[state, ja]: 1.0 for ja in range(j_count)},
            lower=1.0,
            upper=1.0,
        )
        for ja, actions in enumerate(joint_actions):
            q = q_index[state, ja]
            for i, action in enumerate(actions):
                cell = cell_of_state[i, state]
                builder.add({q: 1.0, y_index[i, cell, action]: -1.0}, upper=0.0)

    options: dict[str, float | bool] = {"disp": False, "mip_rel_gap": float(mip_rel_gap)}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    start = perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(nvars, dtype=int),
        bounds=Bounds(np.zeros(nvars), np.ones(nvars)),
        constraints=builder.build(),
        options=options,
    )
    runtime = perf_counter() - start
    if result.x is None:
        return DesignResult(
            value=float("-inf"),
            subset=subset,
            actions_by_state=(),
            success=False,
            message=str(result.message),
            runtime_seconds=runtime,
            mip_gap=getattr(result, "mip_gap", None),
            mip_nodes=getattr(result, "mip_node_count", None),
        )

    actions_by_state: list[tuple[int, ...]] = []
    for state in range(s_count):
        actions: list[int] = []
        for i, action_size in enumerate(game.action_sizes):
            cell = cell_of_state[i, state]
            action = int(np.argmax([result.x[y_index[i, cell, a]] for a in range(action_size)]))
            actions.append(action)
        actions_by_state.append(tuple(actions))
    return DesignResult(
        value=float(-result.fun),
        subset=subset,
        actions_by_state=tuple(actions_by_state),
        success=bool(result.success),
        message=str(result.message),
        runtime_seconds=runtime,
        mip_gap=getattr(result, "mip_gap", None),
        mip_nodes=getattr(result, "mip_node_count", None),
    )


def solve_design_milp(
    game: TeamGame,
    budget: float,
    fixed_subset: Iterable[int] | None = None,
    time_limit: float | None = None,
    mip_rel_gap: float = 0.0,
) -> DesignResult:
    """Jointly optimize public features and deterministic decentralized policies.

    If ``fixed_subset`` is supplied, feature-selection variables are fixed to
    that subset and the routine computes its exact decentralized team value.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    fixed = None if fixed_subset is None else game.validate_subset(fixed_subset, budget)
    if fixed is not None:
        return _solve_fixed_subset_milp(game, fixed, time_limit, mip_rel_gap)

    m = game.num_features
    s_count = game.num_states
    joint_actions = game.joint_actions
    j_count = len(joint_actions)

    x_index = {j: j for j in range(m)}
    cursor = m
    y_index: dict[tuple[int, int, int], int] = {}
    for i, action_size in enumerate(game.action_sizes):
        for s in range(s_count):
            for a in range(action_size):
                y_index[i, s, a] = cursor
                cursor += 1
    q_index: dict[tuple[int, int], int] = {}
    for s in range(s_count):
        for ja in range(j_count):
            q_index[s, ja] = cursor
            cursor += 1
    nvars = cursor

    objective = np.zeros(nvars, dtype=float)
    for s in range(s_count):
        for ja, actions in enumerate(joint_actions):
            payoff = game.payoffs[(s,) + actions]
            objective[q_index[s, ja]] = -game.prior[s] * payoff

    lower = np.zeros(nvars, dtype=float)
    upper = np.ones(nvars, dtype=float)
    if fixed is not None:
        fixed_set = set(fixed)
        for j in range(m):
            lower[x_index[j]] = upper[x_index[j]] = float(j in fixed_set)

    builder = _ConstraintBuilder(nvars)
    builder.add(
        {x_index[j]: float(game.feature_costs[j]) for j in range(m)},
        upper=float(budget),
    )

    # Exactly one action for each agent and state.
    for i, action_size in enumerate(game.action_sizes):
        for s in range(s_count):
            builder.add(
                {y_index[i, s, a]: 1.0 for a in range(action_size)},
                lower=1.0,
                upper=1.0,
            )

    # Feature-controlled nonanticipativity.  If private observations coincide
    # and no selected public feature separates two states, actions must coincide.
    for i, action_size in enumerate(game.action_sizes):
        obs = game.private_observations[i]
        groups: dict[object, list[int]] = {}
        for s in range(s_count):
            groups.setdefault(obs[s], []).append(s)
        for states in groups.values():
            for left_pos in range(len(states)):
                for right_pos in range(left_pos + 1, len(states)):
                    s, t = states[left_pos], states[right_pos]
                    distinguishing = [
                        j
                        for j in range(m)
                        if game.public_features[j, s] != game.public_features[j, t]
                    ]
                    for a in range(action_size):
                        coeffs = {
                            y_index[i, s, a]: 1.0,
                            y_index[i, t, a]: -1.0,
                        }
                        for j in distinguishing:
                            coeffs[x_index[j]] = coeffs.get(x_index[j], 0.0) - 1.0
                        builder.add(coeffs, upper=0.0)

                        coeffs_reverse = {
                            y_index[i, t, a]: 1.0,
                            y_index[i, s, a]: -1.0,
                        }
                        for j in distinguishing:
                            coeffs_reverse[x_index[j]] = coeffs_reverse.get(x_index[j], 0.0) - 1.0
                        builder.add(coeffs_reverse, upper=0.0)

    # One realized joint action per state, consistent with each local action.
    for s in range(s_count):
        builder.add(
            {q_index[s, ja]: 1.0 for ja in range(j_count)},
            lower=1.0,
            upper=1.0,
        )
        for ja, actions in enumerate(joint_actions):
            q = q_index[s, ja]
            for i, action in enumerate(actions):
                builder.add({q: 1.0, y_index[i, s, action]: -1.0}, upper=0.0)

    options: dict[str, float | bool] = {"disp": False, "mip_rel_gap": float(mip_rel_gap)}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    start = perf_counter()
    result = milp(
        c=objective,
        integrality=np.ones(nvars, dtype=int),
        bounds=Bounds(lower, upper),
        constraints=builder.build(),
        options=options,
    )
    runtime = perf_counter() - start

    if result.x is None:
        return DesignResult(
            value=float("-inf"),
            subset=(),
            actions_by_state=(),
            success=False,
            message=str(result.message),
            runtime_seconds=runtime,
            mip_gap=getattr(result, "mip_gap", None),
            mip_nodes=getattr(result, "mip_node_count", None),
        )

    selected = tuple(j for j in range(m) if result.x[x_index[j]] > 0.5)
    actions_by_state: list[tuple[int, ...]] = []
    for s in range(s_count):
        actions = []
        for i, action_size in enumerate(game.action_sizes):
            chosen = int(np.argmax([result.x[y_index[i, s, a]] for a in range(action_size)]))
            actions.append(chosen)
        actions_by_state.append(tuple(actions))

    return DesignResult(
        value=float(-result.fun),
        subset=selected,
        actions_by_state=tuple(actions_by_state),
        success=bool(result.success),
        message=str(result.message),
        runtime_seconds=runtime,
        mip_gap=getattr(result, "mip_gap", None),
        mip_nodes=getattr(result, "mip_node_count", None),
    )


def exact_subset_search(
    game: TeamGame,
    budget: float,
    time_limit_per_subset: float | None = None,
) -> DesignResult:
    """Enumerate feasible feature sets and solve the induced team problem exactly."""
    best: DesignResult | None = None
    start = perf_counter()
    m = game.num_features
    for r in range(m + 1):
        for subset in combinations(range(m), r):
            if game.feature_costs[list(subset)].sum() > budget + 1e-9:
                continue
            result = solve_design_milp(
                game,
                budget=budget,
                fixed_subset=subset,
                time_limit=time_limit_per_subset,
            )
            if not result.success:
                raise RuntimeError(f"MILP failed for subset {subset}: {result.message}")
            if best is None or result.value > best.value + 1e-10:
                best = result
    assert best is not None
    return DesignResult(
        value=best.value,
        subset=best.subset,
        actions_by_state=best.actions_by_state,
        success=True,
        message="exact subset enumeration",
        runtime_seconds=perf_counter() - start,
        mip_gap=0.0,
        mip_nodes=None,
    )


def value_greedy(game: TeamGame, budget: float, tol: float = 1e-10) -> DesignResult:
    """Greedy selection by exact marginal team value per unit cost."""
    selected: tuple[int, ...] = ()
    current = solve_design_milp(game, budget, fixed_subset=selected)
    total_runtime = current.runtime_seconds
    while True:
        candidates = [
            j
            for j in range(game.num_features)
            if j not in selected
            and game.feature_costs[list(selected) + [j]].sum() <= budget + 1e-9
        ]
        if not candidates:
            break
        scored: list[tuple[float, float, int, DesignResult]] = []
        for j in candidates:
            trial = tuple(sorted(selected + (j,)))
            result = solve_design_milp(game, budget, fixed_subset=trial)
            total_runtime += result.runtime_seconds
            marginal = result.value - current.value
            scored.append((marginal / game.feature_costs[j], marginal, -j, result))
        _, marginal, _, best_result = max(scored, key=lambda item: item[:3])
        if marginal <= tol:
            break
        selected = best_result.subset
        current = best_result

    return DesignResult(
        value=current.value,
        subset=selected,
        actions_by_state=current.actions_by_state,
        success=current.success,
        message="exact marginal-value greedy",
        runtime_seconds=total_runtime,
        mip_gap=current.mip_gap,
        mip_nodes=current.mip_nodes,
    )


def evaluate_subset(game: TeamGame, subset: Iterable[int], budget: float | None = None) -> DesignResult:
    selected = game.validate_subset(subset, budget)
    fixed_budget = float(game.feature_costs[list(selected)].sum()) if budget is None else budget
    return solve_design_milp(game, fixed_budget, fixed_subset=selected)
