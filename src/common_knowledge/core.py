"""Core finite-team-game representation for budgeted public observation design.

The model is intentionally finite and explicit.  A world state is sampled from a
known prior.  Agent i observes a private signal o_i(theta).  The designer chooses
which deterministic state features are publicly broadcast; their realizations
are then observed by every agent.  Agents use decentralized policies conditioned
only on their private observation and the selected public feature values.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Hashable, Iterable, Sequence

import numpy as np


Signal = Hashable


@dataclass(frozen=True)
class TeamGame:
    """A finite common-payoff Bayesian team game.

    Parameters
    ----------
    prior:
        Probability vector over states, shape ``(num_states,)``.
    private_observations:
        One signal array per agent; array i has shape ``(num_states,)``.
    public_features:
        Candidate feature matrix with shape ``(num_features, num_states)``.
        Entries may be any hashable Python objects.
    payoffs:
        Common payoff tensor with shape
        ``(num_states, action_sizes[0], ..., action_sizes[n-1])``.
    feature_costs:
        Positive cost per public feature.  Defaults to one.
    feature_names:
        Human-readable feature names.  Defaults to ``f0, f1, ...``.
    state_names:
        Optional human-readable state names.
    """

    prior: np.ndarray
    private_observations: tuple[np.ndarray, ...]
    public_features: np.ndarray
    payoffs: np.ndarray
    feature_costs: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    state_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior, dtype=float)
        if prior.ndim != 1 or prior.size == 0:
            raise ValueError("prior must be a nonempty one-dimensional array")
        if np.any(prior < -1e-12):
            raise ValueError("prior probabilities must be nonnegative")
        total = float(prior.sum())
        if total <= 0:
            raise ValueError("prior must have positive mass")
        prior = prior / total
        object.__setattr__(self, "prior", prior)

        private = tuple(np.asarray(obs, dtype=object) for obs in self.private_observations)
        if not private:
            raise ValueError("at least one agent is required")
        if any(obs.shape != prior.shape for obs in private):
            raise ValueError("each private-observation array must match prior shape")
        object.__setattr__(self, "private_observations", private)

        features = np.asarray(self.public_features, dtype=object)
        if features.ndim != 2 or features.shape[1] != prior.size:
            raise ValueError("public_features must have shape (num_features, num_states)")
        object.__setattr__(self, "public_features", features)

        payoffs = np.asarray(self.payoffs, dtype=float)
        if payoffs.ndim != len(private) + 1 or payoffs.shape[0] != prior.size:
            raise ValueError(
                "payoffs must have shape (num_states, action_size_1, ..., action_size_n)"
            )
        if any(size <= 0 for size in payoffs.shape[1:]):
            raise ValueError("each agent must have at least one action")
        object.__setattr__(self, "payoffs", payoffs)

        m = features.shape[0]
        costs = np.ones(m, dtype=float) if self.feature_costs is None else np.asarray(self.feature_costs, dtype=float)
        if costs.shape != (m,) or np.any(costs <= 0):
            raise ValueError("feature_costs must be a positive vector of length num_features")
        object.__setattr__(self, "feature_costs", costs)

        names = self.feature_names or tuple(f"f{j}" for j in range(m))
        if len(names) != m:
            raise ValueError("feature_names must have length num_features")
        object.__setattr__(self, "feature_names", tuple(names))

        if self.state_names is not None and len(self.state_names) != prior.size:
            raise ValueError("state_names must have length num_states")

    @property
    def num_states(self) -> int:
        return int(self.prior.size)

    @property
    def num_agents(self) -> int:
        return len(self.private_observations)

    @property
    def num_features(self) -> int:
        return int(self.public_features.shape[0])

    @property
    def action_sizes(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.payoffs.shape[1:])

    @property
    def joint_actions(self) -> tuple[tuple[int, ...], ...]:
        return tuple(product(*(range(size) for size in self.action_sizes)))

    def selected_signal(self, state: int, subset: Iterable[int]) -> tuple[Signal, ...]:
        selected = tuple(sorted(set(int(j) for j in subset)))
        return tuple(self.public_features[j, state] for j in selected)

    def information_key(self, agent: int, state: int, subset: Iterable[int]) -> tuple[Signal, ...]:
        return (self.private_observations[agent][state],) + self.selected_signal(state, subset)

    def information_partition(self, agent: int, subset: Iterable[int]) -> dict[tuple[Signal, ...], list[int]]:
        cells: dict[tuple[Signal, ...], list[int]] = {}
        for state in range(self.num_states):
            cells.setdefault(self.information_key(agent, state, subset), []).append(state)
        return cells

    def validate_subset(self, subset: Iterable[int], budget: float | None = None) -> tuple[int, ...]:
        selected = tuple(sorted(set(int(j) for j in subset)))
        if any(j < 0 or j >= self.num_features for j in selected):
            raise ValueError("feature index out of range")
        if budget is not None and float(self.feature_costs[list(selected)].sum()) > budget + 1e-9:
            raise ValueError("subset exceeds budget")
        return selected


def payoff_for_actions(game: TeamGame, state: int, actions: Sequence[int]) -> float:
    """Return the common payoff for one state and joint action."""
    if len(actions) != game.num_agents:
        raise ValueError("one action is required per agent")
    return float(game.payoffs[(state,) + tuple(int(a) for a in actions)])
