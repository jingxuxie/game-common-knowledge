"""Budgeted public observation design for decentralized team games."""

from .core import TeamGame
from .information import mutual_information_greedy, submodularity_ratio
from .optimization import (
    DesignResult,
    evaluate_subset,
    exact_subset_search,
    solve_design_milp,
    value_greedy,
)
from .revelation import RevelationInstance

__all__ = [
    "TeamGame",
    "DesignResult",
    "RevelationInstance",
    "evaluate_subset",
    "exact_subset_search",
    "mutual_information_greedy",
    "solve_design_milp",
    "submodularity_ratio",
    "value_greedy",
]
