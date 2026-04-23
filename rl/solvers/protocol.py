"""Shared solver protocol — all CVRPTW solvers implement this interface."""
from __future__ import annotations

from typing import Protocol


class CvrptwSolver(Protocol):
    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        """Return total routing cost across all open DCs.

        Args:
            open_dc_ids: List of open DC indices.
            demands: customer_id -> demand quantity.
            transport_cost_d_to_c: dc_id -> {customer_id -> transport cost per unit}.
            n_vehicles_per_dc: Number of vehicles available per open DC.

        Returns:
            Total routing cost. Returns 1e6 if infeasible.
        """
        ...
