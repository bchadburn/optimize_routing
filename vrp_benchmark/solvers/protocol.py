"""Shared solver protocol for CVRP benchmark."""
from __future__ import annotations

from typing import Protocol

from vrp_benchmark.data import CVRPInstance


class CVRPSolver(Protocol):
    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        """Solve a CVRP instance.

        Args:
            instance: The CVRP instance to solve.

        Returns:
            (routes, cost) where routes is a list of routes (each route is a list
            of customer node indices 1..n), and cost is the total distance.
            Returns ([], 1e9) on failure.
        """
        ...
