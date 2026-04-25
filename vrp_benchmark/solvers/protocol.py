"""Solver protocols for CVRP and VRPTW benchmarks."""
from __future__ import annotations

from typing import Protocol

from vrp_benchmark.data import CVRPInstance
from vrp_benchmark.data_tw import VRPTWInstance


class CVRPSolver(Protocol):
    """Protocol for CVRP solvers. Returns (routes, cost); ([], 1e9) on failure."""

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]: ...


class VRPTWSolver(Protocol):
    """Protocol for VRPTW solvers. Returns (routes, distance); ([], 1e9) on failure."""

    def solve(self, instance: VRPTWInstance) -> tuple[list[list[int]], float]: ...
