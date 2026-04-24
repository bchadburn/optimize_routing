"""cuOpt CVRPTW solver — GPU-accelerated routing via cuOpt self-hosted service.

Uses the same per-DC decomposition and demand-weighted cost matrix as OrtoolsVrpSolver
so results are directly comparable. Requires a running cuOpt server (see README).

Start the server:
    docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:<version>

Default host/port: localhost:5000.
"""
from __future__ import annotations

import math

import numpy as np

from rl.solvers.ortools_vrp import _assign_customers_to_dcs


class CuOptVrpSolver:
    """Solve DC→customer routing as CVRPTW using NVIDIA cuOpt (self-hosted service)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 5,
        polling_timeout: int | None = None,
    ) -> None:
        from cuopt_sh_client import CuOptServiceSelfHostClient

        self._client = CuOptServiceSelfHostClient(
            ip=host, port=port, polling_timeout=polling_timeout
        )
        self._time_limit_s = time_limit_s

    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        if not demands or all(v == 0.0 for v in demands.values()):
            return 0.0

        cust_ids = sorted(demands.keys())
        dc_customers = _assign_customers_to_dcs(open_dc_ids, cust_ids, transport_cost_d_to_c)

        total_cost = 0.0
        for dc_id in open_dc_ids:
            assigned = dc_customers[dc_id]
            if not assigned:
                continue
            cost = _solve_single_dc_cuopt(
                client=self._client,
                customer_ids=assigned,
                demands=demands,
                costs_to_customers=transport_cost_d_to_c[dc_id],
                n_vehicles=n_vehicles_per_dc,
                time_limit_s=self._time_limit_s,
            )
            total_cost += cost

        return total_cost


def _solve_single_dc_cuopt(
    client,
    customer_ids: list[int],
    demands: dict[int, float],
    costs_to_customers: dict[int, float],
    n_vehicles: int,
    time_limit_s: int,
) -> float:
    """Run cuOpt CVRPTW for a single DC. Returns routing cost or 1e6 on failure.

    Cost matrix uses demand-weighted arcs (same scaling as OrtoolsVrpSolver) so
    the VRP objective is comparable to the LP flow formulation:
        arc cost = transport_cost × demand / 2
    Halved because the VRP round-trip (depot→customer→depot) equals the LP one-way cost.
    """
    n = len(customer_ids)
    if n == 0:
        return 0.0

    # Build (n+1) × (n+1) float cost matrix: row/col 0 is the depot.
    size = n + 1
    matrix = np.zeros((size, size), dtype=float)
    for i, ci in enumerate(customer_ids, start=1):
        arc_cost = costs_to_customers.get(ci, 1e6) * demands[ci] / 2
        matrix[0][i] = arc_cost
        matrix[i][0] = arc_cost
    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                matrix[i][j] = matrix[0][i] + matrix[0][j]

    total_demand = sum(demands[c] for c in customer_ids)
    vehicle_capacity = int(math.ceil(total_demand))

    problem_data: dict = {}

    # Cost matrix: single fleet type "0"
    problem_data["cost_matrix_data"] = {"data": {"0": matrix.tolist()}}

    # Fleet: all vehicles start and end at depot (index 0)
    problem_data["fleet_data"] = {
        "vehicle_locations": [[0, 0]] * n_vehicles,
        "capacities": [[vehicle_capacity] * n_vehicles],
    }

    # Tasks: customers at locations 1..n with their demands
    task_demands = [int(math.ceil(demands[ci])) for ci in customer_ids]
    problem_data["task_data"] = {
        "task_locations": list(range(1, n + 1)),
        "demand": [task_demands],
    }

    problem_data["solver_config"] = {"time_limit": time_limit_s}

    try:
        response = client.get_optimized_routes(problem_data)
        solver_resp = response["response"]["solver_response"]
        if solver_resp["status"] == 0:
            return float(solver_resp["solution_cost"])
    except Exception:
        pass

    return 1e6
