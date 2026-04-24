"""cuOpt CVRP solver via self-hosted REST API.

Requires a running cuOpt server:
    docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:26.4.0-cuda12.9-py3.13

Sends the full distance matrix in a single request (one call per instance),
unlike the supply chain solver which decomposed per-DC. This gives cuOpt the best
chance to show GPU parallelism at large scale.
"""
from __future__ import annotations

from vrp_benchmark.data import CVRPInstance, route_cost


class CuOptSolver:
    """CVRP solver using NVIDIA cuOpt self-hosted service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 120,
    ) -> None:
        from cuopt_sh_client import CuOptServiceSelfHostClient

        self._client = CuOptServiceSelfHostClient(ip=host, port=port, polling_timeout=None)
        self._time_limit_s = time_limit_s

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        dist = instance.dist_matrix.tolist()  # (n+1) × (n+1), depot at index 0

        problem_data: dict = {
            "cost_matrix_data": {"data": {"0": dist}},
            "fleet_data": {
                "vehicle_locations": [[0, 0]] * instance.n_vehicles,
                "capacities": [[instance.capacity] * instance.n_vehicles],
            },
            "task_data": {
                "task_locations": list(range(1, n + 1)),
                "demand": [[int(d) for d in instance.demands]],
            },
            "solver_config": {"time_limit": self._time_limit_s},
        }

        try:
            response = self._client.get_optimized_routes(problem_data)
            solver_resp = response["response"]["solver_response"]
            if solver_resp["status"] != 0:
                return [], 1e9

            routes: list[list[int]] = []
            vehicle_data = solver_resp.get("vehicle_data", {})
            for _vid, vdata in vehicle_data.items():
                # route includes depot (0) at start/end — strip them
                raw = vdata.get("route", [])
                route = [node for node in raw if node != 0]
                if route:
                    routes.append(route)

            cost = route_cost(instance, routes)
            return routes, cost
        except Exception:
            return [], 1e9
