"""cuOpt CVRP solver — self-hosted Docker or NVIDIA NIM cloud API.

Two modes:
  mode="self-hosted"  Requires a running cuOpt Docker container:
      docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:26.4.0-cuda12.9-py3.13

  mode="nim"          Uses the NVIDIA-hosted NIM API (no local GPU needed).
      Reads NVIDIA_API_KEY from the environment. Get a free key at:
      https://build.nvidia.com/nvidia/cuopt
"""
from __future__ import annotations

from vrp_benchmark.data import CVRPInstance, route_cost
from vrp_benchmark.solvers._cuopt_base import CuOptClientMixin


class CuOptSolver(CuOptClientMixin):
    """CVRP solver using NVIDIA cuOpt (self-hosted or NIM cloud API)."""

    def __init__(
        self,
        mode: str = "self-hosted",
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 120,
        api_key: str | None = None,
    ) -> None:
        self._init_client(mode, host, port, time_limit_s, api_key)

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        dist = instance.dist_matrix.tolist()

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

        solver_resp = self._send(problem_data)
        if solver_resp is None:
            return [], 1e9

        routes = self._extract_routes(solver_resp)
        return routes, route_cost(instance, routes)
