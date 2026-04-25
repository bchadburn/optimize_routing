"""cuOpt VRPTW solver — self-hosted Docker or NVIDIA NIM cloud API.

Extends the CVRP solver with time window fields:
  - travel_time_matrix_data: same as cost matrix for Solomon (speed=1)
  - task_data.task_time_windows: [[ready_time, due_date], ...] per customer
  - task_data.service_times: [service_time, ...] per customer
  - fleet_data.vehicle_time_windows: [[depot_ready, depot_due]] per vehicle

Two modes:
  mode="self-hosted"  Requires a running cuOpt Docker container.
  mode="nim"          Uses the NVIDIA NIM API (reads NVIDIA_API_KEY from env).
                      Get a free key at: https://build.nvidia.com/nvidia/cuopt
"""
from __future__ import annotations

from vrp_benchmark.data_tw import VRPTWInstance, route_cost_tw
from vrp_benchmark.solvers._cuopt_base import CuOptClientMixin


class CuOptVRPTWSolver(CuOptClientMixin):
    """VRPTW solver using NVIDIA cuOpt (self-hosted or NIM cloud API)."""

    def __init__(
        self,
        mode: str = "self-hosted",
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 30,
        api_key: str | None = None,
    ) -> None:
        self._init_client(mode, host, port, time_limit_s, api_key)

    def solve(self, instance: VRPTWInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        dist = instance.dist_matrix.tolist()

        # For Solomon, travel time = distance (speed=1), so both matrices are identical.
        problem_data: dict = {
            "cost_matrix_data": {"data": {"0": dist}},
            "travel_time_matrix_data": {"data": {"0": dist}},
            "fleet_data": {
                "vehicle_locations": [[0, 0]] * instance.n_vehicles,
                "capacities": [[int(instance.capacity)] * instance.n_vehicles],
                # shape [n_vehicles, 2]: [earliest_departure, latest_return]
                "vehicle_time_windows": [
                    [int(instance.ready_times[0]), int(instance.due_dates[0])]
                ] * instance.n_vehicles,
            },
            "task_data": {
                "task_locations": list(range(1, n + 1)),
                "demand": [[int(d) for d in instance.demands]],
                # shape [n_customers, 2]: [earliest_arrival, latest_arrival]
                "task_time_windows": [
                    [int(instance.ready_times[i]), int(instance.due_dates[i])]
                    for i in range(1, n + 1)
                ],
                "service_times": [int(instance.service_times[i]) for i in range(1, n + 1)],
            },
            "solver_config": {"time_limit": self._time_limit_s},
        }

        solver_resp = self._send(problem_data)
        if solver_resp is None:
            return [], 1e9

        routes = self._extract_routes(solver_resp)
        result = route_cost_tw(instance, routes)
        return routes, result.distance if result.feasible else 1e9
