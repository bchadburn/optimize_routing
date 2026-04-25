"""cuOpt VRPTW solver via self-hosted REST API.

Extends the CVRP solver with time window fields:
  - travel_time_matrix_data: same as cost matrix for Solomon (speed=1)
  - task_data.task_time_windows: [[ready_time, due_date], ...] per customer
  - task_data.service_times: [service_time, ...] per customer
  - fleet_data.vehicle_time_windows: [[depot_ready, depot_due]] per vehicle

cuOpt time window semantics match Solomon directly:
  task_time_windows[i] = [earliest_arrival, latest_arrival]
  = Solomon's [ready_time, due_date]

Requires a running cuOpt server:
    docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:26.4.0-cuda12.9-py3.13
"""
from __future__ import annotations

from vrp_benchmark.data_tw import VRPTWInstance, route_cost_tw


class CuOptVRPTWSolver:
    """VRPTW solver using NVIDIA cuOpt self-hosted service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 30,
    ) -> None:
        from cuopt_sh_client import CuOptServiceSelfHostClient

        self._client = CuOptServiceSelfHostClient(ip=host, port=port, polling_timeout=None)
        self._time_limit_s = time_limit_s

    def solve(self, instance: VRPTWInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        dist = instance.dist_matrix.tolist()  # (n+1)×(n+1), depot at 0

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

        try:
            response = self._client.get_optimized_routes(problem_data)
            solver_resp = response["response"]["solver_response"]
            if solver_resp["status"] != 0:
                return [], 1e9

            routes: list[list[int]] = []
            vehicle_data = solver_resp.get("vehicle_data", {})
            for _vid, vdata in vehicle_data.items():
                raw = vdata.get("route", [])
                route = [node for node in raw if node != 0]
                if route:
                    routes.append(route)

            result = route_cost_tw(instance, routes)
            cost = result.distance if result.feasible else 1e9
            return routes, cost
        except Exception:
            return [], 1e9
