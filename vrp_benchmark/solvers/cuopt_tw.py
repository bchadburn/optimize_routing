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

import os

from vrp_benchmark.data_tw import VRPTWInstance, route_cost_tw

_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/cuopt"


class CuOptVRPTWSolver:
    """VRPTW solver using NVIDIA cuOpt (self-hosted or NIM cloud API)."""

    def __init__(
        self,
        mode: str = "self-hosted",
        host: str = "localhost",
        port: int = 5000,
        time_limit_s: int = 30,
        api_key: str | None = None,
    ) -> None:
        self._time_limit_s = time_limit_s
        self._mode = mode
        if mode == "nim":
            self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
            if not self._api_key:
                raise ValueError("NIM mode requires NVIDIA_API_KEY env var or api_key argument")
            self._client = None
        else:
            from cuopt_sh_client import CuOptServiceSelfHostClient
            self._client = CuOptServiceSelfHostClient(ip=host, port=port, polling_timeout=None)

    def _request_nim(self, problem_data: dict) -> dict:
        import json
        import urllib.request
        body = json.dumps({"action": "cuOpt_OptimizedRouting", "data": problem_data}).encode()
        req = urllib.request.Request(
            _NIM_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())

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
            if self._mode == "nim":
                response = self._request_nim(problem_data)
                solver_resp = response.get("response", {}).get("solver_response", response)
            else:
                response = self._client.get_optimized_routes(problem_data)
                solver_resp = response["response"]["solver_response"]

            if solver_resp.get("status", -1) != 0:
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
