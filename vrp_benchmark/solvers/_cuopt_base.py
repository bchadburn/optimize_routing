"""Shared client setup and request logic for cuOpt solvers.

Both CuOptSolver (CVRP) and CuOptVRPTWSolver (VRPTW) inherit this mixin.
It handles:
  - mode detection (self-hosted vs NIM cloud API)
  - client initialisation
  - NIM HTTP request
  - response parsing (status check + route extraction from vehicle_data)
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

_NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/cuopt"


class CuOptClientMixin:
    """Mixin providing cuOpt client setup, NIM request, and response parsing."""

    def _init_client(
        self,
        mode: str,
        host: str,
        port: int,
        time_limit_s: int,
        api_key: str | None,
    ) -> None:
        self._time_limit_s = time_limit_s
        self._mode = mode
        if mode == "nim":
            self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
            if not self._api_key:
                raise ValueError(
                    "NIM mode requires NVIDIA_API_KEY env var or api_key argument. "
                    "Get a free key at https://build.nvidia.com/nvidia/cuopt"
                )
            self._client = None
        else:
            from cuopt_sh_client import CuOptServiceSelfHostClient
            self._client = CuOptServiceSelfHostClient(ip=host, port=port, polling_timeout=None)

    def _request_nim(self, problem_data: dict) -> dict:
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

    def _send(self, problem_data: dict) -> dict | None:
        """Send problem_data to cuOpt. Returns solver_response dict or None on failure."""
        try:
            if self._mode == "nim":
                response = self._request_nim(problem_data)
                solver_resp = response.get("response", {}).get("solver_response", response)
            else:
                response = self._client.get_optimized_routes(problem_data)
                solver_resp = response["response"]["solver_response"]
        except Exception as e:
            logger.warning("cuOpt request failed: %s", e, exc_info=True)
            return None

        if solver_resp.get("status", -1) != 0:
            logger.debug("cuOpt returned non-zero status: %s", solver_resp.get("status"))
            return None
        return solver_resp

    @staticmethod
    def _extract_routes(solver_resp: dict) -> list[list[int]]:
        """Extract customer routes (depot-stripped) from a cuOpt solver response."""
        routes: list[list[int]] = []
        for _vid, vdata in solver_resp.get("vehicle_data", {}).items():
            route = [node for node in vdata.get("route", []) if node != 0]
            if route:
                routes.append(route)
        return routes
