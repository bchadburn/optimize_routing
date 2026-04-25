"""VRPTW instance type and feasibility-aware cost function.

Wraps CVRPInstance with per-node time windows and service times, using the
same 0=depot, 1..n=customer index convention throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from vrp_benchmark.data import CVRPInstance


@dataclass
class VRPTWInstance:
    """A CVRP instance extended with time windows and service times.

    All arrays are length n_customers+1; index 0 is the depot.
    Index convention matches CVRPInstance.dist(): 0=depot, 1..n=customers.

    Attributes:
        cvrp: Underlying CVRPInstance (distance matrix, demands, capacity).
        ready_times: Earliest time service can begin at each node.
        due_dates: Latest arrival time (hard constraint) at each node.
        service_times: Time spent at each node before departure.
    """

    cvrp: CVRPInstance
    ready_times: np.ndarray    # shape (n_customers+1,)
    due_dates: np.ndarray      # shape (n_customers+1,)
    service_times: np.ndarray  # shape (n_customers+1,)

    def __post_init__(self) -> None:
        n = self.cvrp.n_customers
        for name, arr in [
            ("ready_times", self.ready_times),
            ("due_dates", self.due_dates),
            ("service_times", self.service_times),
        ]:
            if arr.shape != (n + 1,):
                raise ValueError(f"{name} must have shape ({n + 1},), got {arr.shape}")

    # --- convenience pass-throughs ---

    @property
    def n_customers(self) -> int:
        return self.cvrp.n_customers

    @property
    def depot(self) -> np.ndarray:
        return self.cvrp.depot

    @property
    def coords(self) -> np.ndarray:
        return self.cvrp.coords

    @property
    def demands(self) -> np.ndarray:
        return self.cvrp.demands

    @property
    def capacity(self) -> float:
        return self.cvrp.capacity

    @property
    def n_vehicles(self) -> int:
        return self.cvrp.n_vehicles

    def dist(self, i: int, j: int) -> float:
        return self.cvrp.dist(i, j)

    @property
    def dist_matrix(self) -> np.ndarray:
        return self.cvrp.dist_matrix


class RouteCostTW(NamedTuple):
    distance: float
    feasible: bool
    max_lateness: float  # 0 if feasible; max over all violations otherwise


def route_cost_tw(instance: VRPTWInstance, routes: list[list[int]]) -> RouteCostTW:
    """Compute total distance and check time-window feasibility.

    Walks each route tracking cumulative time. Vehicles wait if they arrive
    before ready_time; a violation occurs if arrival > due_date.

    Returns:
        RouteCostTW with distance, feasible flag, and max_lateness.
    """
    total_dist = 0.0
    max_lateness = 0.0

    for route in routes:
        if not route:
            continue
        t = 0.0
        prev = 0  # depot
        for node in route:
            travel = instance.dist(prev, node)
            total_dist += travel
            arrival = t + travel
            lateness = max(0.0, arrival - instance.due_dates[node])
            max_lateness = max(max_lateness, lateness)
            # wait if early, then serve
            t = max(arrival, instance.ready_times[node]) + instance.service_times[node]
            prev = node
        # return to depot
        travel = instance.dist(prev, 0)
        total_dist += travel
        arrival = t + travel
        lateness = max(0.0, arrival - instance.due_dates[0])
        max_lateness = max(max_lateness, lateness)

    return RouteCostTW(
        distance=total_dist,
        feasible=max_lateness == 0.0,
        max_lateness=max_lateness,
    )
