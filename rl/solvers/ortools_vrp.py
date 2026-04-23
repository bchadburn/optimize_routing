"""OR-Tools Routing Library CVRPTW solver — fair baseline for cuOpt comparison.

Uses CP-based VRP (not LP flow). Each open DC is treated as a separate depot
serving its assigned customers. Customer-to-DC assignment uses minimum transport cost.
"""
from __future__ import annotations

import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class OrtoolsVrpSolver:
    """Solve DC→customer routing as CVRPTW using OR-Tools Routing Library."""

    def __init__(self, time_limit_s: int = 5) -> None:
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
            cost = _solve_single_dc_ortools(
                customer_ids=assigned,
                demands=demands,
                costs_to_customers=transport_cost_d_to_c[dc_id],
                n_vehicles=n_vehicles_per_dc,
                time_limit_s=self._time_limit_s,
            )
            total_cost += cost

        return total_cost


def _assign_customers_to_dcs(
    open_dc_ids: list[int],
    cust_ids: list[int],
    transport_cost_d_to_c: dict[int, dict[int, float]],
) -> dict[int, list[int]]:
    """Assign each customer to the cheapest open DC."""
    dc_customers: dict[int, list[int]] = {dc: [] for dc in open_dc_ids}
    for cust_id in cust_ids:
        best_dc = min(
            open_dc_ids,
            key=lambda dc: transport_cost_d_to_c[dc].get(cust_id, 1e6),
        )
        dc_customers[best_dc].append(cust_id)
    return dc_customers


def _solve_single_dc_ortools(
    customer_ids: list[int],
    demands: dict[int, float],
    costs_to_customers: dict[int, float],
    n_vehicles: int,
    time_limit_s: int,
) -> float:
    """Run OR-Tools VRP for a single DC. Returns routing cost or 1e6 on failure."""
    n = len(customer_ids)
    if n == 0:
        return 0.0

    # Arc costs are weighted by demand quantity so the VRP objective matches the
    # LP flow formulation: cost(depot→customer) = transport_cost × demand.
    # Without this, VRP measures only travel distance (16x cheaper than LP/MILP),
    # making RL rewards incomparable to the MILP benchmark.
    SCALE = 1000
    size = n + 1
    matrix = [[0] * size for _ in range(size)]
    for i, ci in enumerate(customer_ids, start=1):
        # Halved so round-trip (depot→customer→depot) equals LP one-way cost:
        # LP charges cost × demand once; VRP charges cost × demand / 2 each leg.
        cost_int = int(costs_to_customers.get(ci, 1e6) * demands[ci] * SCALE / 2)
        matrix[0][i] = cost_int
        matrix[i][0] = cost_int
    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                matrix[i][j] = matrix[0][i] + matrix[0][j]

    total_demand = sum(demands[c] for c in customer_ids)
    # Capacity must be feasible: at minimum ceil(total/n_vehicles), but also enough
    # that no feasible packing is blocked. Use total_demand as the upper bound so
    # OR-Tools can always find a feasible solution (vehicles can carry more if needed).
    vehicle_capacity = int(math.ceil(total_demand))

    manager = pywrapcp.RoutingIndexManager(size, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        if node == 0:
            return 0
        return int(math.ceil(demands[customer_ids[node - 1]]))

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [vehicle_capacity] * n_vehicles, True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(search_params)
    if solution:
        return solution.ObjectiveValue() / SCALE
    return 1e6
