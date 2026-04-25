"""OR-Tools Routing Library VRPTW solver.

Extends the CVRP solver with time window constraints via AddDimension.
The time callback includes service time at the origin node — omitting this
is the most common VRPTW implementation bug, causing the solver to depart
before service is complete.

Key OR-Tools VRPTW details:
  - AddDimension(cb, slack_max, horizon, fix_start_to_zero, name)
    slack_max = horizon allows vehicles to wait freely at early arrivals.
    fix_start_to_zero = False so vehicles can depart after depot ready_time.
  - Arc cost uses the distance callback; time dimension uses a separate
    transit callback that adds service_time at the from-node.
  - SetRange on each node's CumulVar enforces hard time windows.
"""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vrp_benchmark.data_tw import VRPTWInstance, route_cost_tw
from vrp_benchmark.solvers._ortools_util import extract_routes

TSCALE = 10  # Solomon times are small integers; ×10 gives enough precision


class ORToolsVRPTWSolver:
    """VRPTW solver using OR-Tools Routing Library with guided local search."""

    def __init__(self, time_limit_s: int = 30) -> None:
        self._time_limit_s = time_limit_s

    def solve(self, instance: VRPTWInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        DSCALE = 10_000  # distance scaling (separate from time scaling)

        dist_int = [
            [round(instance.dist(i, j) * DSCALE) for j in range(n + 1)]
            for i in range(n + 1)
        ]
        # Time matrix: travel time = distance (speed=1 in Solomon)
        time_int = [
            [round(instance.dist(i, j) * TSCALE) for j in range(n + 1)]
            for i in range(n + 1)
        ]
        service_int = [round(instance.service_times[i] * TSCALE) for i in range(n + 1)]

        manager = pywrapcp.RoutingIndexManager(n + 1, instance.n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback — used for arc cost (objective)
        def distance_callback(from_idx: int, to_idx: int) -> int:
            return dist_int[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        dist_cb = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(dist_cb)

        # Capacity dimension
        def demand_callback(from_idx: int) -> int:
            node = manager.IndexToNode(from_idx)
            return 0 if node == 0 else int(instance.demands[node - 1])

        demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb, 0, [int(instance.capacity)] * instance.n_vehicles, True, "Capacity"
        )

        # Time callback — travel time + service time at origin node
        # Service time must be included here; if omitted the solver assumes
        # instantaneous service and can violate time windows silently.
        def time_callback(from_idx: int, to_idx: int) -> int:
            i = manager.IndexToNode(from_idx)
            j = manager.IndexToNode(to_idx)
            return time_int[i][j] + service_int[i]

        time_cb = routing.RegisterTransitCallback(time_callback)
        horizon = int(instance.due_dates.max() * TSCALE) + 1
        # slack_max=horizon: vehicles may wait freely when arriving early.
        # fix_start_cumul_to_zero=False: vehicles depart when depot TW opens.
        routing.AddDimension(time_cb, horizon, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        # Enforce per-node time windows
        for node in range(1, n + 1):
            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(
                round(instance.ready_times[node] * TSCALE),
                round(instance.due_dates[node] * TSCALE),
            )

        # Depot time window applied to each vehicle's start
        for v in range(instance.n_vehicles):
            start_idx = routing.Start(v)
            time_dim.CumulVar(start_idx).SetRange(
                round(instance.ready_times[0] * TSCALE),
                round(instance.due_dates[0] * TSCALE),
            )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        # PARALLEL_CHEAPEST_INSERTION handles time windows far better than
        # PATH_CHEAPEST_ARC; the arc-based strategy often fails to find any
        # feasible initial solution on tight-window instances (R1/RC1 families).
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self._time_limit_s

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return [], 1e9

        routes = extract_routes(routing, manager, solution, instance.n_vehicles)
        result = route_cost_tw(instance, routes)
        return routes, result.distance if result.feasible else 1e9
