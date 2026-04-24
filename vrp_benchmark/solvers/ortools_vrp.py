"""OR-Tools Routing Library CVRP solver.

Uses CP-based metaheuristic (guided local search). Good quality up to ~500 customers.
"""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from vrp_benchmark.data import CVRPInstance, route_cost


class ORToolsSolver:
    """CVRP solver using OR-Tools Routing Library with guided local search."""

    def __init__(self, time_limit_s: int = 120) -> None:
        self._time_limit_s = time_limit_s

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        n = instance.n_customers
        SCALE = 10_000

        dist_int = [
            [round(instance.dist(i, j) * SCALE) for j in range(n + 1)]
            for i in range(n + 1)
        ]

        manager = pywrapcp.RoutingIndexManager(n + 1, instance.n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_idx: int, to_idx: int) -> int:
            return dist_int[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        transit_cb = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        def demand_callback(from_idx: int) -> int:
            node = manager.IndexToNode(from_idx)
            return 0 if node == 0 else int(instance.demands[node - 1])

        demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb, 0, [instance.capacity] * instance.n_vehicles, True, "Capacity"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self._time_limit_s

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return [], 1e9

        routes: list[list[int]] = []
        for v in range(instance.n_vehicles):
            index = routing.Start(v)
            route: list[int] = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(node)
                index = solution.Value(routing.NextVar(index))
            if route:
                routes.append(route)

        cost = route_cost(instance, routes)
        return routes, cost
