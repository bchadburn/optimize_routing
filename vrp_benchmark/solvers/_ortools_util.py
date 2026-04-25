"""Shared OR-Tools routing utilities."""
from __future__ import annotations

from ortools.constraint_solver import pywrapcp


def extract_routes(
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    solution: pywrapcp.Assignment,
    n_vehicles: int,
) -> list[list[int]]:
    """Extract customer routes (1-based indices, depot stripped) from an OR-Tools solution."""
    routes: list[list[int]] = []
    for v in range(n_vehicles):
        index = routing.Start(v)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    return routes
