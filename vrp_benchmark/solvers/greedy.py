"""Nearest-neighbour greedy CVRP solver — fast O(n²) construction heuristic.

Serves as the baseline that the DQN is trained to beat.
At each step, visit the nearest unvisited feasible customer.
When no feasible customer remains, return to depot and start a new route.
"""
from __future__ import annotations

from vrp_benchmark.data import CVRPInstance, route_cost


class GreedySolver:
    """Nearest-neighbour greedy construction."""

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        unvisited = set(range(1, instance.n_customers + 1))
        routes: list[list[int]] = []
        current_route: list[int] = []
        current_node = 0
        remaining_cap = instance.capacity

        while unvisited:
            # Find nearest feasible unvisited customer
            best_node = None
            best_dist = float("inf")
            for node in unvisited:
                demand = instance.demands[node - 1]
                if demand <= remaining_cap:
                    d = instance.dist(current_node, node)
                    if d < best_dist:
                        best_dist = d
                        best_node = node

            if best_node is None:
                # No feasible customer — close route, start new one
                if current_route:
                    routes.append(current_route)
                current_route = []
                current_node = 0
                remaining_cap = instance.capacity
            else:
                current_route.append(best_node)
                remaining_cap -= instance.demands[best_node - 1]
                current_node = best_node
                unvisited.remove(best_node)

        if current_route:
            routes.append(current_route)

        cost = route_cost(instance, routes)
        return routes, cost
