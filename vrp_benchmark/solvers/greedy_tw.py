"""Nearest-neighbor greedy baseline for VRPTW.

Builds routes one at a time: at each step, pick the nearest unvisited
customer that is both capacity- and time-window-feasible (including the
ability to return to the depot before its due date). Vehicles can wait
at a customer if they arrive before its ready_time.
"""
from __future__ import annotations

import numpy as np

from vrp_benchmark.data_tw import VRPTWInstance


class GreedyVRPTWSolver:
    def solve(self, instance: VRPTWInstance) -> tuple[list[list[int]], float]:
        """Return (routes, total_distance). Routes use 1-based customer indices."""
        n = instance.n_customers
        unvisited = set(range(1, n + 1))
        routes: list[list[int]] = []
        total_dist = 0.0

        while unvisited:
            route: list[int] = []
            load = 0.0
            t = 0.0        # current time
            current = 0    # depot

            while True:
                best_node = None
                best_dist = float("inf")

                for node in unvisited:
                    # Capacity check
                    if load + instance.demands[node - 1] > instance.capacity:
                        continue
                    # Time window check
                    travel = instance.dist(current, node)
                    arrival = t + travel
                    if arrival > instance.due_dates[node]:
                        continue
                    # Can we return to depot after serving this node?
                    depart = max(arrival, instance.ready_times[node]) + instance.service_times[node]
                    if depart + instance.dist(node, 0) > instance.due_dates[0]:
                        continue
                    if travel < best_dist:
                        best_dist = travel
                        best_node = node

                if best_node is None:
                    break

                # Commit to best_node
                travel = instance.dist(current, best_node)
                total_dist += travel
                arrival = t + travel
                t = max(arrival, instance.ready_times[best_node]) + instance.service_times[best_node]
                load += instance.demands[best_node - 1]
                route.append(best_node)
                unvisited.remove(best_node)
                current = best_node

            # Return to depot
            if route:
                total_dist += instance.dist(current, 0)
                routes.append(route)
            elif unvisited:
                # No feasible insertion found — skip remaining customers to avoid infinite loop
                break

        return routes, total_dist
