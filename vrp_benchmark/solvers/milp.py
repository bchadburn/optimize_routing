"""Exact MILP solver for small CVRP instances using OR-Tools CP-SAT.

Only practical for n ≤ ~10 customers with 2-minute time limit. Returns a
provably optimal solution (status OPTIMAL) or signals failure — it does NOT
return a best-found feasible solution on timeout, because that would be
falsely labelled "exact".

Use MILPSolver only as a ground-truth oracle. Any result it returns is optimal.

Limitation: the arc-based vehicle flow formulation has a weak LP relaxation.
Branch-and-bound struggles past n=10 even with 2 minutes. A set-partitioning
(column generation) formulation would scale further but is much harder to
implement. At n=20, OR-Tools/cuOpt reaching the same solution is the
practical ground truth.

Formulation: standard vehicle flow model.
  - Binary arc variables x[v,i,j]: vehicle v travels arc (i→j)
  - Capacity constraints via cumulative load
  - Each customer visited exactly once
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from vrp_benchmark.data import CVRPInstance, route_cost

MAX_CUSTOMERS_EXACT = 20  # beyond this, MILP solve times become impractical


class MILPSolver:
    """Exact CVRP solver via CP-SAT. Practical for n ≤ 20."""

    def __init__(self, time_limit_s: int = 120) -> None:
        self._time_limit_s = time_limit_s

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        if instance.n_customers > MAX_CUSTOMERS_EXACT:
            return [], 1e9

        n = instance.n_customers
        V = instance.n_vehicles
        SCALE = 10_000  # scale floats to integers for CP-SAT

        dist_int = [
            [round(instance.dist(i, j) * SCALE) for j in range(n + 1)]
            for i in range(n + 1)
        ]

        model = cp_model.CpModel()

        # x[v][i][j] = 1 if vehicle v travels arc i→j
        x = [
            [[model.new_bool_var(f"x_{v}_{i}_{j}") for j in range(n + 1)] for i in range(n + 1)]
            for v in range(V)
        ]

        # No self-loops
        for v in range(V):
            for i in range(n + 1):
                model.add(x[v][i][i] == 0)

        # Each customer visited exactly once across all vehicles
        for j in range(1, n + 1):
            model.add(sum(x[v][i][j] for v in range(V) for i in range(n + 1) if i != j) == 1)

        # Flow conservation: if vehicle enters a node it must leave
        for v in range(V):
            for k in range(1, n + 1):
                model.add(
                    sum(x[v][i][k] for i in range(n + 1) if i != k)
                    == sum(x[v][k][j] for j in range(n + 1) if j != k)
                )

        # Each vehicle leaves depot at most once
        for v in range(V):
            model.add(sum(x[v][0][j] for j in range(1, n + 1)) <= 1)

        # Capacity via MTZ-style load variables
        load = [
            [model.new_int_var(0, instance.capacity, f"load_{v}_{i}") for i in range(n + 1)]
            for v in range(V)
        ]
        for v in range(V):
            model.add(load[v][0] == 0)
            for j in range(1, n + 1):
                for i in range(n + 1):
                    if i == j:
                        continue
                    b = x[v][i][j]
                    # If arc i→j used: load[v][j] = load[v][i] + demand[j]
                    model.add(
                        load[v][j] >= load[v][i] + instance.demands[j - 1] - instance.capacity * (1 - b)
                    )
                    model.add(load[v][j] <= instance.capacity)

        # Objective: minimise total distance
        obj_terms = [
            dist_int[i][j] * x[v][i][j]
            for v in range(V)
            for i in range(n + 1)
            for j in range(n + 1)
            if i != j
        ]
        model.minimize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._time_limit_s

        status = solver.solve(model)

        # Only accept PROVEN optimal — a FEASIBLE result at timeout is not ground truth.
        if status != cp_model.OPTIMAL:
            return [], 1e9

        # Extract routes
        routes: list[list[int]] = []
        for v in range(V):
            if solver.value(x[v][0][0]) == 1:
                continue
            # Check if this vehicle departs the depot
            start = next(
                (j for j in range(1, n + 1) if solver.value(x[v][0][j]) == 1), None
            )
            if start is None:
                continue
            route = []
            current = start
            while current != 0:
                route.append(current)
                current = next(
                    (j for j in range(n + 1) if j != current and solver.value(x[v][current][j]) == 1),
                    0,
                )
            routes.append(route)

        cost = route_cost(instance, routes)
        return routes, cost
