"""Exact/near-exact CVRP solver using OR-Tools CP-SAT with circuit constraints.

Uses CP-SAT's built-in AddCircuit constraint for each vehicle, which handles
subtour elimination far more efficiently than the classic arc-flow MTZ formulation.
The MTZ approach adds O(n²V) big-M constraints with a weak LP relaxation; the
circuit constraint lets CP-SAT's propagation engine do the heavy lifting directly.

Result quality:
  - status OPTIMAL  → proven global optimum
  - status FEASIBLE → best incumbent found within time limit; gap is reported
  - status INFEASIBLE / no solution → returns ([], 1e9, None)

Practical limits with a 300s time limit:
  - n ≤ 15  : usually OPTIMAL
  - n = 20  : often OPTIMAL or tight FEASIBLE (gap < 5%)
  - n = 30  : FEASIBLE with 5–15% gap typical
  - n = 50  : FEASIBLE but gap can be large; cuOpt/OR-Tools are better references
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ortools.sat.python import cp_model

from vrp_benchmark.data import CVRPInstance, route_cost

logger = logging.getLogger(__name__)

MAX_CUSTOMERS_MILP = 50  # skip entirely beyond this — too slow to be informative


@dataclass
class MILPResult:
    routes: list[list[int]]
    cost: float
    is_optimal: bool
    gap_pct: float | None  # (incumbent - lower_bound) / lower_bound * 100; None if no LB


class MILPSolver:
    """CVRP solver via CP-SAT circuit constraint.

    Returns the best solution found within the time limit.
    Use is_optimal / gap_pct from solve_detailed() to interpret quality.
    solve() returns (routes, cost) for protocol compatibility; cost is 1e9 on failure.
    """

    def __init__(self, time_limit_s: int = 300) -> None:
        self._time_limit_s = time_limit_s

    def solve(self, instance: CVRPInstance) -> tuple[list[list[int]], float]:
        result = self.solve_detailed(instance)
        return result.routes, result.cost

    def solve_detailed(self, instance: CVRPInstance) -> MILPResult:
        if instance.n_customers > MAX_CUSTOMERS_MILP:
            return MILPResult([], 1e9, False, None)

        n = instance.n_customers
        V = instance.n_vehicles
        SCALE = 10_000  # scale floats → integers for CP-SAT

        dist_int = [
            [round(instance.dist(i, j) * SCALE) for j in range(n + 1)]
            for i in range(n + 1)
        ]

        model = cp_model.CpModel()

        # Arc variables: x[v][i][j] = 1 if vehicle v uses arc i→j
        # Node 0 is depot. We add a dummy return arc (0→0) for unused vehicles.
        x = {}
        for v in range(V):
            for i in range(n + 1):
                for j in range(n + 1):
                    x[v, i, j] = model.new_bool_var(f"x_{v}_{i}_{j}")

        # --- Circuit constraint per vehicle ---
        # Each vehicle's arc set must form a Hamiltonian circuit over the nodes it visits
        # (including the depot). CP-SAT's AddCircuit efficiently eliminates subtours.
        # Nodes not visited by vehicle v are handled via self-loops (i→i arc = 1).
        for v in range(V):
            arcs = []
            for i in range(n + 1):
                for j in range(n + 1):
                    if i != j:
                        arcs.append((i, j, x[v, i, j]))
                    else:
                        # Self-loop: node i skipped by this vehicle
                        arcs.append((i, i, model.new_bool_var(f"skip_{v}_{i}")))
            model.add_circuit(arcs)

        # Each customer visited exactly once across all vehicles
        for j in range(1, n + 1):
            model.add(
                sum(x[v, i, j] for v in range(V) for i in range(n + 1) if i != j) == 1
            )

        # Capacity: cumulative load per vehicle via auxiliary variables
        load = {}
        for v in range(V):
            for i in range(n + 1):
                load[v, i] = model.new_int_var(0, instance.capacity, f"load_{v}_{i}")

        for v in range(V):
            model.add(load[v, 0] == 0)
            for j in range(1, n + 1):
                for i in range(n + 1):
                    if i == j:
                        continue
                    # If arc i→j used: load[v][j] ≥ load[v][i] + demand[j]
                    demand_j = int(instance.demands[j - 1])
                    model.add(
                        load[v, j] >= load[v, i] + demand_j - instance.capacity * (1 - x[v, i, j])
                    )

        # Objective: minimise total arc cost
        obj = sum(
            dist_int[i][j] * x[v, i, j]
            for v in range(V)
            for i in range(n + 1)
            for j in range(n + 1)
            if i != j
        )
        model.minimize(obj)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._time_limit_s

        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.debug("CP-SAT: no solution for n=%d", n)
            return MILPResult([], 1e9, False, None)

        is_optimal = status == cp_model.OPTIMAL

        # Compute optimality gap: (incumbent - lower_bound) / lower_bound * 100
        obj_val = solver.objective_value
        lb = solver.best_objective_bound
        gap_pct: float | None = None
        if lb > 0:
            gap_pct = (obj_val - lb) / lb * 100

        # Extract routes
        routes: list[list[int]] = []
        for v in range(V):
            start = next(
                (j for j in range(1, n + 1) if solver.value(x[v, 0, j]) == 1), None
            )
            if start is None:
                continue
            route = []
            current = start
            while current != 0:
                route.append(current)
                nxt = next(
                    (j for j in range(n + 1) if j != current and solver.value(x[v, current, j]) == 1),
                    0,
                )
                current = nxt
            routes.append(route)

        cost = route_cost(instance, routes)
        return MILPResult(routes, cost, is_optimal, gap_pct)
