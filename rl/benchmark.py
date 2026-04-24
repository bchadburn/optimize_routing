"""Scalability benchmark: OR-Tools VRP vs cuOpt CVRPTW (cuOpt pending GPU setup).

Sweeps customer counts, times each solver on synthetic single-DC instances,
writes results/cuopt_benchmark.csv.

Usage:
    uv run python -m rl.benchmark
    uv run python -m rl.benchmark --counts 12 50 100 250 500
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from rl.solvers.ortools_vrp import OrtoolsVrpSolver

RESULTS_DIR = Path("results")


def _make_instance(
    n_customers: int, seed: int = 0
) -> tuple[list[int], dict[int, float], dict[int, dict[int, float]], int]:
    """Return (open_dc_ids, demands, transport_costs, n_vehicles) for one DC."""
    rng = np.random.default_rng(seed)
    demands = {i: float(rng.integers(10, 50)) for i in range(n_customers)}
    transport_costs = {0: {i: float(rng.uniform(1.0, 10.0)) for i in range(n_customers)}}
    n_vehicles = max(1, math.ceil(n_customers / 8))
    return [0], demands, transport_costs, n_vehicles


class ScalabilityBenchmark:
    def __init__(
        self,
        results_dir: Path = RESULTS_DIR,
        n_trials: int = 3,
        include_cuopt: bool = False,
    ) -> None:
        self._results_dir = results_dir
        self._n_trials = n_trials
        self._include_cuopt = include_cuopt

        self._ortools = OrtoolsVrpSolver(time_limit_s=10)
        self._cuopt = None
        if include_cuopt:
            try:
                from rl.solvers.cuopt_vrp import CuOptVrpSolver
                self._cuopt = CuOptVrpSolver()
            except ImportError:
                print("WARNING: cuOpt not available — running OR-Tools only.")

    def run(self, customer_counts: list[int] | None = None) -> None:
        """Run benchmark and write results/cuopt_benchmark.csv."""
        if customer_counts is None:
            customer_counts = [12, 50, 100, 250, 500]

        self._results_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for n in customer_counts:
            open_dc_ids, demands, transport_costs, n_vehicles = _make_instance(n, seed=n)

            ortools_times, ortools_cost = self._time_solver(
                self._ortools, open_dc_ids, demands, transport_costs, n_vehicles
            )
            rows.append({
                "n_customers": n,
                "n_vehicles": n_vehicles,
                "solver": "ortools_vrp",
                "solve_time_s": round(sum(ortools_times) / len(ortools_times), 6),
                "total_cost": round(ortools_cost, 4),
            })
            print(
                f"OR-Tools  n={n:4d}  v={n_vehicles:3d}  "
                f"cost={ortools_cost:8.1f}  time={ortools_times[-1]*1000:7.1f}ms"
            )

            if self._cuopt is not None:
                cuopt_times, cuopt_cost = self._time_solver(
                    self._cuopt, open_dc_ids, demands, transport_costs, n_vehicles
                )
                speedup = ortools_times[-1] / cuopt_times[-1] if cuopt_times[-1] > 0 else 0
                rows.append({
                    "n_customers": n,
                    "n_vehicles": n_vehicles,
                    "solver": "cuopt_cvrptw",
                    "solve_time_s": round(sum(cuopt_times) / len(cuopt_times), 6),
                    "total_cost": round(cuopt_cost, 4),
                })
                print(
                    f"cuOpt     n={n:4d}  v={n_vehicles:3d}  "
                    f"cost={cuopt_cost:8.1f}  time={cuopt_times[-1]*1000:7.1f}ms  "
                    f"({speedup:.1f}x speedup)"
                )

        out_path = self._results_dir / "cuopt_benchmark.csv"
        fieldnames = ["n_customers", "n_vehicles", "solver", "solve_time_s", "total_cost"]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")

    def _time_solver(self, solver, open_dc_ids, demands, transport_costs, n_vehicles):
        times = []
        cost = 1e6
        for _ in range(self._n_trials):
            t0 = time.perf_counter()
            cost = solver.solve(open_dc_ids, demands, transport_costs, n_vehicles_per_dc=n_vehicles)
            times.append(time.perf_counter() - t0)
        return times, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[12, 50, 100, 250, 500])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cuopt", action="store_true", help="Include cuOpt (requires GPU)")
    args = parser.parse_args()
    ScalabilityBenchmark(n_trials=args.trials, include_cuopt=args.cuopt).run(args.counts)
