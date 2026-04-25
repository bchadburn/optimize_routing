"""Benchmark solvers against Solomon VRPTW instances with known optimal solutions.

Reports gap vs published best-known solution (BKS) for distance and vehicle count.
Distance is the primary objective; vehicle count is secondary (informational).

Usage:
    uv run python -m vrp_benchmark.benchmark_solomon
    uv run python -m vrp_benchmark.benchmark_solomon --cuopt
    uv run python -m vrp_benchmark.benchmark_solomon --family C1 R1
    uv run python -m vrp_benchmark.benchmark_solomon --instances C101 R101 RC101
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from vrp_benchmark._bench_util import add_cuopt_args, init_cuopt_solver, write_csv
from vrp_benchmark.data_tw import VRPTWInstance, route_cost_tw
from vrp_benchmark.datasets.solomon import (
    DEFAULT_INSTANCES,
    FAMILIES,
    SolomonBenchmarkInstance,
    load,
)
from vrp_benchmark.solvers.cuopt_tw import CuOptVRPTWSolver
from vrp_benchmark.solvers.greedy_tw import GreedyVRPTWSolver
from vrp_benchmark.solvers.ortools_tw import ORToolsVRPTWSolver
from vrp_benchmark.solvers.protocol import VRPTWSolver

RESULTS_DIR = Path("results")


def _solve(solver: VRPTWSolver, inst: VRPTWInstance) -> tuple[float, int, bool, float]:
    """Return (distance, n_vehicles_used, feasible, elapsed_s)."""
    t0 = time.perf_counter()
    routes, cost = solver.solve(inst)
    elapsed = time.perf_counter() - t0

    if cost >= 1e8:
        return cost, 0, False, elapsed

    result = route_cost_tw(inst, routes)
    return result.distance, len(routes), result.feasible, elapsed


def run(
    instance_names: list[str],
    include_cuopt: bool = False,
    nim_mode: bool = False,
    ortools_time_s: int = 30,
    cuopt_time_s: int = 30,
) -> None:
    solvers: dict[str, VRPTWSolver] = {
        "greedy": GreedyVRPTWSolver(),
        "ortools": ORToolsVRPTWSolver(time_limit_s=ortools_time_s),
    }
    cuopt = init_cuopt_solver(include_cuopt, nim_mode, cuopt_time_s, CuOptVRPTWSolver)
    if cuopt:
        solvers["cuopt"] = cuopt

    rows: list[dict] = []

    for name in instance_names:
        print(f"\nLoading {name} ...")
        bench: SolomonBenchmarkInstance = load(name)
        inst = bench.instance
        family = name[:2] if name[1].isdigit() else name[:3]
        print(
            f"  n={inst.n_customers}  capacity={inst.capacity:.0f}  "
            f"vehicles={inst.n_vehicles}  "
            f"BKS={bench.bks_distance:.2f} ({bench.bks_vehicles}v)"
        )

        for solver_name, solver in solvers.items():
            dist, n_veh, feasible, elapsed = _solve(solver, inst)

            if feasible:
                gap = (dist - bench.bks_distance) / bench.bks_distance * 100
                dist_str = f"{dist:,.2f}"
                gap_str = f"{gap:+.1f}%"
            else:
                gap = float("nan")
                dist_str = "INFEASIBLE" if dist < 1e8 else "FAIL"
                gap_str = "  -"

            print(
                f"  {solver_name:<10} dist={dist_str:>10}  "
                f"gap_vs_bks={gap_str:>7}  vehicles={n_veh:>3}  "
                f"feasible={feasible}  time={elapsed*1000:.0f}ms"
            )

            rows.append({
                "instance": name,
                "family": family,
                "n_customers": inst.n_customers,
                "bks_distance": bench.bks_distance,
                "bks_vehicles": bench.bks_vehicles,
                "solver": solver_name,
                "distance": round(dist, 2) if feasible else None,
                "n_vehicles_used": n_veh if feasible else None,
                "gap_vs_bks_pct": round(gap, 2) if feasible else None,
                "time_s": round(elapsed, 3),
                "feasible": feasible,
            })

    out_path = RESULTS_DIR / "solomon_benchmark.csv"
    fieldnames = [
        "instance", "family", "n_customers", "bks_distance", "bks_vehicles",
        "solver", "distance", "n_vehicles_used", "gap_vs_bks_pct", "time_s", "feasible",
    ]
    write_csv(out_path, fieldnames, rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=None)
    parser.add_argument("--family", nargs="+", default=None,
                        help="Run all instances in a family (e.g. C1 R1 RC2)")
    parser.add_argument("--ortools-time", type=int, default=30)
    add_cuopt_args(parser)
    args = parser.parse_args()

    if args.instances:
        names = args.instances
    elif args.family:
        names = []
        for fam in args.family:
            if fam not in FAMILIES:
                raise SystemExit(f"Unknown family {fam!r}. Available: {sorted(FAMILIES)}")
            names.extend(FAMILIES[fam])
    else:
        names = DEFAULT_INSTANCES

    run(names, include_cuopt=args.cuopt, nim_mode=args.nim,
        ortools_time_s=args.ortools_time, cuopt_time_s=args.cuopt_time)
