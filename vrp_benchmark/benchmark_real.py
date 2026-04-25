"""Benchmark solvers against Uchoa et al. real CVRP instances with known optimal costs.

Unlike the synthetic benchmark, this measures gap vs published best-known solution (BKS)
for each instance — giving an absolute quality metric instead of a relative one.

Usage:
    uv run python -m vrp_benchmark.benchmark_real
    uv run python -m vrp_benchmark.benchmark_real --cuopt
    uv run python -m vrp_benchmark.benchmark_real --instances X-n101-k25 X-n200-k36
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from vrp_benchmark.datasets.uchoa import DEFAULT_INSTANCES, BenchmarkInstance, load
from vrp_benchmark.solvers.greedy import GreedySolver
from vrp_benchmark.solvers.ortools_vrp import ORToolsSolver

RESULTS_DIR = Path("results")


def _solve(solver, inst: BenchmarkInstance) -> tuple[float, float]:
    """Return (cost, elapsed_s)."""
    t0 = time.perf_counter()
    _, cost = solver.solve(inst.instance)
    return cost, time.perf_counter() - t0


def run(
    instance_names: list[str],
    include_cuopt: bool = False,
    nim_mode: bool = False,
    ortools_time_s: int = 30,
    cuopt_time_s: int = 30,
) -> None:
    solvers: dict[str, object] = {
        "greedy": GreedySolver(),
        "ortools": ORToolsSolver(time_limit_s=ortools_time_s),
    }

    if include_cuopt:
        try:
            from vrp_benchmark.solvers.cuopt_vrp import CuOptSolver
            solvers["cuopt"] = CuOptSolver(time_limit_s=cuopt_time_s, mode="nim" if nim_mode else "self-hosted")
        except Exception as e:
            print(f"WARNING: cuOpt unavailable: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for name in instance_names:
        print(f"\nLoading {name} ...")
        bench = load(name)
        n = bench.instance.n_customers
        print(f"  n={n} customers  capacity={bench.instance.capacity}  BKS={bench.bks:,}")

        for solver_name, solver in solvers.items():
            cost, elapsed = _solve(solver, bench)
            gap = (cost - bench.bks) / bench.bks * 100 if cost < 1e8 else float("nan")
            success = cost < 1e8

            cost_str = f"{cost:,.0f}" if success else "FAIL"
            gap_str = f"{gap:+.1f}%" if success else "  -"
            print(f"  {solver_name:<10} cost={cost_str:>10}  gap_vs_bks={gap_str:>7}  time={elapsed*1000:.0f}ms")

            rows.append({
                "instance": name,
                "n_customers": n,
                "bks": bench.bks,
                "solver": solver_name,
                "cost": round(cost, 1) if success else None,
                "gap_vs_bks_pct": round(gap, 2) if success else None,
                "time_s": round(elapsed, 3),
                "success": success,
            })

    out_path = RESULTS_DIR / "real_benchmark.csv"
    fieldnames = ["instance", "n_customers", "bks", "solver", "cost", "gap_vs_bks_pct", "time_s", "success"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=DEFAULT_INSTANCES)
    parser.add_argument("--cuopt", action="store_true")
    parser.add_argument("--ortools-time", type=int, default=30)
    parser.add_argument("--nim", action="store_true", help="Use NVIDIA NIM cloud API instead of self-hosted (set NVIDIA_API_KEY)")
    parser.add_argument("--cuopt-time", type=int, default=30)
    args = parser.parse_args()
    run(args.instances, include_cuopt=args.cuopt, nim_mode=args.nim,
        ortools_time_s=args.ortools_time, cuopt_time_s=args.cuopt_time)
