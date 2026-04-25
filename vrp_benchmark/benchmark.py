"""VRP benchmark: compare MILP, OR-Tools, cuOpt, Greedy, and DQN.

For each customer count, solves N_EVAL instances with every available solver,
recording solve time and solution cost. Outputs results/vrp_benchmark.csv.

Usage:
    uv run python -m vrp_benchmark.benchmark                        # OR-Tools + Greedy + DQN
    uv run python -m vrp_benchmark.benchmark --cuopt               # include cuOpt
    uv run python -m vrp_benchmark.benchmark --milp                # include MILP (slow, small only)
    uv run python -m vrp_benchmark.benchmark --counts 10 20 50 100 500 1000
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from vrp_benchmark.data import CVRPInstance, generate_instance
from vrp_benchmark.solvers.greedy import GreedySolver
from vrp_benchmark.solvers.ortools_vrp import ORToolsSolver

RESULTS_DIR = Path("results")
MODELS_DIR = Path("vrp_benchmark/models")
N_EVAL = 10  # instances per (solver, n_customers) cell
CAPACITY = 50
SEED_OFFSET = 99_000  # eval seeds separate from training seeds


def _load_dqn_solver(n: int) -> object | None:
    model_path = MODELS_DIR / f"dqn_n{n}.pt"
    if not model_path.exists():
        return None
    try:
        from vrp_benchmark.solvers.dqn import DQNAgent, DQNSolver
        agent = DQNAgent.load(str(model_path))
        return DQNSolver(agent)
    except Exception as e:
        print(f"  WARNING: could not load DQN model for n={n}: {e}")
        return None


def _time_solver(
    solver,
    instances: list[CVRPInstance],
) -> tuple[float, float, float]:
    """Return (mean_time_s, mean_cost, success_rate)."""
    times, costs = [], []
    for inst in instances:
        t0 = time.perf_counter()
        _, cost = solver.solve(inst)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        costs.append(cost)
    success = sum(1 for c in costs if c < 1e8) / len(costs)
    valid_costs = [c for c in costs if c < 1e8]
    mean_cost = float(np.mean(valid_costs)) if valid_costs else 1e9
    return float(np.mean(times)), mean_cost, success


def _time_milp(
    solver,
    instances: list[CVRPInstance],
) -> tuple[float, float, float, float, float]:
    """Return (mean_time_s, mean_cost, success_rate, optimal_rate, mean_gap_pct).

    Reports both the feasible incumbent quality and whether it's proven optimal.
    gap_pct is the mean optimality gap: (incumbent - lower_bound) / lower_bound * 100.
    """
    times, costs, gaps, n_optimal = [], [], [], 0
    for inst in instances:
        t0 = time.perf_counter()
        result = solver.solve_detailed(inst)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        costs.append(result.cost)
        if result.is_optimal:
            n_optimal += 1
        if result.gap_pct is not None:
            gaps.append(result.gap_pct)
    success = sum(1 for c in costs if c < 1e8) / len(costs)
    valid_costs = [c for c in costs if c < 1e8]
    mean_cost = float(np.mean(valid_costs)) if valid_costs else 1e9
    mean_gap = float(np.mean(gaps)) if gaps else float("nan")
    return float(np.mean(times)), mean_cost, success, n_optimal / len(instances), mean_gap


def run(
    customer_counts: list[int],
    include_milp: bool = False,
    include_cuopt: bool = False,
    nim_mode: bool = False,
    n_eval: int = N_EVAL,
    ortools_time_s: int = 30,
    milp_time_s: int = 300,
    milp_budget_s: int = 3600,
    cuopt_time_s: int = 10,
) -> None:
    solvers: dict[str, object] = {
        "greedy": GreedySolver(),
        "ortools": ORToolsSolver(time_limit_s=ortools_time_s),
    }

    milp_solver = None
    milp_cumulative_s: float = 0.0  # tracks total wall time spent on MILP
    if include_milp:
        from vrp_benchmark.solvers.milp import MILPSolver
        milp_solver = MILPSolver(time_limit_s=milp_time_s)

    if include_cuopt:
        try:
            from vrp_benchmark.solvers.cuopt_vrp import CuOptSolver
            solvers["cuopt"] = CuOptSolver(time_limit_s=cuopt_time_s, mode="nim" if nim_mode else "self-hosted")
        except Exception as e:
            print(f"WARNING: cuOpt unavailable: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n in customer_counts:
        instances = [
            generate_instance(n, capacity=CAPACITY, seed=SEED_OFFSET + i)
            for i in range(n_eval)
        ]

        # Optionally load DQN for this n
        dqn = _load_dqn_solver(n)
        active_solvers = dict(solvers)
        if dqn:
            active_solvers["dqn"] = dqn
        else:
            print(f"  (no DQN model for n={n} — run train_dqn.py first)")

        print(f"\nn={n} customers ({n_eval} instances each):")

        # Run MILP separately to capture gap info
        # Stop running MILP once the cumulative wall time would exceed the budget.
        milp_result: tuple | None = None
        if milp_solver is not None:
            if milp_cumulative_s >= milp_budget_s:
                print(f"  (MILP skipped — cumulative budget {milp_budget_s}s exhausted)")
            else:
                milp_result = _time_milp(milp_solver, instances)
                milp_cumulative_s += milp_result[0] * len(instances)
                mt, mc, ms, opt_rate, mgap = milp_result
                cost_str = f"{mc:8.4f}" if mc < 1e8 else "    FAIL"
                gap_str = f"opt_gap={mgap:+5.1f}%" if not np.isnan(mgap) else "opt_gap=    -"
                print(
                    f"  {'milp':<14} time={mt*1000:8.1f}ms  cost={cost_str}  "
                    f"{gap_str}  optimal={opt_rate:.0%}  success={ms:.0%}"
                )

        # Run all other solvers
        results: dict[str, tuple[float, float, float]] = {}
        for name, solver in active_solvers.items():
            mean_time, mean_cost, success = _time_solver(solver, instances)
            results[name] = (mean_time, mean_cost, success)

        # Choose gap reference: MILP if all instances proven optimal, else OR-Tools, else greedy
        if milp_result is not None and milp_result[3] == 1.0:  # opt_rate == 1.0
            ref_name, ref_cost = "milp_exact", milp_result[1]
        elif "ortools" in results:
            ref_name, ref_cost = "ortools", results["ortools"][1]
        else:
            ref_name, ref_cost = "greedy", results["greedy"][1]

        for name, (mean_time, mean_cost, success) in results.items():
            gap = (mean_cost - ref_cost) / ref_cost * 100 if ref_cost < 1e8 else float("nan")
            rows.append({
                "n_customers": n,
                "solver": name,
                "mean_time_s": round(mean_time, 4),
                "mean_cost": round(mean_cost, 4) if mean_cost < 1e8 else None,
                f"gap_vs_{ref_name}_pct": round(gap, 2) if mean_cost < 1e8 else None,
                "success_rate": round(success, 3),
                "gap_reference": ref_name,
                "milp_opt_gap_pct": None,
                "milp_optimal_rate": None,
            })
            cost_str = f"{mean_cost:8.4f}" if mean_cost < 1e8 else "    FAIL"
            gap_str = f"{gap:+6.1f}%" if mean_cost < 1e8 else "      -"
            print(
                f"  {name:<14} time={mean_time*1000:8.1f}ms  "
                f"cost={cost_str}  gap_vs_{ref_name}={gap_str}  "
                f"success={success:.0%}"
            )

        # Add MILP row
        if milp_result is not None:
            mt, mc, ms, opt_rate, mgap = milp_result
            gap = (mc - ref_cost) / ref_cost * 100 if ref_cost < 1e8 and mc < 1e8 else float("nan")
            label = "milp_exact" if opt_rate == 1.0 else "milp_feasible"
            rows.append({
                "n_customers": n,
                "solver": label,
                "mean_time_s": round(mt, 4),
                "mean_cost": round(mc, 4) if mc < 1e8 else None,
                f"gap_vs_{ref_name}_pct": round(gap, 2) if mc < 1e8 else None,
                "success_rate": round(ms, 3),
                "gap_reference": ref_name,
                "milp_opt_gap_pct": round(mgap, 2) if not np.isnan(mgap) else None,
                "milp_optimal_rate": round(opt_rate, 3),
            })

    out_path = RESULTS_DIR / "vrp_benchmark.csv"
    fieldnames = [
        "n_customers", "solver", "mean_time_s", "mean_cost",
        "gap_vs_milp_exact_pct", "gap_vs_ortools_pct", "gap_vs_greedy_pct",
        "success_rate", "gap_reference", "milp_opt_gap_pct", "milp_optimal_rate",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[10, 20, 50, 100, 250, 500])
    parser.add_argument("--milp", action="store_true")
    parser.add_argument("--cuopt", action="store_true")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--ortools-time", type=int, default=30, help="OR-Tools time limit (s)")
    parser.add_argument("--milp-time", type=int, default=300, help="MILP per-instance time limit (s)")
    parser.add_argument("--milp-budget", type=int, default=3600, help="Total MILP wall-time budget (s); stops at the n that would exceed it")
    parser.add_argument("--nim", action="store_true", help="Use NVIDIA NIM cloud API instead of self-hosted (set NVIDIA_API_KEY)")
    parser.add_argument("--cuopt-time", type=int, default=10, help="cuOpt time limit (s)")
    args = parser.parse_args()
    run(
        args.counts,
        include_milp=args.milp,
        include_cuopt=args.cuopt,
        nim_mode=args.nim,
        n_eval=args.n_eval,
        ortools_time_s=args.ortools_time,
        milp_time_s=args.milp_time,
        milp_budget_s=args.milp_budget,
        cuopt_time_s=args.cuopt_time,
    )
