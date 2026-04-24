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


def run(
    customer_counts: list[int],
    include_milp: bool = False,
    include_cuopt: bool = False,
    n_eval: int = N_EVAL,
) -> None:
    solvers: dict[str, object] = {
        "greedy": GreedySolver(),
        "ortools": ORToolsSolver(time_limit_s=120),
    }

    if include_milp:
        from vrp_benchmark.solvers.milp import MILPSolver
        solvers["milp_exact"] = MILPSolver(time_limit_s=120)

    if include_cuopt:
        try:
            from vrp_benchmark.solvers.cuopt_vrp import CuOptSolver
            solvers["cuopt"] = CuOptSolver(time_limit_s=10)
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

        # Collect all results first, then compute gaps vs the best available reference.
        # Reference priority: MILP exact (proven optimal) > OR-Tools > Greedy.
        # Gap vs a non-exact reference is labelled accordingly in the CSV.
        results: dict[str, tuple[float, float, float]] = {}
        print(f"\nn={n} customers ({n_eval} instances each):")
        for name, solver in active_solvers.items():
            mean_time, mean_cost, success = _time_solver(solver, instances)
            results[name] = (mean_time, mean_cost, success)

        # Choose gap reference: MILP if it succeeded (all instances proven optimal),
        # otherwise OR-Tools, otherwise greedy.
        if "milp_exact" in results and results["milp_exact"][2] == 1.0:
            ref_name, ref_cost = "milp_exact", results["milp_exact"][1]
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
            })
            cost_str = f"{mean_cost:8.4f}" if mean_cost < 1e8 else "    FAIL"
            gap_str = f"{gap:+6.1f}%" if mean_cost < 1e8 else "      -"
            print(
                f"  {name:<14} time={mean_time*1000:8.1f}ms  "
                f"cost={cost_str}  gap_vs_{ref_name}={gap_str}  "
                f"success={success:.0%}"
            )

    out_path = RESULTS_DIR / "vrp_benchmark.csv"
    fieldnames = ["n_customers", "solver", "mean_time_s", "mean_cost", "gap_vs_milp_exact_pct",
                  "gap_vs_ortools_pct", "gap_vs_greedy_pct", "success_rate", "gap_reference"]
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
    args = parser.parse_args()
    run(args.counts, include_milp=args.milp, include_cuopt=args.cuopt, n_eval=args.n_eval)
