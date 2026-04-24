"""Train a DQN agent on CVRP instances and save the model.

Usage:
    uv run python -m vrp_benchmark.train_dqn
    uv run python -m vrp_benchmark.train_dqn --n 50 --episodes 10000 --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vrp_benchmark.solvers.dqn import DQNSolver, generate_instance, train


MODELS_DIR = Path("vrp_benchmark/models")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of customers")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=32, help="Parallel environments")
    args = parser.parse_args()

    print(f"Training DQN: n={args.n}, episodes={args.episodes}, device={args.device}")
    agent = train(
        n_customers=args.n,
        n_episodes=args.episodes,
        capacity=args.capacity,
        device=args.device,
        seed=args.seed,
        n_envs=args.n_envs,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"dqn_n{args.n}.pt"
    agent.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Quick eval on 20 held-out instances
    print("\nEvaluating on 20 held-out instances...")
    solver = DQNSolver(agent)
    costs = []
    for i in range(20):
        inst = generate_instance(args.n, capacity=args.capacity, seed=10_000 + i)
        _, cost = solver.solve(inst)
        costs.append(cost)

    import numpy as np
    print(f"  Mean cost: {np.mean(costs):.4f}  Std: {np.std(costs):.4f}")


if __name__ == "__main__":
    main()
