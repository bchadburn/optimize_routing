"""Train and evaluate RL agent using SupplyChainEnvVrp with a pluggable solver.

Usage:
    uv run python -m rl.train_vrp --solver ortools   # OR-Tools VRP sub-solver
    uv run python -m rl.train_vrp --solver cuopt     # cuOpt sub-solver (requires GPU)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimizer.run_optimizer import build_supply_chain_data
from rl.agent import QLearningAgent
from rl.environment_vrp import SupplyChainEnvVrp
from rl.train import DEFAULT_PARAMS
from utils.results import write_csv, write_learning_curve

RESULTS_DIR = Path("results")


def train_vrp(
    solver_name: str = "ortools",
    episodes: int = 5_000,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    n_vehicles_per_dc: int = 3,
    seed: int = 42,
    log_interval: int = 500,
) -> tuple[QLearningAgent, list[float]]:
    """Train Q-learning agent with VRP sub-solver. Returns (agent, episode_rewards)."""
    solver = _make_solver(solver_name)
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(
        supply_chain_data, num_days, decision_rolling_period,
        seed=seed, solver=solver, n_vehicles_per_dc=n_vehicles_per_dc,
    )
    agent = QLearningAgent(
        num_days=num_days,
        num_demand_buckets=3,
        num_dcs=len(supply_chain_data.distribution_sites),
        seed=seed,
    )

    episode_rewards: list[float] = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % log_interval == 0:
            recent = np.mean(episode_rewards[-log_interval:])
            print(
                f"[{solver_name}] Episode {ep+1}/{episodes} | "
                f"avg reward: {recent:.1f} | epsilon={agent.epsilon:.4f}"
            )

    return agent, episode_rewards


def evaluate_vrp(
    agent: QLearningAgent,
    solver_name: str = "ortools",
    num_eval_episodes: int = 50,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    n_vehicles_per_dc: int = 3,
    seed: int = 99,
) -> list[float]:
    """Evaluate greedy policy. Returns list of per-episode total costs."""
    solver = _make_solver(solver_name)
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(
        supply_chain_data, num_days, decision_rolling_period,
        seed=seed, solver=solver, n_vehicles_per_dc=n_vehicles_per_dc,
    )
    costs = []
    for _ in range(num_eval_episodes):
        state = env.reset()
        total_cost = 0.0
        done = False
        while not done:
            action = agent.greedy_action(state)
            state, reward, done = env.step(action)
            total_cost -= reward
        costs.append(total_cost)
    return costs


def _make_solver(solver_name: str):
    if solver_name == "cuopt":
        from rl.solvers.cuopt_vrp import CuOptVrpSolver
        return CuOptVrpSolver()
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    return OrtoolsVrpSolver()


def run_vrp(solver_name: str = "ortools", episodes: int = 5_000) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Training RL with {solver_name} VRP solver ({episodes} episodes)...")
    agent, episode_rewards = train_vrp(solver_name=solver_name, episodes=episodes)

    write_learning_curve(episode_rewards, RESULTS_DIR / f"rl_vrp_{solver_name}_curve.csv")

    print("Evaluating greedy policy...")
    costs = evaluate_vrp(agent, solver_name=solver_name)
    mean_cost = float(np.mean(costs))
    std_cost = float(np.std(costs))
    print(f"[{solver_name}] Eval mean cost: ${mean_cost:,.0f} +/- ${std_cost:,.0f}")

    write_csv(
        [{"solver": solver_name, "episode": i, "total_cost": c} for i, c in enumerate(costs)],
        RESULTS_DIR / f"rl_vrp_{solver_name}.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["ortools", "cuopt"], default="ortools")
    parser.add_argument("--episodes", type=int, default=5_000)
    args = parser.parse_args()
    run_vrp(solver_name=args.solver, episodes=args.episodes)
