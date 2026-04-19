"""Training loop, policy evaluation, and results export for Q-learning agent."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from optimizer.run_optimizer import build_supply_chain_data, run_global_milp, run_daily_myopic
from rl.agent import QLearningAgent
from rl.environment import SupplyChainEnv
from utils.results import write_csv, write_learning_curve, write_policy_table

RESULTS_DIR = Path("results")

DEFAULT_PARAMS = dict(
    distribution_opening_costs=[350, 320, 375, 400, 550],
    mfg_site_capacity=[600000, 600000],
    mean_demand=[20, 30, 25, 40, 35, 28, 32, 50, 26, 38, 34, 27],
    std_dev_demand=[5.0] * 12,
    transport_cost_m_to_d=[
        [3.5, 2.5, 4.5, 2.5, 3.0],
        [2.5, 4.5, 5.5, 6.5, 8.5],
    ],
    transport_cost_d_to_c=[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    ],
)


def train(
    episodes: int = 15_000,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    seed: int = 42,
    log_interval: int = 500,
) -> tuple[QLearningAgent, list[float]]:
    """Train Q-learning agent. Returns (agent, episode_rewards)."""
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnv(supply_chain_data, num_days, decision_rolling_period, seed=seed)
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
            print(f"Episode {ep+1}/{episodes} | avg reward (last {log_interval}): {recent:.1f} | epsilon={agent.epsilon:.4f}")

    return agent, episode_rewards


def evaluate_rl(
    agent: QLearningAgent,
    num_eval_episodes: int = 100,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    seed: int = 99,
) -> dict:
    """Evaluate greedy policy over multiple episodes."""
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnv(supply_chain_data, num_days, decision_rolling_period, seed=seed)
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
    return {
        "mean_total_cost": float(np.mean(costs)),
        "std_total_cost": float(np.std(costs)),
        "costs_per_episode": costs,
    }


def extract_policy_table(
    agent: QLearningAgent,
    num_days: int = 10,
) -> dict[tuple[int, int], int]:
    """Extract greedy policy as (day, demand_bucket) -> action map."""
    policy = {}
    for day in range(num_days):
        for bucket in range(3):
            action = int(np.argmax(agent.q_table[day, bucket, 0]))
            policy[(day, bucket)] = action
    return policy


def run_all(
    episodes: int = 15_000,
    num_milp_simulations: int = 10,
    seed: int = 42,
) -> None:
    """Run all three methods and write results CSVs."""
    RESULTS_DIR.mkdir(exist_ok=True)
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)

    print("Running MILP global solve...")
    milp_global = run_global_milp(supply_chain_data)
    write_csv(
        [{"method": "milp_global", "total_cost": milp_global["total_cost"],
          "transport_cost_m_to_d": milp_global["transport_cost_m_to_d"],
          "transport_cost_d_to_c": milp_global["transport_cost_d_to_c"]}],
        RESULTS_DIR / "milp_global.csv",
    )
    print(f"  MILP global total cost: ${milp_global['total_cost']:,.0f}")

    print("Running MILP daily myopic solves...")
    milp_daily = run_daily_myopic(supply_chain_data, num_simulations=num_milp_simulations)
    write_csv(
        [{"method": "milp_daily", "simulation": i, "total_cost": c}
         for i, c in enumerate(milp_daily["costs_per_simulation"])],
        RESULTS_DIR / "milp_daily.csv",
    )
    print(f"  MILP daily mean cost: ${milp_daily['mean_total_cost']:,.0f} +/- {milp_daily['std_total_cost']:,.0f}")

    print(f"Training Q-learning agent ({episodes} episodes)...")
    agent, episode_rewards = train(episodes=episodes, seed=seed)
    write_learning_curve(episode_rewards, RESULTS_DIR / "learning_curve.csv")

    print("Evaluating RL policy...")
    rl_results = evaluate_rl(agent, seed=seed + 1)
    write_csv(
        [{"method": "rl", "episode": i, "total_cost": c}
         for i, c in enumerate(rl_results["costs_per_episode"])],
        RESULTS_DIR / "rl_policy.csv",
    )
    print(f"  RL mean cost: ${rl_results['mean_total_cost']:,.0f} +/- {rl_results['std_total_cost']:,.0f}")

    policy_map = extract_policy_table(agent)
    write_policy_table(
        policy_map,
        RESULTS_DIR / "rl_policy_table.csv",
        num_dcs=len(supply_chain_data.distribution_sites),
    )

    milp_opt = milp_global["total_cost"]
    rl_mean = rl_results["mean_total_cost"]
    daily_mean = milp_daily["mean_total_cost"]
    print("\n=== Summary ===")
    print(f"{'Method':<25} {'Mean Cost':>12} {'Gap vs MILP Global':>20}")
    print(f"{'MILP Global (optimal)':<25} ${milp_opt:>11,.0f} {'--':>20}")
    print(f"{'MILP Daily Myopic':<25} ${daily_mean:>11,.0f} {(daily_mean/milp_opt - 1)*100:>19.1f}%")
    print(f"{'Q-Learning':<25} ${rl_mean:>11,.0f} {(rl_mean/milp_opt - 1)*100:>19.1f}%")


if __name__ == "__main__":
    run_all()
