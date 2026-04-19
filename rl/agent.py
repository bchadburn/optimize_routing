"""Tabular Q-learning agent for supply chain DC-open decisions.

Q-table shape: (num_days, num_demand_buckets, dc_status_bitmask, action)
  = (10, 3, 32, 32) for a 5-DC, 10-day problem.

State: (day, demand_bucket, dc_status_bitmask)
Action: integer 0–31 — desired DC open set as bitmask.
"""
from __future__ import annotations

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_days: int,
        num_demand_buckets: int,
        num_dcs: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        self.num_days = num_days
        self.num_dcs = num_dcs
        self.num_actions = 2 ** num_dcs
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (num_days, num_demand_buckets, self.num_actions, self.num_actions),
            dtype=np.float64,
        )

    def select_action(self, state: tuple[int, int, int]) -> int:
        """Epsilon-greedy action selection. Returns action bitmask."""
        day, demand_bucket, dc_mask = state
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(self.q_table[day, demand_bucket, dc_mask]))

    def update(
        self,
        state: tuple[int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int],
        done: bool,
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))"""
        day, demand_bucket, dc_mask = state
        current_q = self.q_table[day, demand_bucket, dc_mask, action]

        if done:
            target = reward
        else:
            next_day, next_bucket, next_dc_mask = next_state
            next_max_q = np.max(self.q_table[next_day, next_bucket, next_dc_mask])
            target = reward + self.gamma * next_max_q

        self.q_table[day, demand_bucket, dc_mask, action] += self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_action(self, state: tuple[int, int, int]) -> int:
        """Return greedy action for policy evaluation (no exploration)."""
        day, demand_bucket, dc_mask = state
        return int(np.argmax(self.q_table[day, demand_bucket, dc_mask]))
