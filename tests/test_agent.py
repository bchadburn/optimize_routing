import numpy as np
import pytest

from rl.agent import QLearningAgent


def test_q_table_shape():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    assert agent.q_table.shape == (10, 3, 32, 32)

def test_initial_q_table_is_zero():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    assert np.all(agent.q_table == 0.0)

def test_select_action_returns_valid_action():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    action = agent.select_action(state=(0, 1, 0))
    assert 0 <= action <= 31

def test_epsilon_greedy_explores_at_high_epsilon():
    """With epsilon=1.0 all actions should be random (non-greedy)."""
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, epsilon=1.0, seed=0)
    agent.q_table[0, 0, 0, 5] = 999.0
    actions = [agent.select_action((0, 0, 0)) for _ in range(50)]
    assert len(set(actions)) > 1

def test_update_changes_q_value():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, alpha=0.5, gamma=0.9)
    state = (0, 1, 0)
    action = 3
    reward = -500.0
    next_state = (1, 1, 3)
    agent.update(state, action, reward, next_state, done=False)
    assert agent.q_table[0, 1, 0, 3] != 0.0

def test_update_at_terminal_state_ignores_next():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, alpha=1.0, gamma=0.9)
    state = (9, 0, 0)
    action = 1
    reward = -300.0
    next_state = (10, 0, 0)  # beyond horizon
    agent.update(state, action, reward, next_state, done=True)
    assert agent.q_table[9, 0, 0, 1] == pytest.approx(-300.0)

def test_epsilon_decays():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5,
                           epsilon=1.0, epsilon_end=0.01, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.25)

def test_epsilon_does_not_decay_below_epsilon_end():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5,
                           epsilon=0.015, epsilon_end=0.01, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.01)
