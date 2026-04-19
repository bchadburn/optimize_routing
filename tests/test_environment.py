import pytest
import numpy as np
from optimizer.run_optimizer import build_supply_chain_data
from rl.environment import SupplyChainEnv

PARAMS = dict(
    distribution_opening_costs=[350, 320, 375, 400, 550],
    mfg_site_capacity=[600000, 600000],
    mean_demand=[20, 30, 25, 40, 35, 28, 32, 50, 26, 38, 34, 27],
    std_dev_demand=[5.0] * 12,
    transport_cost_m_to_d=[[3.5, 2.5, 4.5, 2.5, 3.0], [2.5, 4.5, 5.5, 6.5, 8.5]],
    transport_cost_d_to_c=[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    ],
)

@pytest.fixture
def env():
    data = build_supply_chain_data(**PARAMS)
    return SupplyChainEnv(supply_chain_data=data, num_days=10, decision_rolling_period=3, seed=42)

def test_reset_returns_valid_state(env):
    state = env.reset()
    day, demand_bucket, dc_mask = state
    assert day == 0
    assert demand_bucket in (0, 1, 2)
    assert 0 <= dc_mask <= 31

def test_step_returns_negative_reward(env):
    env.reset()
    # Open DC 0 only (bitmask = 1)
    next_state, reward, done = env.step(action=1)
    assert reward < 0
    assert not done

def test_done_on_final_day(env):
    env.reset()
    for _ in range(9):
        env.step(action=1)
    _, _, done = env.step(action=1)
    assert done

def test_rolling_window_forces_dc_open(env):
    """DC opened on day 0 must stay open for rolling_period days."""
    env.reset()
    # Open DC 0 (bitmask=1) on day 0
    env.step(action=1)
    # Try to close all DCs (action=0) on day 1 — DC 0 must remain forced open
    _, _, _ = env.step(action=0)
    assert env.forced_open_mask & 1  # bit 0 still set

def test_demand_bucket_coverage(env):
    """After many resets, all 3 demand buckets should appear."""
    buckets = set()
    for _ in range(200):
        state = env.reset()
        buckets.add(state[1])
    assert buckets == {0, 1, 2}
