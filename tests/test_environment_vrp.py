"""Integration tests: one episode of SupplyChainEnvVrp produces valid cost."""
from optimizer.run_optimizer import build_supply_chain_data
from rl.train import DEFAULT_PARAMS


def _run_episode(env) -> float:
    """Run one episode with all DCs open, return total cost."""
    state = env.reset()
    total_cost = 0.0
    done = False
    while not done:
        action = (1 << env.num_dcs) - 1  # all DCs open
        state, reward, done = env.step(action)
        total_cost -= reward
    return total_cost


def test_ortools_episode_produces_finite_cost():
    """Episode with OR-Tools VRP sub-solver returns finite positive cost."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(data, num_days=3, solver=OrtoolsVrpSolver(), seed=0)
    cost = _run_episode(env)

    assert cost > 0.0
    assert cost < 1e6  # not hitting infeasibility sentinel


def test_default_solver_is_ortools():
    """SupplyChainEnvVrp defaults to OrtoolsVrpSolver when no solver passed."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(data, num_days=2, seed=0)
    assert isinstance(env._solver, OrtoolsVrpSolver)


def test_zero_dc_action_returns_large_penalty():
    """Action 0 (no DCs open) returns reward of -1e6."""
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(data, num_days=2, seed=0)
    env.reset()
    _, reward, _ = env.step(0)
    assert reward == -1e6
