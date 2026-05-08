"""Phase 2 tests: chance-constrained MILP wrapper around the existing global MILP."""
from __future__ import annotations

import numpy as np
import pytest

from optimizer.run_optimizer import build_supply_chain_data
from vrp_benchmark.solvers.chance_constrained_milp import (
    ChanceConstrainedResult,
    _calculate_shortage_rate,
    solve_chance_constrained,
)

# Match the existing test fixtures
BASE_PARAMS = dict(
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


def _fake_scenarios(n_scenarios: int, n_days: int, n_customers: int, seed: int = 0):
    """Build a synthetic posterior-shaped array. We use a known distribution so
    the percentile is predictable without invoking PyMC."""
    rng = np.random.default_rng(seed)
    means = np.array(BASE_PARAMS["mean_demand"][:n_customers])
    return np.maximum(
        0.0,
        rng.normal(
            loc=means[None, None, :],   # (1, 1, n_cust)
            scale=8.0,
            size=(n_scenarios, n_days, n_customers),
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def test_calculate_shortage_rate_returns_zero_when_plan_dominates():
    scenarios = _fake_scenarios(50, 5, 4, seed=1)
    plan = scenarios.max(axis=0) + 1.0   # plan above max scenario for every cell
    rate = _calculate_shortage_rate(plan, scenarios)
    assert rate == 0.0


def test_calculate_shortage_rate_returns_100_when_plan_is_zero():
    scenarios = _fake_scenarios(50, 5, 4, seed=1)
    plan = np.zeros_like(scenarios.max(axis=0))
    rate = _calculate_shortage_rate(plan, scenarios)
    # Every cell with positive demand triggers shortage; with our synth all > 0
    assert rate > 99.0


# ── Validation guards ─────────────────────────────────────────────────────────


def test_solve_chance_constrained_rejects_wrong_dim_scenarios():
    data = build_supply_chain_data(**BASE_PARAMS)
    bad = np.zeros((5, 10))   # 2D not 3D
    with pytest.raises(ValueError, match="must be 3D"):
        solve_chance_constrained(
            scenarios=bad,
            customers=list(range(12)),
            supply_chain_data=data,
            num_days=10,
        )


def test_solve_chance_constrained_rejects_horizon_mismatch():
    data = build_supply_chain_data(**BASE_PARAMS)
    # Scenarios horizon=5 but num_days=10
    bad = _fake_scenarios(20, 5, 12, seed=1)
    with pytest.raises(ValueError, match="horizon"):
        solve_chance_constrained(
            scenarios=bad,
            customers=list(range(12)),
            supply_chain_data=data,
            num_days=10,
        )


def test_solve_chance_constrained_rejects_customer_count_mismatch():
    """Scenarios last-axis must equal len(customers)."""
    data = build_supply_chain_data(**BASE_PARAMS)
    # Scenarios with 5 customers but the customers list claims 12 -> rejected.
    bad = _fake_scenarios(20, 10, 5, seed=1)
    with pytest.raises(ValueError, match="n_customers"):
        solve_chance_constrained(
            scenarios=bad,
            customers=list(range(12)),
            supply_chain_data=data,
            num_days=10,
        )


@pytest.mark.parametrize("confidence", [-0.1, 0.0, 1.0, 1.5])
def test_solve_chance_constrained_rejects_invalid_confidence(confidence):
    data = build_supply_chain_data(**BASE_PARAMS)
    scenarios = _fake_scenarios(20, 10, 12, seed=1)
    with pytest.raises(ValueError, match="confidence"):
        solve_chance_constrained(
            scenarios=scenarios,
            customers=list(range(12)),
            supply_chain_data=data,
            num_days=10,
            confidence=confidence,
        )


# ── End-to-end behaviour ─────────────────────────────────────────────────────


def test_solve_chance_constrained_runs_and_returns_expected_shape():
    """End-to-end: feed synthetic scenarios, get a ChanceConstrainedResult back
    with sensible cost and shortage rate."""
    data = build_supply_chain_data(**BASE_PARAMS)
    n_days = 10
    n_customers = 12
    scenarios = _fake_scenarios(200, n_days, n_customers, seed=1)

    result = solve_chance_constrained(
        scenarios=scenarios,
        customers=list(range(n_customers)),
        supply_chain_data=data,
        num_days=n_days,
        confidence=0.95,
    )

    assert isinstance(result, ChanceConstrainedResult)
    assert result.total_cost > 0
    # Shape check: per-day per-customer plan
    assert result.demand_pct.shape == (n_days, n_customers)
    # Per construction the 95th-percentile plan should miss ≤ 5% of (scenario,
    # day, customer) cells. Allow a 1pp tolerance for sample noise.
    assert result.shortage_rate_pct <= 6.0
    assert len(result.dc_decisions) == n_days


def test_chance_constrained_costs_more_than_mean_demand_baseline():
    """A 95th-percentile plan must cost ≥ a mean-demand plan — we're paying for
    robustness. Loose bound: expect at least equal cost (sometimes the global
    optimum is identical because the higher demand doesn't change which DCs
    open — we just ship more)."""
    data = build_supply_chain_data(**BASE_PARAMS)
    n_days = 10
    n_cust = 12
    scenarios = _fake_scenarios(300, n_days, n_cust, seed=42)

    p95_result = solve_chance_constrained(
        scenarios=scenarios,
        customers=list(range(n_cust)),
        supply_chain_data=data,
        num_days=n_days,
        confidence=0.95,
    )
    p50_result = solve_chance_constrained(
        scenarios=scenarios,
        customers=list(range(n_cust)),
        supply_chain_data=data,
        num_days=n_days,
        confidence=0.50,
    )

    # 95th-percentile demand >= 50th-percentile demand cell-by-cell
    np.testing.assert_array_compare(np.greater_equal, p95_result.demand_pct, p50_result.demand_pct)
    # ⇒ total cost of meeting it is ≥ as well
    assert p95_result.total_cost >= p50_result.total_cost - 1e-3
    # And the 95th-percentile plan must have a lower shortage rate
    assert p95_result.shortage_rate_pct <= p50_result.shortage_rate_pct


def test_chance_constrained_does_not_mutate_input_supply_chain_data():
    """The wrapper must not modify the SupplyChainData the caller passed in —
    other solvers may run against the same instance."""
    data = build_supply_chain_data(**BASE_PARAMS)
    n_days, n_cust = 10, 12
    scenarios = _fake_scenarios(50, n_days, n_cust, seed=1)

    customers_before = {cid: cust.daily_demand for cid, cust in data.customers.items()}

    solve_chance_constrained(
        scenarios=scenarios,
        customers=list(range(n_cust)),
        supply_chain_data=data,
        num_days=n_days,
    )

    customers_after = {cid: cust.daily_demand for cid, cust in data.customers.items()}
    assert customers_before == customers_after, (
        "solve_chance_constrained mutated input SupplyChainData — must clone instead"
    )
