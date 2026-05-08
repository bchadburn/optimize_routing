"""Chance-constrained MILP wrapper for Experiment 1 (multi-echelon supply chain).

Reduces a Bayesian posterior of customer demand into a single per-day percentile
vector, then feeds it to the existing global MILP unchanged. By construction
the resulting plan satisfies demand in at least ``confidence``% of posterior
scenarios — the cheapest possible "stochastic-aware" upgrade over the deterministic
baseline. No new variables, no new constraints, no new solver dependency.

This mirrors Approach 1 from the FICO Xpress Community blog post on PyMC + robust
optimization (Vieira & Saunders, 2026), adapted from energy LP to multi-echelon
MILP.

Usage:

    scenarios, customers = forecast_demand(history, horizon_days=10, cache_path=...)
    result = solve_chance_constrained(
        scenarios=scenarios,
        customers=customers,
        supply_chain_data=base_data,    # the existing SupplyChainData (no daily_demand set)
        num_days=10,
        confidence=0.95,
    )

The shortage rate is computed via Monte Carlo against the *full* posterior so it
isn't tautologically zero (the optimizer only ever saw the percentile vector).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimizer.construct_data_objects import SupplyChainData
from optimizer.run_optimizer import run_global_milp


@dataclass(frozen=True)
class ChanceConstrainedResult:
    """Output of ``solve_chance_constrained`` — what the comparison row needs.

    ``shortage_rate_pct`` is the empirical fraction of (scenario, day) pairs where
    the resulting plan would have failed under the full posterior. By construction
    this should be ≤ (1 - confidence) on the same posterior we sampled the
    percentile from; computed via Monte Carlo so the comparison against CVaR is
    apples-to-apples.
    """

    total_cost: float
    shortage_rate_pct: float
    demand_pct: np.ndarray            # (num_days, num_customers) — what the optimizer saw
    dc_decisions: list[set]           # per-day chosen distribution sites


def _build_data_with_daily_demand(
    base_data: SupplyChainData,
    daily_demand: np.ndarray,
    customers: list[int],
) -> SupplyChainData:
    """Clone ``base_data`` and populate each customer's ``daily_demand`` with
    the corresponding column of ``daily_demand`` (shape ``(n_days, n_customers)``).

    Uses ``SupplyChainData.clone()`` so future schema additions on the data
    container are picked up without changes here.
    """
    new = base_data.clone()
    for col_idx, cust_id in enumerate(customers):
        if cust_id not in new.customers:
            raise ValueError(f"customer {cust_id} not in supply_chain_data")
        new.customers[cust_id].daily_demand = daily_demand[:, col_idx].tolist()
    return new


def _calculate_shortage_rate(
    daily_demand_used: np.ndarray,     # (n_days, n_customers) the optimizer saw
    full_scenarios: np.ndarray,        # (n_scenarios, n_days, n_customers) posterior
) -> float:
    """Monte Carlo shortage rate: fraction of (scenario, day, customer) cells
    where the actual scenario demand exceeds the planning value the optimizer
    used. By construction the per-customer 95th-percentile plan should miss in
    ≤ 5% of cells.

    Mirrors FICO's ``calculate_shortage_rate`` at
    ``stochastic_energy_planning.py:122``.
    """
    # Broadcast (n_days, n_cust) against (n_scenarios, n_days, n_cust)
    plan = daily_demand_used[None, :, :]
    return float(np.mean(full_scenarios > plan) * 100)


def solve_chance_constrained(
    scenarios: np.ndarray,                  # (n_scenarios, n_days, n_customers)
    customers: list[int],                   # column ids in `scenarios` last axis
    supply_chain_data: SupplyChainData,
    num_days: int = 10,
    confidence: float = 0.95,
    decision_rolling_period: int = 3,
) -> ChanceConstrainedResult:
    """Solve the chance-constrained MILP for the multi-echelon SC problem.

    Args:
        scenarios: posterior predictive demand from ``forecast_demand``, shape
            ``(n_scenarios, n_days, n_customers)``.
        customers: customer ids in the column order of ``scenarios`` last axis.
        supply_chain_data: existing SupplyChainData (without per-customer
            ``daily_demand`` set). The function does not mutate this — it
            returns a fresh ``SupplyChainData`` clone with ``daily_demand``
            populated.
        num_days: planning horizon in days. Must match ``scenarios.shape[1]``.
        confidence: probability that the plan satisfies demand under the
            posterior. Default 0.95 (5% acceptable shortage rate).
        decision_rolling_period: passed through to the existing MILP.

    Returns:
        ``ChanceConstrainedResult`` with total cost, Monte Carlo shortage rate,
        the demand vector the optimizer actually saw, and per-day DC decisions.
    """
    if scenarios.ndim != 3:
        raise ValueError(f"scenarios must be 3D, got shape {scenarios.shape}")
    if scenarios.shape[1] != num_days:
        raise ValueError(
            f"scenarios horizon {scenarios.shape[1]} != num_days {num_days}"
        )
    if scenarios.shape[2] != len(customers):
        raise ValueError(
            f"scenarios n_customers {scenarios.shape[2]} != len(customers) {len(customers)}"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    # Per-day per-customer percentile collapses 4000 scenarios -> 1 vector.
    demand_pct = np.percentile(scenarios, confidence * 100, axis=0)
    assert demand_pct.shape == (num_days, len(customers))

    # Build a fresh SupplyChainData with daily_demand set per customer.
    data_with_demand = _build_data_with_daily_demand(
        supply_chain_data, demand_pct, customers,
    )

    milp_result = run_global_milp(
        data_with_demand,
        num_days=num_days,
        decision_rolling_period=decision_rolling_period,
    )

    shortage_rate = _calculate_shortage_rate(demand_pct, scenarios)

    return ChanceConstrainedResult(
        total_cost=milp_result["total_cost"],
        shortage_rate_pct=shortage_rate,
        demand_pct=demand_pct,
        dc_decisions=milp_result["dc_decisions"],
    )
