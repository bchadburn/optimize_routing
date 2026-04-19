"""Supply chain RL environment.

State: (day: int, demand_bucket: int, dc_status_bitmask: int)
  - day: 0–(num_days-1)
  - demand_bucket: 0=low, 1=med, 2=high (based on total demand percentiles)
  - dc_status_bitmask: integer 0–(2**num_dcs - 1) indicating open DCs

Action: integer 0–(2**num_dcs - 1) — desired DC open set as bitmask.
  Rolling window constraint is enforced before execution: any DC opened within
  the last rolling_period days cannot be closed.

Reward: -(dc_opening_costs + lp_routing_cost) for the executed action.
  Returns -1e6 if no DC is open (infeasible).
"""
from __future__ import annotations

import numpy as np
from ortools.linear_solver import pywraplp

from optimizer.construct_data_objects import SupplyChainData


class SupplyChainEnv:
    def __init__(
        self,
        supply_chain_data: SupplyChainData,
        num_days: int = 10,
        decision_rolling_period: int = 3,
        seed: int | None = None,
    ) -> None:
        self.data = supply_chain_data
        self.num_days = num_days
        self.rolling_period = decision_rolling_period
        self.num_dcs = len(supply_chain_data.distribution_sites)
        self.rng = np.random.default_rng(seed)

        # Compute demand percentile thresholds from distribution (1000 samples)
        samples = np.array([
            sum(
                max(0, self.rng.normal(c.mean_demand, c.std_dev_demand))
                for c in supply_chain_data.customers.values()
            )
            for _ in range(1000)
        ])
        self._low_thresh = float(np.percentile(samples, 33))
        self._high_thresh = float(np.percentile(samples, 66))

        self._day: int = 0
        self._dc_status_bitmask: int = 0
        self._open_start: dict[int, int] = {}  # dc_id -> day it was last opened
        self.forced_open_mask: int = 0
        self._daily_demands: list[dict[int, float]] = []

    def reset(self, seed: int | None = None) -> tuple[int, int, int]:
        """Reset environment. Returns initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._day = 0
        self._dc_status_bitmask = 0
        self._open_start = {}
        self.forced_open_mask = 0
        self._daily_demands = self._sample_demands()
        return self._get_state()

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        """Execute action. Returns (next_state, reward, done)."""
        if self._day >= self.num_days:
            raise RuntimeError("Episode is done. Call reset() first.")

        executed_action = self._enforce_rolling_window(action)
        reward = self._compute_reward(executed_action)
        self._update_dc_status(executed_action)
        self._day += 1
        done = self._day >= self.num_days
        return self._get_state(), reward, done

    def _get_state(self) -> tuple[int, int, int]:
        total_demand = sum(self._daily_demands[min(self._day, self.num_days - 1)].values())
        if total_demand <= self._low_thresh:
            bucket = 0
        elif total_demand <= self._high_thresh:
            bucket = 1
        else:
            bucket = 2
        return (self._day, bucket, self._dc_status_bitmask)

    def _sample_demands(self) -> list[dict[int, float]]:
        return [
            {
                cust_id: max(0.0, float(self.rng.normal(c.mean_demand, c.std_dev_demand)))
                for cust_id, c in self.data.customers.items()
            }
            for _ in range(self.num_days)
        ]

    def _enforce_rolling_window(self, desired_action: int) -> int:
        """Return the executed action: desired OR forced-open DCs."""
        self.forced_open_mask = 0
        for dc_id, open_day in self._open_start.items():
            if self._day < open_day + self.rolling_period:
                self.forced_open_mask |= (1 << dc_id)
        return desired_action | self.forced_open_mask

    def _update_dc_status(self, executed_action: int) -> None:
        prev = self._dc_status_bitmask
        self._dc_status_bitmask = executed_action
        for dc_id in range(self.num_dcs):
            newly_opened = (executed_action >> dc_id) & 1 and not (prev >> dc_id) & 1
            if newly_opened:
                self._open_start[dc_id] = self._day

    def _compute_reward(self, executed_action: int) -> float:
        if executed_action == 0:
            return -1e6  # no DCs open — infeasible

        open_dcs = [dc_id for dc_id in range(self.num_dcs) if (executed_action >> dc_id) & 1]
        demands = self._daily_demands[self._day]

        # DC opening cost: incurred only when DC wasn't open in the previous rolling window
        dc_cost = 0.0
        for dc_id in open_dcs:
            last_opened = self._open_start.get(dc_id, -self.rolling_period)
            if self._day >= last_opened + self.rolling_period or last_opened == -self.rolling_period:
                dc_cost += self.data.distribution_sites[dc_id].opening_cost

        lp_cost = self._solve_routing_lp(open_dcs, demands)
        return -(dc_cost + lp_cost)

    def _solve_routing_lp(self, open_dcs: list[int], demands: dict[int, float]) -> float:
        """Solve a single-day LP to find optimal routing cost given open DCs."""
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return 1e6

        mf_ids = list(self.data.manufacturing_sites.keys())
        cust_ids = list(self.data.customers.keys())

        x_md = {
            (m, d): solver.NumVar(0, solver.infinity(), f"x_md_{m}_{d}")
            for m in mf_ids for d in open_dcs
        }
        x_dc = {
            (d, c): solver.NumVar(0, solver.infinity(), f"x_dc_{d}_{c}")
            for d in open_dcs for c in cust_ids
        }

        for m in mf_ids:
            solver.Add(sum(x_md[m, d] for d in open_dcs) <= self.data.manufacturing_sites[m].capacity)

        for c in cust_ids:
            solver.Add(sum(x_dc[d, c] for d in open_dcs) == demands[c])

        for d in open_dcs:
            solver.Add(
                sum(x_md[m, d] for m in mf_ids) == sum(x_dc[d, c] for c in cust_ids)
            )

        obj = solver.Objective()
        for m in mf_ids:
            for d in open_dcs:
                obj.SetCoefficient(x_md[m, d], self.data.manufacturing_sites[m].transport_cost_m_to_d[d])
        for d in open_dcs:
            for c in cust_ids:
                obj.SetCoefficient(x_dc[d, c], self.data.distribution_sites[d].transport_cost_d_to_c[c])
        obj.SetMinimization()

        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return solver.Objective().Value()
        return 1e6
