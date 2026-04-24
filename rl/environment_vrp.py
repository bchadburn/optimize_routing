"""SupplyChainEnv subclass that accepts any CvrptwSolver for the DC→customer leg.

The DC→customer routing uses the pluggable CvrptwSolver (OR-Tools VRP or cuOpt).
The manufacturing→DC flow is still solved with an LP, identical to the original
_solve_routing_lp, because:
  - mfg→DC is a continuous-flow allocation (no vehicle routing structure)
  - It accounts for ~57% of total cost in this problem ($9,674 of $17,020)
  - Omitting it makes RL rewards incomparable to the MILP benchmark

All state management, rolling-window, and episode logic is inherited from SupplyChainEnv.
"""
from __future__ import annotations

from ortools.linear_solver import pywraplp

from optimizer.construct_data_objects import SupplyChainData
from rl.environment import SupplyChainEnv
from rl.solvers.protocol import CvrptwSolver


class SupplyChainEnvVrp(SupplyChainEnv):
    def __init__(
        self,
        supply_chain_data: SupplyChainData,
        num_days: int = 10,
        decision_rolling_period: int = 3,
        seed: int | None = None,
        solver: CvrptwSolver | None = None,
        n_vehicles_per_dc: int = 3,
    ) -> None:
        super().__init__(supply_chain_data, num_days, decision_rolling_period, seed)
        if solver is None:
            from rl.solvers.ortools_vrp import OrtoolsVrpSolver
            solver = OrtoolsVrpSolver()
        self._solver = solver
        self._n_vehicles_per_dc = n_vehicles_per_dc

    def _compute_reward(self, executed_action: int) -> float:
        if executed_action == 0:
            return -1e6

        open_dcs = [dc_id for dc_id in range(self.num_dcs) if (executed_action >> dc_id) & 1]
        demands = self._daily_demands[self._day]

        dc_cost = 0.0
        for dc_id in open_dcs:
            was_open = (self._dc_status_bitmask >> dc_id) & 1
            if not was_open:
                dc_cost += self.data.distribution_sites[dc_id].opening_cost

        transport_costs = {
            dc_id: dict(self.data.distribution_sites[dc_id].transport_cost_d_to_c)
            for dc_id in open_dcs
        }
        float_demands = {cid: float(qty) for cid, qty in demands.items()}

        # Determine which customers each DC serves (same assignment the VRP uses)
        # so the mfg→DC LP enforces per-DC flow balance, matching MILP constraints.
        from rl.solvers.ortools_vrp import _assign_customers_to_dcs
        dc_customers = _assign_customers_to_dcs(
            open_dcs, sorted(float_demands.keys()), transport_costs
        )
        demand_per_dc = {
            dc_id: sum(float_demands[c] for c in customers)
            for dc_id, customers in dc_customers.items()
        }

        mfg_to_dc_cost = self._solve_mfg_to_dc_lp(open_dcs, demand_per_dc)
        dc_to_cust_cost = self._solver.solve(
            open_dc_ids=open_dcs,
            demands=float_demands,
            transport_cost_d_to_c=transport_costs,
            n_vehicles_per_dc=self._n_vehicles_per_dc,
        )
        return -(dc_cost + mfg_to_dc_cost + dc_to_cust_cost)

    def _solve_mfg_to_dc_lp(
        self, open_dcs: list[int], demand_per_dc: dict[int, float]
    ) -> float:
        """LP for manufacturing→DC flow cost given per-DC demand requirements.

        Args:
            open_dcs: Open DC indices.
            demand_per_dc: dc_id -> total demand that DC must receive (from customer assignment).

        Minimises sum(transport_cost_m_to_d[m][d] * flow[m][d]) subject to:
          - Each manufacturing site stays within capacity
          - Each open DC receives exactly its assigned customer demand
        Returns 1e6 if infeasible.
        """
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return 1e6

        mf_ids = list(self.data.manufacturing_sites.keys())

        x = {
            (m, d): solver.NumVar(0, solver.infinity(), f"x_{m}_{d}")
            for m in mf_ids for d in open_dcs
        }

        for m in mf_ids:
            solver.Add(
                sum(x[m, d] for d in open_dcs) <= self.data.manufacturing_sites[m].capacity
            )
        # Per-DC flow balance: each DC receives exactly its assigned customer demand
        for d in open_dcs:
            solver.Add(sum(x[m, d] for m in mf_ids) == demand_per_dc.get(d, 0.0))

        obj = solver.Objective()
        for m in mf_ids:
            for d in open_dcs:
                obj.SetCoefficient(
                    x[m, d], self.data.manufacturing_sites[m].transport_cost_m_to_d[d]
                )
        obj.SetMinimization()

        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return solver.Objective().Value()
        return 1e6
