"""SupplyChainEnv subclass that accepts any CvrptwSolver for the routing leg.

The existing _solve_routing_lp is replaced with solver.solve(). All state
management, rolling-window enforcement, and episode logic are inherited unchanged.
"""
from __future__ import annotations

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
        routing_cost = self._solver.solve(
            open_dc_ids=open_dcs,
            demands={cid: float(qty) for cid, qty in demands.items()},
            transport_cost_d_to_c=transport_costs,
            n_vehicles_per_dc=self._n_vehicles_per_dc,
        )
        return -(dc_cost + routing_cost)
