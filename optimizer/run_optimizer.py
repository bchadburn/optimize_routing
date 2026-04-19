import logging
import numpy as np

from optimizer.construct_data_objects import SupplyChainData, SimulationParameters


def build_supply_chain_data(
    distribution_opening_costs: list[float],
    mfg_site_capacity: list[float],
    mean_demand: list[float],
    std_dev_demand: list[float],
    transport_cost_m_to_d: list[list[float]],
    transport_cost_d_to_c: list[list[float]],
) -> SupplyChainData:
    """Build a SupplyChainData object from raw parameter lists."""
    data = SupplyChainData()
    for mf_id, cap in enumerate(mfg_site_capacity):
        data.add_manufacturing_site(site_id=mf_id, capacity=cap)
    for dist_id, cost in enumerate(distribution_opening_costs):
        data.add_distribution_site(site_id=dist_id, opening_cost=cost)
    for cust_id, (mean, std) in enumerate(zip(mean_demand, std_dev_demand)):
        data.add_customer(customer_id=cust_id, mean_demand=mean, std_dev_demand=std)
    for mf_id in range(len(mfg_site_capacity)):
        for dist_id in range(len(distribution_opening_costs)):
            data.manufacturing_sites[mf_id].set_mf_to_dist_transport_costs(
                dist_id, transport_cost_m_to_d[mf_id][dist_id]
            )
    for dist_id in range(len(distribution_opening_costs)):
        for cust_id in range(len(mean_demand)):
            data.distribution_sites[dist_id].set_dist_to_cust_transport_costs(
                cust_id, transport_cost_d_to_c[dist_id][cust_id]
            )
    return data


def _make_model(logger: logging.Logger):
    from ortools_objects.model import ORToolsCPModel
    return ORToolsCPModel(
        logger=logger,
        max_time=30,
        rel_gap=0.00,
        solver_log=False,
        shallow_substitute=True,
    )


def run_global_milp(
    supply_chain_data: SupplyChainData,
    num_days: int = 10,
    decision_rolling_period: int = 3,
) -> dict:
    """Run a single global MILP solve over the full horizon.

    Returns dict with keys: total_cost, dc_decisions, transport_cost_m_to_d, transport_cost_d_to_c.
    """
    import utils.log as log
    from ortools.linear_solver import pywraplp
    from optimizer.math_model_declaration import create_math_model
    from optimizer.math_model_constraints import minimize_cost_objective

    logger = log.get_logger("MILP-Global")
    sim_params = SimulationParameters(num_days, 1, decision_rolling_period)
    model = _make_model(logger)
    create_math_model(model, supply_chain_data, sim_params)
    model.construct_model()
    status = model.solve_model()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Global MILP: no feasible solution found")

    total_cost = minimize_cost_objective(model).solution_value()
    dc_decisions = [
        {d for d in model.s_distribution_sites() if model.bv_distribution_on[day, d].solution_value() > 0.5}
        for day in model.s_time_indices()
    ]
    transport_m_to_d = sum(
        model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d[d, day, m].solution_value()
        for d in model.s_distribution_sites()
        for day in model.s_time_indices()
        for m in model.s_manufacturing_sites()
    )
    transport_d_to_c = sum(
        model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c[d, day, c].solution_value()
        for d in model.s_distribution_sites()
        for day in model.s_time_indices()
        for c in model.s_customers()
    )
    return {
        "total_cost": total_cost,
        "dc_decisions": dc_decisions,
        "transport_cost_m_to_d": transport_m_to_d,
        "transport_cost_d_to_c": transport_d_to_c,
    }


def run_daily_myopic(
    supply_chain_data: SupplyChainData,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    num_simulations: int = 10,
) -> dict:
    """Run daily myopic MILP solves (re-solve each day with fresh demand sample).

    Returns dict with keys: mean_total_cost, std_total_cost, costs_per_simulation.
    """
    import utils.log as log
    from ortools.linear_solver import pywraplp
    from optimizer.math_model_declaration import create_math_model
    from optimizer.math_model_constraints import minimize_cost_objective

    logger = log.get_logger("MILP-Daily")
    costs = []
    for _ in range(num_simulations):
        sim_params = SimulationParameters(1, 1, decision_rolling_period)
        total_cost = 0.0
        for _day in range(num_days):
            model = _make_model(logger)
            create_math_model(model, supply_chain_data, sim_params)
            model.construct_model()
            status = model.solve_model()
            if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                total_cost += 1e6
            else:
                total_cost += minimize_cost_objective(model).solution_value()
        costs.append(total_cost)
    return {
        "mean_total_cost": float(np.mean(costs)),
        "std_total_cost": float(np.std(costs)),
        "costs_per_simulation": costs,
    }
