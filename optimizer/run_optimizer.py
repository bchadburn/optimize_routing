import utils.log as log
from typing import List
from ortools.linear_solver import pywraplp
from optimizer.math_model_declaration import create_math_model
from optimizer.construct_data_objects import SupplyChainData, SimulationParameters
from ortools_objects.cost_objects import PowerUse
from ortools_objects.model import ORToolsCPModel
from optimizer.math_model_constraints import minimize_cost_objective

def calculate_transport_costs(model: ORToolsCPModel) -> List[float]:
    # Calculate transportation costs
    total_transport_cost_m_to_d = sum([model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d[d, day, m]
                            for d in model.s_distribution_sites()
                            for day in model.s_time_indices()
                            for m in model.s_manufacturing_sites()])

    total_transport_cost_d_to_c = sum([model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c[d, day, c]
                            for d in model.s_distribution_sites()
                            for day in model.s_time_indices()
                            for c in model.s_customers()])
    return total_transport_cost_m_to_d, total_transport_cost_d_to_c
    
def calculate_opening_distribution_costs(model: ORToolsCPModel) -> list:
    opening_distribution_cost = [model.p_distribution_opening_cost[d] * model.bv_distribution_cost_incurred[day, d]
                                for day in model.s_time_indices()
                                for d in model.s_distribution_sites()
                                ]
    return opening_distribution_cost

def optimize(distribution_opening_costs: list, mfg_site_capacity: list, mean_demand: list, std_dev_demand: list, transport_cost_m_to_d: list, transport_cost_d_to_c: list,num_days: int=10, num_simulations: int=10, decision_rolling_period: int=3) -> None:
    logger = log.get_logger("SCIP Solver")
    
    num_customers = len(mean_demand)
    num_distribution_sites = len(distribution_opening_costs)
    num_manufacturing_sites = len(mfg_site_capacity)
    
    # Create class to organize supply chain data
    supply_chain_data = SupplyChainData()
    sim_params = SimulationParameters(num_days, num_simulations, decision_rolling_period)
    
    # Adding manufacturing sites. Simply assigning ids based on idx
    mf_ids = [mf_id for mf_id in range(num_manufacturing_sites)]
    for dist_id in mf_ids:
        supply_chain_data.add_manufacturing_site(site_id=dist_id, capacity=mfg_site_capacity[dist_id])

    # Adding distribution sites
    dist_ids = [dist_ids for dist_ids in range(num_distribution_sites)]
    for dist_id in dist_ids:   
        supply_chain_data.add_distribution_site(site_id=dist_id, opening_cost=distribution_opening_costs[dist_id])

    # Adding customers
    cust_ids = [cust_id for cust_id in range(num_customers)]
    for customer_id in cust_ids:
        supply_chain_data.add_customer(customer_id=customer_id, mean_demand=mean_demand[customer_id], std_dev_demand=std_dev_demand[customer_id])

    # Set transport costs for manufacturing sites
    for mf_id in range(num_manufacturing_sites):
        for dist_id in range(num_distribution_sites):
            supply_chain_data.manufacturing_sites[mf_id].set_mf_to_dist_transport_costs(dist_id, transport_cost_m_to_d[mf_id][dist_id])
            
    # Set transport costs for distribution sites
    for dist_id in range(num_distribution_sites):
        for cust_id in range(num_customers):
            supply_chain_data.distribution_sites[dist_id].set_dist_to_cust_transport_costs(cust_id, transport_cost_d_to_c[dist_id][cust_id])

    or_math_model = ORToolsCPModel(
        logger=logger,
        max_time=30,
        rel_gap=0.00,
        solver_log=True,
        shallow_substitute=True,
    )

    create_math_model(
        or_math_model,
        supply_chain_data,
        sim_params,
    )

    or_math_model.construct_model()

    status = or_math_model.solve_model()
    if status != pywraplp.Solver.OPTIMAL:
        logger.info("Optimizer didn't find optimal solution")
        return [None for _ in range(num_distribution_sites) for _ in range(num_days)],float('inf')
    else:
        logger.info(f"total_cost: {minimize_cost_objective(or_math_model)}")
        logger.info("Optimal solution Found")
        
        total_transport_cost_m_to_d, total_transport_cost_d_to_c = calculate_transport_costs(or_math_model)
        opening_distribution_costs = calculate_opening_distribution_costs(or_math_model)    
        total_transport_cost = total_transport_cost_m_to_d + total_transport_cost_d_to_c + sum(opening_distribution_costs)
        
        # Return objective value, DC opening decision, and total transportation cost
        return opening_distribution_costs, total_transport_cost    


if __name__ == "__main__":
    num_days=10
    num_simulations=10 
    decision_rolling_period=3
    
    # distribution_opening_costs = [350, 320, 375, 400, 550]
    distribution_opening_costs = [1000, 100000, 100000, 100000, 100000]
    mfg_site_capacity = [600000, 600000]

    # mean_demand = [20, 30, 25, 40, 35, 28, 32, 50, 26, 38, 34, 27]
    mean_demand = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    # std_dev_demand = [20, 18, 15, 20, 20, 5, 5, 12.4, 12.6, 13.8, 13.4, 12.7]
    std_dev_demand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Transportation costs
    # transport_cost_m_to_d = [
    #     [3.5, 2.5, 4.5, 2.5, 3.0],  # Manufacturing site 1
    #     [2.5, 4.5, 5.5, 6.5, 8.5]  # Manufacturing site 2
    # ]
    # transport_cost_d_to_c = [
    #     [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],  # Distribution site 1
    #     [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],  # Distribution site 2
    #     [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],  # Distribution site 3
    #     [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2],  # Distribution site 4
    #     [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1]   # Distribution site 5
    # ]

    transport_cost_m_to_d = [
        [3, 3, 3, 3, 3],  # Manufacturing site 1
        [3, 3, 3, 3, 3]  # Manufacturing site 2
    ]
    transport_cost_d_to_c = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Distribution site 1
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Distribution site 2
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Distribution site 3
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Distribution site 4
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # Distribution site 5
    ]

    opening_distribution_costs, total_transport_cost = optimize(distribution_opening_costs, mfg_site_capacity, mean_demand, std_dev_demand, transport_cost_m_to_d, transport_cost_d_to_c,
             num_days=num_days, num_simulations=num_simulations, decision_rolling_period=decision_rolling_period)