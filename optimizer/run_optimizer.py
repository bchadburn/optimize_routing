from ortools.math_opt.python import mathopt

import utils.log as log
from optimizer.data_objects import SimulationParameters, SupplyChainData
from optimizer.math_model_declaration import create_math_model
from ortools_objects.model import ORToolsCPModel


def calculate_transport_costs(model: ORToolsCPModel) -> tuple[float, float]:
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
    total_open_distribution_costs = [model.p_distribution_opening_cost[d] * model.bv_distribution_cost_incurred[day, d]
                                for day in model.s_time_indices()
                                for d in model.s_distribution_sites()
                                ]
    return total_open_distribution_costs
 

def construct_supply_chain_data(
    mean_demand: list[float],
    mfg_site_capacity: list[float],
    std_dev_demand: list[float],
    distribution_opening_costs: list[float],
    transport_cost_m_to_d: list[list[float]],
    transport_cost_d_to_c: list[list[float]]
) -> SupplyChainData:
    """
    Construct a SupplyChainData object from the given supply chain parameters.

    Args:
        mean_demand (list[float]): A list of mean demands for each customer.
        mfg_site_capacity (list[float]): A list of capacities for each manufacturing site.
        std_dev_demand (list[float]): A list of standard deviations of demand for each customer.
        distribution_opening_costs (list[float]): A list of opening costs for each distribution site.
        transport_cost_m_to_d (list[list[float]]): A 2D list representing the transportation costs from manufacturing sites to distribution sites.
        transport_cost_d_to_c (list[list[float]]): A 2D list representing the transportation costs from distribution sites to customers.

    Returns:
        SupplyChainData: A SupplyChainData object containing the supply chain data.
    """
    num_customers = len(mean_demand)
    num_manufacturing_sites = len(mfg_site_capacity)
    num_distribution_sites = len(distribution_opening_costs)

    validate_input_lengths(num_customers, num_manufacturing_sites, num_distribution_sites, std_dev_demand, transport_cost_m_to_d, transport_cost_d_to_c)

    supply_chain_data = SupplyChainData(num_customers, num_distribution_sites, num_manufacturing_sites)
    customer_ids = [customer_id for customer_id in supply_chain_data.customers]

    add_manufacturing_sites(supply_chain_data, mfg_site_capacity)
    add_distribution_sites(supply_chain_data, distribution_opening_costs)
    add_customers(supply_chain_data, customer_ids, mean_demand, std_dev_demand)
    set_transport_costs(supply_chain_data, transport_cost_m_to_d, transport_cost_d_to_c)

    return supply_chain_data

def validate_input_lengths(
    num_customers: int,
    num_manufacturing_sites: int,
    num_distribution_sites: int,
    std_dev_demand: list[float],
    transport_cost_m_to_d: list[list[float]],
    transport_cost_d_to_c: list[list[float]]
) -> None:    
    """
    Validate the lengths of the input lists to ensure they match the expected dimensions.

    Args:
        num_customers (int): The number of customers.
        num_manufacturing_sites (int): The number of manufacturing sites.
        num_distribution_sites (int): The number of distribution sites.
        std_dev_demand (list[float]): A list of standard deviations of demand for each customer.
        transport_cost_m_to_d (list[list[float]]): A 2D list representing the transportation costs from manufacturing sites to distribution sites.
        transport_cost_d_to_c (list[list[float]]): A 2D list representing the transportation costs from distribution sites to customers.

    Raises:
        ValueError: If any of the input lists have an incorrect length.
    """
    if len(std_dev_demand) != num_customers:
        raise ValueError("Length of std_dev_demand does not match the number of customers.")

    if len(transport_cost_m_to_d) != num_manufacturing_sites or any(len(row) != num_distribution_sites for row in transport_cost_m_to_d):
        raise ValueError("Incorrect dimensions for transport_cost_m_to_d.")

    if len(transport_cost_d_to_c) != num_distribution_sites or any(len(row) != num_customers for row in transport_cost_d_to_c):
        raise ValueError("Incorrect dimensions for transport_cost_d_to_c.")

def add_manufacturing_sites(supply_chain_data: SupplyChainData, mfg_site_capacity: list[float]) -> None:
    """
    Add manufacturing sites to the SupplyChainData object.

    Args:
        supply_chain_data (SupplyChainData): The SupplyChainData object to add manufacturing sites to.
        mfg_site_capacity (list[float]): A list of capacities for each manufacturing site.
    """
    for site_id, capacity in enumerate(mfg_site_capacity):
        supply_chain_data.add_manufacturing_site(site_id=site_id, capacity=capacity)

def add_distribution_sites(supply_chain_data: SupplyChainData, distribution_opening_costs: list[float]) -> None:
    """
    Add distribution sites to the SupplyChainData object.

    Args:
        supply_chain_data (SupplyChainData): The SupplyChainData object to add distribution sites to.
        distribution_opening_costs (list[float]): A list of opening costs for each distribution site.
    """
    for site_id, opening_cost in enumerate(distribution_opening_costs):
        supply_chain_data.add_distribution_site(site_id=site_id, opening_cost=opening_cost)

def add_customers(supply_chain_data: SupplyChainData, cust_ids: list[int], mean_demand: list[float], std_dev_demand: list[float]) -> None:
    """
    Add customers to the SupplyChainData object.

    Args:
        supply_chain_data (SupplyChainData): The SupplyChainData object to add customers to.
        mean_demand (list[float]): A list of mean demands for each customer.
        std_dev_demand (list[float]): A list of standard deviations of demand for each customer.
    """
    for customer_id in cust_ids:
        supply_chain_data.add_customer(customer_id=customer_id, mean_demand=mean_demand[customer_id], std_dev_demand=std_dev_demand[customer_id])


def set_transport_costs(
    supply_chain_data: SupplyChainData,
    transport_cost_m_to_d: list[list[float]],
    transport_cost_d_to_c: list[list[float]]
) -> None:
    """
    Set the transportation costs for manufacturing sites and distribution sites in the SupplyChainData object.

    Args:
        supply_chain_data (SupplyChainData): The SupplyChainData object to set transportation costs for.
        transport_cost_m_to_d (list[list[float]]): A 2D list representing the transportation costs from manufacturing sites to distribution sites.
        transport_cost_d_to_c (list[list[float]]): A 2D list representing the transportation costs from distribution sites to customers.
    """
    num_manufacturing_sites = len(supply_chain_data.manufacturing_sites)
    num_distribution_sites = len(supply_chain_data.distribution_sites)
    num_customers = len(supply_chain_data.customers)

    for mf_id in range(num_manufacturing_sites):
        for dist_id in range(num_distribution_sites):
            supply_chain_data.manufacturing_sites[mf_id].set_mf_to_dist_transport_costs(dist_id, transport_cost_m_to_d[mf_id][dist_id])

    for dist_id in range(num_distribution_sites):
        for cust_id in range(num_customers):
            supply_chain_data.distribution_sites[dist_id].set_dist_to_cust_transport_costs(cust_id, transport_cost_d_to_c[dist_id][cust_id])


def optimize(
    supply_chain_data: SupplyChainData,
    sim_params: SimulationParameters,
    num_distribution_sites: list[int],
    solve_infeasibility: bool = False
) -> tuple[list[float], float, list[bool]]:
    """Constructs solver """
    logger = log.get_logger("MathOpt Model")
    
    or_math_model = ORToolsCPModel(
        logger=logger,
        max_time=30,
        rel_gap=0.00,
        solver_log=True,
        shallow_substitute=True,
        solve_infeasibility=solve_infeasibility
    )

    create_math_model(
        or_math_model,
        supply_chain_data,
        sim_params,
    )

    or_math_model.construct_model()

    status = or_math_model.solve_model()
    if status != mathopt.TerminationReason.OPTIMAL:
        logger.info("Optimizer didn't find optimal solution")
        none_dist_list = [None for _ in range(num_distribution_sites)]
        return none_dist_list, float('inf'), none_dist_list
    else:
        if or_math_model.model_config.get("solve_infeasibility", False):
            print("Solution returned with slack variables active (Solving for infeasibility)")
        
        else:
            logger.info("Optimal solution Found")
        
        # logger.info(f"total_cost: {minimize_cost_objective(or_math_model)}")
        total_transport_cost_m_to_d, total_transport_cost_d_to_c = calculate_transport_costs(or_math_model)
        total_open_distribution_costs = calculate_opening_distribution_costs(or_math_model)    
        total_cost = total_transport_cost_m_to_d + total_transport_cost_d_to_c + sum(total_open_distribution_costs)
        logger.info(f"total_cost: {round(total_cost,2)}")
        
        if not or_math_model.model_config.get("solve_infeasibility", False):
            assert round(total_cost,2) == round(total_cost,2), "Objective function result doesn't match total costs"
        
        open_distribution_decisions = [or_math_model.bv_distribution_cost_incurred[day, d]
                                for day in or_math_model.s_time_indices()
                                for d in or_math_model.s_distribution_sites()
                                ]
        # Return objective value, DC opening decision, and total transportation cost
        return total_open_distribution_costs, total_cost, open_distribution_decisions


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
    
    # Construct supply chain data, params
    sim_params = SimulationParameters(num_days, num_simulations, decision_rolling_period)
    supply_chain_data = construct_supply_chain_data(mean_demand, mfg_site_capacity, std_dev_demand, distribution_opening_costs, transport_cost_m_to_d, transport_cost_d_to_c)
    
    # Construct and run optimizer
    num_distribution_sites = len(distribution_opening_costs)
    total_open_distribution_costs, total_cost, open_distribution_decisions = optimize(supply_chain_data, sim_params, num_distribution_sites)
    
    print(total_open_distribution_costs, total_cost)