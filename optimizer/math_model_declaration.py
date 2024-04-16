import numpy as np

import optimizer.math_model_constraints as constraint
from optimizer.construct_data_objects import SimulationParameters, SupplyChainData
from ortools_objects.constraint import IndexedORStandardConst
from ortools_objects.model import ORToolsCPModel
from ortools_objects.objective import ORObjective
from ortools_objects.param import IndexedORParam, ScalarORParam
from ortools_objects.set import ORSet
from ortools_objects.var import IndexedORBoolVariable, IndexedORContinuousVariable


def _add_base_model_sets(
    model: ORToolsCPModel, supply_chain_data: SupplyChainData, sim_params: SimulationParameters
) -> None:
    """Add base sets to math model. Any specialized sets should be added via specialized
    implementations using multi-method.

    Args:
        model (ORToolsCPModel): Model containing sets, parameters, variables, constraints
        supply_chain_data (SupplyChainData): Supply chain data such as distribution, manufacturing, and customer data
        sim_params (SimulationParameters): Simulation meta-parameters such as number of simulations or days to simulate. 
    """
    
    model.s_time_indices = ORSet(
        name="time_index",
        doc="List of time indices for model",
        initialize=[time_idx for time_idx in range(sim_params.num_days)],
    )
    
    model.s_manufacturing_sites = ORSet(
        name="s_manufacturing_sites",
        doc="List of manufacturing sites",
        initialize=list(supply_chain_data.manufacturing_sites.keys()),
    )
    
    model.s_distribution_sites = ORSet(
        name="s_distribution_sites", 
        doc="List of distribution sites", 
        initialize=list(supply_chain_data.distribution_sites.keys())
    )
        
    model.s_customers = ORSet(
        name="s_customers", 
        doc="List of s_customers sites", 
        initialize=list(supply_chain_data.customers.keys())
    )
    

def _add_base_model_parameters(
    model: ORToolsCPModel,
    supply_chain_data: SupplyChainData,
    sim_params: SimulationParameters
) -> None:
    
    """
    Add fixed values for the optimization process. They can be scalers, but are more often indexed over sets. 
    To achieve this, we create the sets first and then assign values to all instances within those sets. 

    Args:
        model (ORToolsCPModel): Model object to which parameters should be added.
        schedule_list_dict (List[Dict]): List of dictionaries (one for each time period) with data about each period
        supply_chain_data (SupplyChainData): Supply chain data such as distribution, manufacturing, and customer data
    """
    # Create parameters
    model.p_manufacturing_site_capacity = IndexedORParam(model.s_manufacturing_sites, name="p_manufacturing_site_capacity",
        doc="Avg daily demand from customers", initialize={
            manufacturing_site: supply_chain_data.manufacturing_sites[manufacturing_site].capacity
            for manufacturing_site in model.s_manufacturing_sites()
        }
    )
        
    model.p_big_m = ScalarORParam(
        name="big_m", doc="Big M value", initialize=10000
    )

    model.p_decision_rolling_period = ScalarORParam(
        name="p_distribution_site_rolling_period", doc="Number of days distribution site must remain open once opened", initialize=sim_params.decision_rolling_period
    ) 
    
    model.p_distribution_opening_cost = IndexedORParam(model.s_distribution_sites, name="p_distribution_opening_cost",
        doc="Cost of opening distribution", initialize={
            distribution_site: supply_chain_data.distribution_sites[distribution_site].opening_cost
            for distribution_site in model.s_distribution_sites()
        }
    )
 
    model.p_transport_cost_m_to_d = IndexedORParam(model.s_manufacturing_sites, model.s_distribution_sites, name="p_transport_cost_m_to_d",
        doc="Transportation cost from manufacturing site to distribution site", initialize={
            (manufacturing_site, distribution_site): supply_chain_data.manufacturing_sites[manufacturing_site].transport_cost_m_to_d[distribution_site]
            for distribution_site in model.s_distribution_sites()
            for manufacturing_site in model.s_manufacturing_sites()
        }
    )       
       
    model.p_transport_cost_d_to_c = IndexedORParam(model.s_distribution_sites, model.s_customers, name="p_transport_cost_d_to_c",
        doc="Transportation cost from distribution site to customer", initialize={
            (distribution_site, customer): supply_chain_data.distribution_sites[distribution_site].transport_cost_d_to_c[customer]
            for distribution_site in model.s_distribution_sites()
            for customer in model.s_customers()
        }
    )    
    
    
    # Simulate by sampling from customer demand distribution
    model.p_customer_demand = IndexedORParam(
        model.s_time_indices,
        model.s_customers,
        name="s_mean_demand",
        doc="Avg daily demand from customers",
        initialize={
            (time_idx, cust_idx): max(0, np.random.normal(supply_chain_data.customers[cust_idx].mean_demand, supply_chain_data.customers[cust_idx].std_dev_demand)) 
            for time_idx in model.s_time_indices()
            for cust_idx in model.s_customers()
        },
    )
    
    # Log vals such as computed vals
    # model.p_example_computed_val.log_parameter_values(model.logger)


def _add_base_model_variables(model: ORToolsCPModel) -> None:
    """Adds variables to the model including Boolean (0 or 1) or continuous variables 
    with specified lower and upper bounds, or default bounds if not provided.

    Args:
        model (ORToolsCPModel): Model object containing all model-specific objects for the solve.
    """
    model.bv_distribution_cost_incurred = IndexedORBoolVariable(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_cost_incurred",
        doc="Boolean indicating distribution was turned on (indicating a cost was incurred) during a specific time step",
    ) 
    
    model.bv_distribution_on = IndexedORBoolVariable(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_on",
        doc="Whether or not a certain distribution site is on (either turned on or is remainin on) during a specific time step",
    )

    model.v_transport_m_to_d = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="transport_m_to_d",
        doc="Number of shipments from manufacturing sites to distribution sites",
        log_solution=True,
    )

    model.v_transport_d_to_c = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="transport_d_to_c",
        doc="Number of shipments from distribution sites to customers",
        log_solution=True,
    )

def _add_slack_variables(model: ORToolsCPModel) -> None:
    
    model.v_transport_m_to_d_shipments_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="ransport_m_to_d_shipments_slack",
        log_solution=True,
    )
    
    model.v_transport_m_to_d_capacity_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="transport_m_to_d_capacity_slack",
        log_solution=True,
    )
    
    model.v_transport_d_to_c_shipments_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="transport_d_to_c_shipments_slack",
        log_solution=True,
    )    

    model.v_transport_d_to_c_shipments_equal_demand_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="transport_d_to_c_shipments_equal_demand_slack",
        log_solution=True,
    )
    
    model.v_transport_d_to_c_demand_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="transport_d_to_c_demand_slack",
        log_solution=True,
    )
    
    
def _add_base_model_constraints(model: ORToolsCPModel) -> None:
    """Adds model constraints by invoking functions defined in the constraints file. 
    Functions are associated with specific elements of an index. To skip an index, 
    use if-else statement with pass.

    Args:
        model (ORToolsCPModel): Model containing all sets, parameters, variables, and constraints.
    """
   
    model.c_distribution_cost_incurred_boolean_rolling_constraint = IndexedORStandardConst(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_cost_incurred_constraint",
        doc="Forces boolean indicating a cost was incurred for opening distribution site to be triggered only once within a single rolling window",
        rule=constraint.distribution_cost_boolean_incurred_rolling_constraint,
    )  
    
    model.c_distribution_opening_duration_rolling_window_constraint = IndexedORStandardConst(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_open_constraint",
        doc="Forces distribution to remain open once opened for a set number of days",
        rule=constraint.distribution_site_open_duration_rolling_constraint,
    )
      
    model.c_distribution_status_constrained_by_d_to_c_supply = IndexedORStandardConst(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="distribution_status_by_d_to_c_supply",
        doc="Ensure DC status coincides with distribution supply",
        rule=constraint.distribution_status_constrained_by_d_to_c_supply
    )

    model.c_distribution_status_constrained_by_m_to_d_supply = IndexedORStandardConst(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="distribution_status_constrained_by_m_to_d_supply",
        doc="Ensure DC status coincides with manufacturing supply",
        rule=constraint.distribution_status_constrained_by_m_to_d_supply,
    )
    
    model.c_manufacturing_supply_equal_capacity = IndexedORStandardConst(
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="manufacturing_supply_equal_capacity",
        doc="Ensure manufacturing supply, i.e. the number of products shipped to all distribution sites, is equal to supply capacity supported by the manufacturing site",
        rule=constraint.distribution_supply_equal_capacity
    )
        
    model.c_distribution_shipments_equal_customer_demand = IndexedORStandardConst(
        model.s_time_indices,
        model.s_customers,
        name="distribution_shipments_equal_customer_demand",
        doc="Ensure distribution shipments is equal to the customer demand",
        rule=constraint.distribution_shipments_equal_customer_demand
    )
    
    model.c_distribution_shipments_equal_total_received_shipments = IndexedORStandardConst(
        model.s_time_indices,
        model.s_distribution_sites, 
        name="distribution_shipments_equal_total_received_shipments",
        doc="Ensure distribution shipments is equal to received shipments",
        rule=constraint.distribution_shipments_equal_total_received_shipments
    )

def _add_model_objective(model: ORToolsCPModel):
    model.objective = ORObjective(
        name="minimum_cost_objective",
        doc="Minimum cost expression generated by sympy passed via model config",
        rule=constraint.minimize_cost_objective,
    )


def create_math_model(
    model: ORToolsCPModel,
    supply_chain_data: SupplyChainData,
    sim_params: SimulationParameters,
) -> None:
    """Creates math model for a specific model object list type. Create a base set of sets, parameters, variables, and constraints. However, the objective differs
    often times between sites, and there are specific constraints that need to be added for different product types.

    Args:
        model (ORToolsCPModel): Math model object created in original function. Attributes will be added to this one
        supply_chain_data (SupplyChainData): Supply chain data such as distribution, manufacturing, and customer data
        sim_params (SimulationParameters): Simulation meta-parameters such as number of simulations or days to simulate. 
    """
    _add_base_model_sets(model, supply_chain_data, sim_params)
    _add_base_model_parameters(
        model,
        supply_chain_data,
        sim_params
    )
    _add_base_model_variables(model)
    _add_base_model_constraints(model)
    
    if model.model_config.get("solve_infeasibility", False):
        _add_slack_variables(model)
        
    _add_model_objective(model)
