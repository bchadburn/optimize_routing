import numpy as np

from optimizer.data_objects import SimulationParameters, SupplyChainData
from ortools_objects.model import ORToolsCPModel
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
        doc="list of time indices for model",
        initialize=[time_idx for time_idx in range(sim_params.num_days)],
    )
    
    model.s_manufacturing_sites = ORSet(
        name="s_manufacturing_sites",
        doc="list of manufacturing sites",
        initialize=list(supply_chain_data.manufacturing_sites.keys()),
    )
    
    model.s_distribution_sites = ORSet(
        name="s_distribution_sites", 
        doc="list of distribution sites", 
        initialize=list(supply_chain_data.distribution_sites.keys())
    )
        
    model.s_customers = ORSet(
        name="s_customers", 
        doc="list of s_customers sites", 
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
        doc="avg daily demand from customers", initialize={
            manufacturing_site: supply_chain_data.manufacturing_sites[manufacturing_site].capacity
            for manufacturing_site in model.s_manufacturing_sites()
        }
    )
        
    model.p_big_m = ScalarORParam(
        name="big_m", doc="big M value", initialize=10000
    )

    model.p_decision_rolling_period = ScalarORParam(
        name="p_distribution_site_rolling_period", doc="number of days distribution site must remain open once opened", initialize=sim_params.decision_rolling_period
    ) 
    
    model.p_distribution_opening_cost = IndexedORParam(model.s_distribution_sites, name="p_distribution_opening_cost",
        doc="cost of opening distribution", initialize={
            distribution_site: supply_chain_data.distribution_sites[distribution_site].opening_cost
            for distribution_site in model.s_distribution_sites()
        }
    )
 
    model.p_transport_cost_m_to_d = IndexedORParam(model.s_manufacturing_sites, model.s_distribution_sites, name="p_transport_cost_m_to_d",
        doc="transportation cost from manufacturing site to distribution site", initialize={
            (manufacturing_site, distribution_site): supply_chain_data.manufacturing_sites[manufacturing_site].transport_cost_m_to_d[distribution_site]
            for distribution_site in model.s_distribution_sites()
            for manufacturing_site in model.s_manufacturing_sites()
        }
    )       
       
    model.p_transport_cost_d_to_c = IndexedORParam(model.s_distribution_sites, model.s_customers, name="p_transport_cost_d_to_c",
        doc="transportation cost from distribution site to customer", initialize={
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
        doc="avg daily demand from customers",
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
        doc="boolean indicating distribution was turned on (indicating a cost was incurred) during a specific time step",
    ) 
    
    model.bv_distribution_on = IndexedORBoolVariable(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_on",
        doc="whether or not a certain distribution site is on (either turned on or is remaining on) during a specific time step",
    )

    model.v_transport_m_to_d = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="transport_m_to_d",
        doc="number of shipments from manufacturing sites to distribution sites",
        log_solution=True,
    )

    model.v_transport_d_to_c = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="transport_d_to_c",
        doc="number of shipments from distribution sites to customers",
        log_solution=True,
    )

def _add_slack_variables(model: ORToolsCPModel) -> None:
    
    model.v_transport_m_to_d_shipments_slack = IndexedORContinuousVariable(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="transport_m_to_d_shipments_slack",
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
    """Adds constraints, or the "subject to" part of the math model. Each
    of these call a function defined in the constraints file. For more information on those,
    see the file referenced by the "constraint"

    Args:
        model (ORToolsCPModel): Model containing all sets, parameters, variables, and constraints.
    """
    
    @model.IndexedORStandardConst(
       model.s_time_indices,
       model.s_distribution_sites,
       name="distribution_cost_incurred_constraint",
       doc="forces boolean indicating a cost was incurred for opening distribution site to be triggered only once within a single rolling window"
    )  
    def distribution_cost_boolean_incurred_rolling_constraint(model, time_period, distribution_site):
        """Rolling Constraint to ensure the cost of turning on distribution site is only assigned once within a set amount of time (decision_rolling_period). 
        The distribution_site_open_duration_rolling_constraint will then be used to ensure the site remains open for set amount of time."""
        open_duration = model.p_decision_rolling_period[None]
        min_open_day = max(0, time_period-open_duration+1)
        return sum([model.bv_distribution_cost_incurred[duration_day, distribution_site] 
                            for duration_day in range(min_open_day, time_period+1)]) <= 1
    
    @model.IndexedORStandardConst(
        model.s_time_indices,
        model.s_distribution_sites,
        name="distribution_open_constraint",
        doc="forces distribution to remain open once opened for a set number of days",
    )
    def distribution_site_open_duration_rolling_constraint(model, time_period, distribution_site):
        """Ensure the site remains open after distribution site was initially opened (i.e. time idx cost was incurred) for set amount of time."""
        open_duration = model.p_decision_rolling_period[None]
        min_open_day = max(0, time_period-open_duration+1)
        bv_cost_incurred_rolling_period = sum([model.bv_distribution_cost_incurred[duration_day, distribution_site]
                                            for duration_day in range(min_open_day, time_period+1)])
        return model.bv_distribution_on[time_period, distribution_site] == bv_cost_incurred_rolling_period 
      
    @model.IndexedORStandardConst(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_customers,
        name="distribution_status_by_d_to_c_supply",
        doc="ensure DC status coincides with distribution supply"
    )
    def distribution_status_constrained_by_d_to_c_supply(model, distribution_site, time_period, customer):
        return model.v_transport_d_to_c[distribution_site, time_period, customer] <= model.bv_distribution_on[time_period, distribution_site] * model.p_big_m[None]

    @model.IndexedORStandardConst(
        model.s_distribution_sites,
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="distribution_status_constrained_by_m_to_d_supply",
        doc="ensure DC status coincides with manufacturing supply"
    )
    def distribution_status_constrained_by_m_to_d_supply(model, distribution_site, time_period, manufacturing_site):
        return (
            model.v_transport_m_to_d[distribution_site, time_period, manufacturing_site] <= model.bv_distribution_on[time_period, distribution_site] * model.p_big_m[None]
        )
    
    @model.IndexedORStandardConst(
        model.s_time_indices,
        model.s_manufacturing_sites,
        name="manufacturing_supply_equal_capacity",
        doc="ensure manufacturing supply, i.e. the number of products shipped to all distribution sites, is equal to supply capacity supported by the manufacturing site"
    )
    def distribution_supply_equal_capacity(model, time_period, manufacturing_site):
        # Supply constraints so total mfg site supply is not more than capacity at a given distribution site  
        total_supply_m = sum([model.v_transport_m_to_d[d, time_period, manufacturing_site] for d in model.s_distribution_sites()])
        slack_var = sum([model.v_transport_m_to_d_capacity_slack[d, time_period, manufacturing_site] for d in model.s_distribution_sites() if model.model_config.get("solve_infeasibility", False)])
        return total_supply_m <= model.p_manufacturing_site_capacity[manufacturing_site] + slack_var
    
    @model.IndexedORStandardConst(
        model.s_time_indices,
        model.s_customers,
        name="distribution_shipments_equal_customer_demand",
        doc="ensure distribution shipments is equal to the customer demand"
    )
    def distribution_shipments_equal_customer_demand(model, time_period, customer):
        # Demand constraints so total shipments is equal to customer demand
        added_shipments_slack_var = sum([model.v_transport_d_to_c_shipments_equal_demand_slack[d, time_period, customer] for d in model.s_distribution_sites() if model.model_config.get("solve_infeasibility", False)])
        added_customer_demand_slack_var = sum([model.v_transport_d_to_c_demand_slack[d, time_period, customer] for d in model.s_distribution_sites() if model.model_config.get("solve_infeasibility", False)])
        return sum([model.v_transport_d_to_c[d, time_period, customer] for d in model.s_distribution_sites()]) + added_shipments_slack_var == model.p_customer_demand[time_period, customer] + added_customer_demand_slack_var

    @model.IndexedORStandardConst(
        model.s_time_indices,
        model.s_distribution_sites, 
        name="distribution_shipments_equal_total_received_shipments",
        doc="ensure distribution shipments is equal to received shipments"
    )
    def distribution_shipments_equal_total_received_shipments(model, time_period, distribution_site):
        # Limit distribution center shipments to the total of received shipments
        total_shipment_from_m_to_d = sum([model.v_transport_m_to_d[distribution_site, time_period, m] for m in model.s_manufacturing_sites()])
        total_shipment_from_d_to_c = sum([model.v_transport_d_to_c[distribution_site, time_period, customer] for customer in model.s_customers()])
        
        shipments_m_to_d_slack_var = sum([model.v_transport_m_to_d_shipments_slack[distribution_site, time_period, m] for m in model.s_manufacturing_sites() if model.model_config.get("solve_infeasibility", False)])
        shipments_d_to_c_slack_var = sum([model.v_transport_d_to_c_shipments_slack[distribution_site, time_period, c] for c in model.s_customers() if model.model_config.get("solve_infeasibility", False)])

        return total_shipment_from_m_to_d + shipments_m_to_d_slack_var == total_shipment_from_d_to_c + shipments_d_to_c_slack_var

def _add_model_objective(model: ORToolsCPModel):
    @model.ORObjective(
        name="minimum_cost_objective",
        doc="minimum cost expression generated by sympy passed via model config"
    )
    def minimize_cost_objective(model):
        if not model.model_config.get("solve_infeasibility", False):
            total_cost = sum([model.p_distribution_opening_cost[d] * model.bv_distribution_cost_incurred[day, d]
                                        for day in model.s_time_indices()
                                        for d in model.s_distribution_sites()
                                        ]) + \
                            sum([model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d[d, day, m]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for m in model.s_manufacturing_sites()]) + \
                            sum([model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c[d, day, c]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for c in model.s_customers()])
        else:
            total_cost = sum([100*model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d_shipments_slack[d, day, m]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for m in model.s_manufacturing_sites()]) + \
                        sum([100*model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c_shipments_slack[d, day, c]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for c in model.s_customers()]) + \
                        sum([200*model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d_capacity_slack[d, day, m]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for m in model.s_manufacturing_sites()]) + \
                        sum([500*model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c_shipments_equal_demand_slack[d, day, c]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for c in model.s_customers()]) + \
                        sum([500*model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c_demand_slack[d, day, c]
                                        for d in model.s_distribution_sites()
                                        for day in model.s_time_indices()
                                        for c in model.s_customers()]) 
        return total_cost

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
        model.logger.info("Slack variables used to solver for infeasibility")
        _add_slack_variables(model)
        
    _add_model_objective(model)





# Example of using gray notation
# def _compute_gray_notation(value: int) -> int:
#     return value ^ (int(value / 2))


# def _add_xval_model_sets....
# ....

#     model.s_bitwise_xval_binary_set = ORSet(
#         name="xval_log_binary_set",
#         doc="Takes advantage of logarithmic encoding style of Gray binary encoding schema to create a set of binary variables for xval",
#         initialize=[
#             (time_index, site, log_index)
#             for time_index in model.s_time_indices
#             for site in model.s_xval_segment_starts
#             for log_index in range(
#                 int(
#                     ceil(
#                         (
#                             log(
#                                 len(
#                                     schedule_list_dict[time_index][
#                                         f"effective_xval_gph_{site}_piecewise"
#                                     ]
#                                 )
#                             )
#                             / log(2)
#                         )
#                     )
#                 )
#             )
#             if f"effective_xval_gph_{site}_piecewise" in schedule_list_dict[time_index]
#         ],
#     )
#     model.s_bitwise_xval_binary_indexed_set = IndexedORSet(
#         model.s_time_indices,
#         model.s_xval_segment_starts,
#         name="xval_log_binary_indexed_set",
#         doc="Takes advantage of logarithmic encoding style of Gray binary encoding schema to create a set of binary variables for xval",
#         initialize={
#             (time_index, site): [
#                 log_index
#                 for log_index in range(
#                     int(
#                         ceil(
#                             (
#                                 log(
#                                     len(
#                                         schedule_list_dict[time_index][
#                                             f"effective_xval_gph_{site}_piecewise"
#                                         ]
#                                     )
#                                 )
#                                 / log(2)
#                             )
#                         )
#                     )
#                 )
#             ]
#             for time_index in model.s_time_indices
#             for site in model.s_xval_segment_starts
#             if f"effective_xval_gph_{site}_piecewise" in schedule_list_dict[time_index]
#         },
#     )
#     model.s_bitwise_binary_zeroes_indexed_set = IndexedORSet(
#         model.s_bitwise_xval_binary_set,
#         name="bitwise_binary_zeroes_indexed_set",
#         doc="For each natural logarithm binary index, store which piecewise points should have the zero value constraint applied to them",
#         initialize={
#             (time_index, site, log_index): (
#                 [0]
#                 + [
#                     point_index
#                     for point_index in range(
#                         1,
#                         len(
#                             schedule_list_dict[time_index][
#                                 f"effective_xval_gph_{site}_piecewise"
#                             ]
#                         )
#                         + 1,
#                     )
#                     if f"effective_xval_gph_{site}_piecewise"
#                     in schedule_list_dict[time_index]
#                     and (1 & _compute_gray_notation(point_index) >> log_index) == 0
#                     and (1 & _compute_gray_notation(point_index - 1) >> log_index) == 0
#                 ]
#             )
#             for time_index, site, log_index in model.s_bitwise_xval_binary_set
#         },
#     )
#     model.s_bitwise_binary_ones_indexed_set = IndexedORSet(
#         model.s_bitwise_xval_binary_set,
#         name="bitwise_binary_ones_indexed_set",
#         doc="For each natural logarithm binary index, store which piecewise points should have the one value constraint applied to them",
#         initialize={
#             (time_index, site, log_index): [
#                 point_index
#                 for point_index in range(
#                     1,
#                     len(
#                         schedule_list_dict[time_index][
#                             f"effective_xval_gph_{site}_piecewise"
#                         ]
#                     )
#                     + 1,
#                 )
#                 if f"effective_xval_gph_{site}_piecewise"
#                 in schedule_list_dict[time_index]
#                 and (1 & _compute_gray_notation(point_index) >> log_index) == 1
#                 and (1 & _compute_gray_notation(point_index - 1) >> log_index) == 1
#             ]
#             for time_index, site, log_index in model.s_bitwise_xval_binary_set
#         },
#     )


# def _add_xval_model_variables(model: ORToolsCPModel, pipeline_name: str):
#     model.v_log_piecewise_active = IndexedORBoolVariable(
#         model.s_bitwise_xval_binary_set,
#         name="piecewise_bool_log_active",
#         doc="Whether or not the bit of the Gray encoding is active for SOS2 logarithmic encoding",
#     )


# def _add_xval_model_constraints(model: ORToolsCPModel):
#     """
#     Adds constraints to the OR-Tools CP model for the xval (xval Reducing Agent) model.

#     Args:
#         model (ORToolsCPModel): The OR-Tools CP model to add the constraints to.

#     Returns:
#         None
#     """
#     @model.IndexedORStandardConst(
#         model.s_bitwise_xval_binary_set,
#         name="binary_bitwise_ones_constraint",
#         doc="Constraint that takes Gray notation with the individual piecewise points to enforce neighboring points to be one",
#     )
#     def binary_bitwise_ones_constraint(model, time_index, site, log_index):
#         return (
#             sum(
#                 model.v_piecewise_point_weights[time_index, site, point_index]
#                 for point_index in model.s_bitwise_binary_ones_indexed_set[
#                     time_index, site, log_index
#                 ]
#             )
#             <= model.v_log_piecewise_active[time_index, site, log_index]
#         )

#     @model.IndexedORStandardConst(
#         model.s_bitwise_xval_binary_set,
#         name="binary_bitwise_zeroes_constraint",
#         doc="Constraint that takes Gray notation with the individual piecewise points to enforce neighboring points to be zero",
#     )
#     def binary_bitwise_zeroes_constraint(model, time_index, site, log_index):
#         return sum(
#             model.v_piecewise_point_weights[time_index, site, point_index]
#             for point_index in model.s_bitwise_binary_zeroes_indexed_set[
#                 time_index, site, log_index
#             ]
#         ) <= (1 - model.v_log_piecewise_active[time_index, site, log_index])



# Piecewise examples

# def minimum_piecewise_yval(model, time_period, site, piecewise_segment):
#     return (
#         model.v_piecewise_yval[time_period, site, piecewise_segment]
#         - model.bv_piecewise_on[time_period, site, piecewise_segment]
#         * model.p_piecewise_yval_min[time_period, site, piecewise_segment]
#         >= 0
#     )

# def piecewise_linear_yval_xval_link(model, time_period, site, piecewise_segment):
#     return (
#         model.v_piecewise_yval[time_period, site, piecewise_segment]
#         - model.p_piecewise_slope[time_period, site, piecewise_segment]
#         * model.v_piecewise_xval[time_period, site, piecewise_segment]
#         - model.p_piecewise_intercept[time_period, site, piecewise_segment]
#         * model.bv_piecewise_on[time_period, site, piecewise_segment]
#         == 0
#     )

# def piecewise_xval_value_max(model, time_period, site, piecewise_segment):
#     return (
#         model.p_piecewise_xval_max[time_period, site, piecewise_segment]
#         * model.bv_piecewise_on[time_period, site, piecewise_segment]
#         - model.v_piecewise_xval[time_period, site, piecewise_segment]
#         >= 0
#     )

# def piecewise_yval_sos_constraint(model, time_period, site):
#     return (
#         sum(
#             model.bv_piecewise_on[time_period, site, piecewise_segment]
#             for piecewise_segment in model.s_piecewise_yval_indexed_set[
#                 time_period, site
#             ]
#         )
#         <= 1
#     )

# def piecewise_yval_link(model, time_period, site):
#     return (
#         sum(
#             model.v_piecewise_yval[time_period, site, piecewise_segment]
#             for piecewise_segment in model.s_piecewise_yval_indexed_set[
#                 time_period, site
#             ]
#         )
#         == model.v_distribution_yval[time_period, site]
#     )

# def piecewise_xval_link(model, time_period, site):
#     return (
#         sum(
#             model.v_piecewise_xval[time_period, site, piecewise_segment]
#             for piecewise_segment in model.s_piecewise_yval_indexed_set[
#                 time_period, site
#             ]
#         )
#         == model.v_distribution_xval[time_period, site]
#     )

# def link_distribution_state(model, time_period, pump_site):
#     return (
#         model.v_distribution_yval[time_period, pump_site]
#         <= model.bv_distribution_on[time_period, pump_site] * model.p_big_m[None][None]
#     )
