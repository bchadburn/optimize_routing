# supply constraints
def distribution_status_constrained_by_d_to_c_supply(model, distribution_site, time_period, customer):
    return model.v_transport_d_to_c[distribution_site, time_period, customer] <= model.bv_distribution_on[time_period, distribution_site] * model.p_big_m[None]
    

def distribution_status_constrained_by_m_to_d_supply(model, distribution_site, time_period, manufacturing_site):
    return (
        model.v_transport_m_to_d[distribution_site, time_period, manufacturing_site] <= model.bv_distribution_on[time_period, distribution_site] * model.p_big_m[None]
    )
 
def distribution_cost_boolean_incurred_rolling_constraint(model, time_period, distribution_site):
    """Rolling Constraint to ensure the cost of turning on distribution site is only assigned once within a set amount of time (decision_rolling_period). 
    The distribution_site_open_duration_rolling_constraint will then be used to ensure the site remains open for set amount of time."""
    open_duration = model.p_decision_rolling_period[None]
    min_open_day = max(0, time_period-open_duration+1)
    return sum([model.bv_distribution_cost_incurred[duration_day, distribution_site] 
                        for duration_day in range(min_open_day, time_period+1)]) <= 1
                   
def distribution_site_open_duration_rolling_constraint(model, time_period, distribution_site):
    """Ensure the site remains open after distribution site was initially opened (i.e. time idx cost was encurred) for set amount of time."""
    open_duration = model.p_decision_rolling_period[None]
    min_open_day = max(0, time_period-open_duration+1)
    bv_cost_incurred_rolling_period = sum([model.bv_distribution_cost_incurred[duration_day, distribution_site]
                                           for duration_day in range(min_open_day, time_period+1)])
    return model.bv_distribution_on[time_period, distribution_site] == bv_cost_incurred_rolling_period 
                
def distribution_supply_equal_capacity(model, time_period, manufacturing_site):
    # Supply constraints so total mfg site supply is not more than capacity at a given distribution site  
    total_supply_m = sum([model.v_transport_m_to_d[d, time_period, manufacturing_site] for d in model.s_distribution_sites()])
    return total_supply_m <= model.p_manufacturing_site_capacity[manufacturing_site]
 
def distribution_shipments_equal_customer_demand(model, time_period, customer):
    # Demand constraints so total shipments is equal to customer demand
    return sum([model.v_transport_d_to_c[d, time_period, customer] for d in model.s_distribution_sites()]) == model.p_customer_demand[time_period, customer]
            
def minimize_cost_objective(model):
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
    return total_cost
    
def distribution_shipments_equal_total_received_shipments(model, time_period, distribution_site):
    # Limit distribution center shipments to the total of received shipments
    total_shipment_from_m_to_d = sum([model.v_transport_m_to_d[distribution_site, time_period, manufacturing_site] for manufacturing_site in model.s_manufacturing_sites()])
    total_shipment_from_d_to_c = sum([model.v_transport_d_to_c[distribution_site, time_period, customer] for customer in model.s_customers()])
    return total_shipment_from_m_to_d == total_shipment_from_d_to_c


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
