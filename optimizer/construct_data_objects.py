from numbers import Number


class ManufacturingSite:
    """
    Represents a manufacturing site.

    Args:
        site_id (int): Unique identifier for the site.
        capacity (float): Manufacturing site capacity.
    """
    def __init__(self, site_id: int, capacity: Number):
        self.site_id = site_id
        self.capacity = capacity
        self.transport_cost_m_to_d = {}
        
    def set_mf_to_dist_transport_costs(self, dist_id, transport_cost_m_to_d: Number):
        self.transport_cost_m_to_d[dist_id] = transport_cost_m_to_d
        
class DistributionSite:
    """
    Represents a distribution site.

    Args:
        site_id (int): Unique identifier for the distribution site.
        opening_cost (float): Cost associated with opening the site.
    """
    
    def __init__(self, site_id: int, opening_cost: Number):
        self.site_id = site_id
        self.opening_cost = opening_cost
        self.transport_cost_d_to_c = {}

    def set_dist_to_cust_transport_costs(self, cust_id, transport_cost_d_to_c: Number):
        self.transport_cost_d_to_c[cust_id] = transport_cost_d_to_c
        

class Customer:
    """
    Represents a customer.

    Args:
        customer_id(int): Unique identifier for the customer.
        mean_demand (float): Mean demand from the customer.
        std_dev_demand (float): Standard deviation of demand.
    """
    def __init__(self, customer_id: int, mean_demand: Number, std_dev_demand: Number):
        self.customer_name = customer_id
        self.mean_demand = mean_demand
        self.std_dev_demand = std_dev_demand

class SupplyChainData:
    """
    Container of supply chain data.
    Args:
        manufacturing_sites (dict): Dictionary of manufacturing sites (site_id -> ManufacturingSite).
        distribution_sites (dict): Dictionary of distribution sites (site_id -> DistributionSite).
        customers (dict): Dictionary of customers (customer_id -> Customer).
    """
    def __init__(self):
        self.manufacturing_sites = {}
        self.distribution_sites = {}
        self.customers = {}

    def add_manufacturing_site(self, site_id: int, capacity: Number):
        self.manufacturing_sites[site_id] = ManufacturingSite(site_id, capacity)

    def add_distribution_site(self, site_id: int, opening_cost: Number):
        self.distribution_sites[site_id] = DistributionSite(site_id, opening_cost)

    def add_customer(self, customer_id: int, mean_demand: Number, std_dev_demand: Number):
        self.customers[customer_id] = Customer(customer_id, mean_demand, std_dev_demand)


class SimulationParameters:
    """Include simulation meta-parameters such as number of simulations or days to simulate. 
    Model parameters should be passed to ORToolsCPModel
    Args:
        num_days (int): Number of days to simulate.
        num_simulations (int): Number of simulation runs.
        decision_rolling_period (int): Rolling period for decision updates.
    """
    def __init__(self, num_days, num_simulations, decision_rolling_period):
        self.num_days = num_days
        self.num_simulations = num_simulations
        self.decision_rolling_period = decision_rolling_period
        