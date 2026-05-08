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
        daily_demand (list[float] | None): Optional explicit per-day demand vector
            (length must equal ``num_days`` of the simulation). When set, the
            optimizer uses these values directly instead of resampling
            ``Normal(mean_demand, std_dev_demand)`` per day. Used by the
            Bayesian-forecast pipeline (#13) to feed the chance-constrained
            MILP a 95th-percentile demand series, and by the CVaR scenario-MILP
            to feed each scenario's realized demand.
    """
    def __init__(
        self,
        customer_id: int,
        mean_demand: Number,
        std_dev_demand: Number,
        daily_demand: list[float] | None = None,
    ):
        self.customer_id = customer_id
        self.mean_demand = mean_demand
        self.std_dev_demand = std_dev_demand
        self.daily_demand = daily_demand

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

    def add_customer(
        self,
        customer_id: int,
        mean_demand: Number,
        std_dev_demand: Number,
        daily_demand: list[float] | None = None,
    ):
        self.customers[customer_id] = Customer(
            customer_id, mean_demand, std_dev_demand, daily_demand=daily_demand,
        )

    def clone(self) -> "SupplyChainData":
        """Deep-copy this SupplyChainData. Used by stochastic optimization
        wrappers that want to inject per-customer ``daily_demand`` without
        mutating the original (other solvers may run against the same instance).

        Centralizing the clone here — vs. reaching into private dicts at the
        call site — keeps the copy logic correct as new fields get added to
        SupplyChainData / its child classes.
        """
        new = SupplyChainData()
        for site_id, site in self.manufacturing_sites.items():
            new.add_manufacturing_site(site_id, site.capacity)
            for d, cost in site.transport_cost_m_to_d.items():
                new.manufacturing_sites[site_id].set_mf_to_dist_transport_costs(d, cost)
        for site_id, site in self.distribution_sites.items():
            new.add_distribution_site(site_id, site.opening_cost)
            for c, cost in site.transport_cost_d_to_c.items():
                new.distribution_sites[site_id].set_dist_to_cust_transport_costs(c, cost)
        for cust_id, cust in self.customers.items():
            new.add_customer(
                customer_id=cust_id,
                mean_demand=cust.mean_demand,
                std_dev_demand=cust.std_dev_demand,
                daily_demand=(
                    list(cust.daily_demand) if cust.daily_demand is not None else None
                ),
            )
        return new


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
        