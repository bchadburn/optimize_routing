from dataclasses import dataclass, field
from numbers import Number
from typing import Dict


@dataclass
class ManufacturingSite:
    """
    Represents a manufacturing site.

    Args:
        site_id (int): Unique identifier for the site.
        capacity (int): Manufacturing site capacity.
    """
    site_id: int
    capacity: int
    transport_cost_m_to_d: Dict[int, float] = field(default_factory=dict)
    
    def set_mf_to_dist_transport_costs(self, dist_id, transport_cost_m_to_d: Number):
        self.transport_cost_m_to_d[dist_id] = transport_cost_m_to_d
          
        
@dataclass
class DistributionSite:
    """
    Represents a distribution site.

    Args:
        site_id (int): Unique identifier for the distribution site.
        opening_cost (float): Cost associated with opening the site.
    """
    site_id: int
    opening_cost: float
    transport_cost_d_to_c: Dict[int, float] = field(default_factory=dict)

    def set_dist_to_cust_transport_costs(self, cust_id: int, transport_cost_d_to_c: float):
        self.transport_cost_d_to_c[cust_id] = transport_cost_d_to_c

    
@dataclass
class Customer:
    """
    Represents a customer.

    Args:
        customer_id(int): Unique identifier for the customer.
        mean_demand (float): Mean demand from the customer.
        std_dev_demand (float): Standard deviation of demand.
    """
    customer_id: int
    mean_demand: float
    std_dev_demand: float
    

@dataclass
class SupplyChainData:
    """
    Container of supply chain data.
    """
    manufacturing_sites: Dict[int, ManufacturingSite] = field(default_factory=dict)
    distribution_sites: Dict[int, DistributionSite] = field(default_factory=dict)
    customers: Dict[int, 'Customer'] = field(default_factory=dict)

    def add_manufacturing_site(self, site_id: int, capacity: int):
        self.manufacturing_sites[site_id] = ManufacturingSite(site_id, capacity)

    def add_distribution_site(self, site_id: int, opening_cost: float):
        self.distribution_sites[site_id] = DistributionSite(site_id, opening_cost)

    def add_customer(self, customer_id: int, mean_demand: float, std_dev_demand: float):
        self.customers[customer_id] = Customer(customer_id, mean_demand, std_dev_demand)


from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class SimulationParameters:
    """Include simulation meta-parameters such as number of simulations or days to simulate.
    Model parameters should be passed to ORToolsCPModel
    Args:
        num_days (int): Number of days to simulate.
        num_simulations (int): Number of simulation runs.
        decision_rolling_period (int): Rolling period for decision updates.
    """
    num_days: int
    num_simulations: int
    decision_rolling_period: int

    MIN_VALUE: ClassVar[int] = 0  # Minimum allowed value for all attributes

    def __post_init__(self):
        self._validate_range('num_days', self.num_days)
        self._validate_range('num_simulations', self.num_simulations)
        self._validate_range('decision_rolling_period', self.decision_rolling_period)

    def _validate_range(self, attr_name: str, attr_value: int):
        if attr_value < self.MIN_VALUE:
            raise ValueError(f"{attr_name} must be greater than or equal to {self.MIN_VALUE}")

    
        