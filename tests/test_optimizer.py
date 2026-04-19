import pytest
from optimizer.run_optimizer import build_supply_chain_data
from optimizer.construct_data_objects import SupplyChainData

PARAMS = dict(
    distribution_opening_costs=[350, 320, 375, 400, 550],
    mfg_site_capacity=[600000, 600000],
    mean_demand=[20, 30, 25, 40, 35, 28, 32, 50, 26, 38, 34, 27],
    std_dev_demand=[5.0] * 12,
    transport_cost_m_to_d=[[3.5, 2.5, 4.5, 2.5, 3.0], [2.5, 4.5, 5.5, 6.5, 8.5]],
    transport_cost_d_to_c=[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    ],
)

def test_build_supply_chain_data_returns_correct_counts():
    data = build_supply_chain_data(**PARAMS)
    assert isinstance(data, SupplyChainData)
    assert len(data.manufacturing_sites) == 2
    assert len(data.distribution_sites) == 5
    assert len(data.customers) == 12

def test_build_supply_chain_data_transport_costs():
    data = build_supply_chain_data(**PARAMS)
    assert data.manufacturing_sites[0].transport_cost_m_to_d[0] == 3.5
    assert data.distribution_sites[2].transport_cost_d_to_c[4] == 1
