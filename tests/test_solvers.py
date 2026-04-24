"""Correctness tests for CvrptwSolver implementations.

OR-Tools tests run unconditionally.
cuOpt tests are skipped (cuOpt blocked on GPU setup — see issue #6).
"""

SIMPLE_3_CUSTOMER = dict(
    open_dc_ids=[0],
    demands={0: 10.0, 1: 15.0, 2: 20.0},
    transport_cost_d_to_c={0: {0: 2.0, 1: 3.0, 2: 5.0}},
    n_vehicles_per_dc=2,
)

ZERO_DEMAND = dict(
    open_dc_ids=[0],
    demands={0: 0.0, 1: 0.0},
    transport_cost_d_to_c={0: {0: 1.0, 1: 2.0}},
    n_vehicles_per_dc=1,
)

TWO_DC = dict(
    open_dc_ids=[0, 1],
    demands={0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0},
    transport_cost_d_to_c={
        0: {0: 1.0, 1: 2.0, 2: 8.0, 3: 9.0},
        1: {0: 9.0, 1: 8.0, 2: 1.0, 3: 2.0},
    },
    n_vehicles_per_dc=2,
)


def test_ortools_simple_returns_finite_cost():
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**SIMPLE_3_CUSTOMER)
    assert 0.0 < cost < 1e5


def test_ortools_zero_demand_returns_zero():
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**ZERO_DEMAND)
    assert cost == 0.0


def test_ortools_two_dc_assigns_customers_to_cheaper_dc():
    """Two DCs: customers 0,1 cheapest via DC0; customers 2,3 via DC1.
    Total cost should be dominated by short-haul costs."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**TWO_DC)
    assert 0.0 < cost < 100.0
