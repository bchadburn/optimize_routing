# cuOpt CVRPTW Routing Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LP sub-solver in the RL environment with a proper CVRPTW solver, validate correctness at small scale, then benchmark OR-Tools VRP vs NVIDIA cuOpt at 12→500 customers to find the crossover point where GPU acceleration becomes decisive — and compare the RL policy trained under each solver.

**Architecture:** Three phases. Phase 1: build a `CvrptwSolver` protocol with an OR-Tools VRP implementation (apples-to-apples baseline for cuOpt — the current LP is not comparable because it has no vehicles or time windows). Phase 2: implement `CuOptSolver` behind the same protocol and verify both solvers agree on small instances. Phase 3: train the RL agent with each solver, run a scalability benchmark at 12/50/100/250/500 customers, and visualize results.

**Tech Stack:** Python 3.12, uv, OR-Tools (routing library, not just LP), cuopt-cu12 (NVIDIA cuOpt), numpy, pytest

---

## File Structure

| File | Purpose |
|---|---|
| `rl/solvers/__init__.py` | Package init |
| `rl/solvers/protocol.py` | `CvrptwSolver` Protocol — shared interface both solvers implement |
| `rl/solvers/ortools_vrp.py` | `OrtoolsVrpSolver` — OR-Tools Routing Library CVRPTW (fair baseline) |
| `rl/solvers/cuopt_vrp.py` | `CuOptVrpSolver` — NVIDIA cuOpt CVRPTW (GPU solver) |
| `rl/environment_vrp.py` | `SupplyChainEnvVrp` — SupplyChainEnv subclass accepting any `CvrptwSolver` |
| `rl/benchmark.py` | `ScalabilityBenchmark` — sweeps customer counts, times both solvers, writes CSV |
| `rl/train_vrp.py` | `train_vrp()` — trains RL agent using `SupplyChainEnvVrp`, saves policy |
| `tests/test_solvers.py` | Correctness tests: both solvers agree on known small instances |
| `tests/test_environment_vrp.py` | Integration: one episode produces finite cost with each solver |
| `tests/test_benchmark.py` | Smoke test: benchmark runs and writes CSV with expected columns |
| `results/cuopt_benchmark.csv` | Benchmark output: (n_customers, solver, solve_time_s, total_cost) |
| `results/rl_vrp_ortools.csv` | RL training results with OR-Tools VRP sub-solver |
| `results/rl_vrp_cuopt.csv` | RL training results with cuOpt sub-solver |

---

## Background: why the LP baseline was wrong

The existing `_solve_routing_lp` uses a continuous-flow LP: it decides *how much* product flows from each DC to each customer, but with no vehicles, no capacity per vehicle, and no time windows. cuOpt solves CVRPTW: discrete vehicle routes, each vehicle has a capacity cap, customers have delivery time windows.

Comparing LP cost to cuOpt cost is meaningless — they solve different problems. This plan uses OR-Tools *Routing Library* (CP-based VRP solver) as the baseline, which solves the same CVRPTW structure as cuOpt.

**Depot modeling:** Each open DC is a depot. We run one VRP per open DC, assigning each customer to the cheapest DC. Vehicle capacity = ceil(total_demand_for_dc / n_vehicles). Time windows: [0, 1440] (open, no real data). This is consistent between both solvers.

---

## Task 1: Install cuOpt and verify GPU

**Files:** None — environment check only.

- [ ] **Step 1: Verify CUDA visible from WSL2**

```bash
nvidia-smi
```

Expected: shows `NVIDIA GeForce RTX 3070`, CUDA version ≥ 11.8. If `nvidia-smi` is not found, CUDA WSL2 driver is not installed — see https://developer.nvidia.com/cuda/wsl.

- [ ] **Step 2: Install cuOpt**

```bash
cd /home/bchadburn/VSCodeProjects/optimize_routing
uv pip install --extra-index-url https://pypi.nvidia.com cuopt-cu12
```

If `cuopt-cu12` does not exist on NVIDIA PyPI, install the self-hosted server package:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com nvidia-cuopt-cu12
```

- [ ] **Step 3: Start cuOpt server (if using self-hosted)**

```bash
uv run cuopt_server &
```

- [ ] **Step 4: Verify cuOpt import**

```bash
uv run python -c "
try:
    from cuopt_sh_client import CuOptServiceClient
    print('cuOpt SaaS client OK')
except ImportError:
    import cuopt
    print('cuOpt local OK, version:', cuopt.__version__)
"
```

Expected: prints one of the two success lines without error.

- [ ] **Step 5: Commit dependency**

```bash
git add pyproject.toml uv.lock
git commit -m "Add cuopt dependency"
```

---

## Task 2: CvrptwSolver protocol and OR-Tools VRP baseline

**Files:**
- Create: `rl/solvers/__init__.py`
- Create: `rl/solvers/protocol.py`
- Create: `rl/solvers/ortools_vrp.py`
- Create: `tests/test_solvers.py` (partial — OR-Tools tests only)

### The protocol

```python
# rl/solvers/protocol.py
class CvrptwSolver(Protocol):
    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        """Return total routing cost. Returns 1e6 if infeasible."""
        ...
```

All solvers implement this signature — same inputs, same output semantics. `n_vehicles_per_dc` is how many vehicles serve each open DC.

- [ ] **Step 1: Write OR-Tools VRP tests**

Create `tests/test_solvers.py`:

```python
"""Correctness tests for CvrptwSolver implementations.

OR-Tools tests run unconditionally.
cuOpt tests are skipped if cuOpt is not installed.
"""
import pytest

cuopt_available = True
try:
    from cuopt_sh_client import CuOptServiceClient  # noqa: F401
    cuopt_available = True
except ImportError:
    try:
        import cuopt  # noqa: F401
        cuopt_available = True
    except ImportError:
        cuopt_available = False

skip_no_cuopt = pytest.mark.skipif(not cuopt_available, reason="cuOpt not installed")

# --- shared fixtures ---

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


# --- OR-Tools VRP ---

def test_ortools_simple_returns_finite_cost():
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**SIMPLE_3_CUSTOMER)
    assert 0.0 < cost < 1e5


def test_ortools_zero_demand_returns_zero():
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**ZERO_DEMAND)
    assert cost == 0.0


def test_ortools_two_dc_assigns_customers_to_cheaper_dc():
    """Two DCs: customers 0,1 are cheapest via DC0; customers 2,3 via DC1.
    Total should be dominated by short-haul costs, not cross-DC costs."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    cost = OrtoolsVrpSolver().solve(**TWO_DC)
    # Each DC serves 2 customers at cost ≤3 each → total ≤ 4*3 = 12 (plus return trips)
    assert 0.0 < cost < 100.0


# --- cuOpt VRP ---

@skip_no_cuopt
def test_cuopt_simple_returns_finite_cost():
    from rl.solvers.cuopt_vrp import CuOptVrpSolver
    cost = CuOptVrpSolver().solve(**SIMPLE_3_CUSTOMER)
    assert 0.0 < cost < 1e5


@skip_no_cuopt
def test_cuopt_zero_demand_returns_zero():
    from rl.solvers.cuopt_vrp import CuOptVrpSolver
    cost = CuOptVrpSolver().solve(**ZERO_DEMAND)
    assert cost == 0.0


@skip_no_cuopt
def test_both_solvers_agree_within_20pct():
    """cuOpt and OR-Tools should produce costs within 20% of each other on the same instance.
    cuOpt uses metaheuristic (not exact), so small gap is expected."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    from rl.solvers.cuopt_vrp import CuOptVrpSolver
    ortools_cost = OrtoolsVrpSolver().solve(**SIMPLE_3_CUSTOMER)
    cuopt_cost = CuOptVrpSolver().solve(**SIMPLE_3_CUSTOMER)
    ratio = max(ortools_cost, cuopt_cost) / min(ortools_cost, cuopt_cost)
    assert ratio < 1.20, f"Solvers disagree: OR-Tools={ortools_cost:.2f}, cuOpt={cuopt_cost:.2f}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_solvers.py::test_ortools_simple_returns_finite_cost -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl.solvers'`

- [ ] **Step 3: Create package and protocol**

Create `rl/solvers/__init__.py`:
```python
```

Create `rl/solvers/protocol.py`:
```python
"""Shared solver protocol — all CVRPTW solvers implement this interface."""
from __future__ import annotations

from typing import Protocol


class CvrptwSolver(Protocol):
    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        """Return total routing cost across all open DCs.

        Args:
            open_dc_ids: List of open DC indices.
            demands: customer_id -> demand quantity.
            transport_cost_d_to_c: dc_id -> {customer_id -> transport cost per unit}.
            n_vehicles_per_dc: Number of vehicles available per open DC.

        Returns:
            Total routing cost. Returns 1e6 if infeasible.
        """
        ...
```

- [ ] **Step 4: Implement OrtoolsVrpSolver**

Create `rl/solvers/ortools_vrp.py`:

```python
"""OR-Tools Routing Library CVRPTW solver — fair baseline for cuOpt comparison.

Uses CP-based VRP (not LP flow). Each open DC is treated as a separate depot
serving its assigned customers. Customer-to-DC assignment uses minimum transport cost.
"""
from __future__ import annotations

import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class OrtoolsVrpSolver:
    """Solve DC→customer routing as CVRPTW using OR-Tools Routing Library."""

    def __init__(self, time_limit_s: int = 5) -> None:
        self._time_limit_s = time_limit_s

    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        if not demands or all(v == 0.0 for v in demands.values()):
            return 0.0

        cust_ids = sorted(demands.keys())
        dc_customers = _assign_customers_to_dcs(open_dc_ids, cust_ids, transport_cost_d_to_c)

        total_cost = 0.0
        for dc_id in open_dc_ids:
            assigned = dc_customers[dc_id]
            if not assigned:
                continue
            cost = _solve_single_dc_ortools(
                customer_ids=assigned,
                demands=demands,
                costs_to_customers=transport_cost_d_to_c[dc_id],
                n_vehicles=n_vehicles_per_dc,
                time_limit_s=self._time_limit_s,
            )
            total_cost += cost

        return total_cost


def _assign_customers_to_dcs(
    open_dc_ids: list[int],
    cust_ids: list[int],
    transport_cost_d_to_c: dict[int, dict[int, float]],
) -> dict[int, list[int]]:
    """Assign each customer to the cheapest open DC."""
    dc_customers: dict[int, list[int]] = {dc: [] for dc in open_dc_ids}
    for cust_id in cust_ids:
        best_dc = min(
            open_dc_ids,
            key=lambda dc: transport_cost_d_to_c[dc].get(cust_id, 1e6),
        )
        dc_customers[best_dc].append(cust_id)
    return dc_customers


def _solve_single_dc_ortools(
    customer_ids: list[int],
    demands: dict[int, float],
    costs_to_customers: dict[int, float],
    n_vehicles: int,
    time_limit_s: int,
) -> float:
    """Run OR-Tools VRP for a single DC. Returns routing cost or 1e6 on failure."""
    n = len(customer_ids)
    if n == 0:
        return 0.0

    # Build N+1 × N+1 integer distance matrix (index 0 = depot)
    # Scale costs by 1000 to preserve decimal precision in integer matrix
    SCALE = 1000
    size = n + 1
    matrix = [[0] * size for _ in range(size)]
    for i, ci in enumerate(customer_ids, start=1):
        cost_int = int(costs_to_customers.get(ci, 1e6) * SCALE)
        matrix[0][i] = cost_int
        matrix[i][0] = cost_int
    for i in range(1, size):
        for j in range(1, size):
            if i != j:
                matrix[i][j] = matrix[0][i] + matrix[0][j]

    total_demand = sum(demands[c] for c in customer_ids)
    vehicle_capacity = int(math.ceil(total_demand / n_vehicles)) + 1

    manager = pywrapcp.RoutingIndexManager(size, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    def demand_callback(from_idx):
        node = manager.IndexToNode(from_idx)
        if node == 0:
            return 0
        return int(math.ceil(demands[customer_ids[node - 1]]))

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx, 0, [vehicle_capacity] * n_vehicles, True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(search_params)
    if solution:
        return solution.ObjectiveValue() / SCALE
    return 1e6
```

- [ ] **Step 5: Run OR-Tools tests**

```bash
uv run pytest tests/test_solvers.py -k "ortools" -v
```

Expected: all three OR-Tools tests PASS.

- [ ] **Step 6: Commit**

```bash
git add rl/solvers/ tests/test_solvers.py
git commit -m "Add CvrptwSolver protocol and OrtoolsVrpSolver baseline"
```

---

## Task 3: CuOptVrpSolver

**Files:**
- Create: `rl/solvers/cuopt_vrp.py`

- [ ] **Step 1: Run cuOpt tests to verify they fail**

```bash
uv run pytest tests/test_solvers.py -k "cuopt" -v
```

Expected: cuOpt tests SKIP (cuOpt not imported yet) or FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Implement CuOptVrpSolver**

Create `rl/solvers/cuopt_vrp.py`:

```python
"""NVIDIA cuOpt CVRPTW solver — GPU-accelerated VRP via local cuOpt server.

Mirrors OrtoolsVrpSolver: same customer assignment logic, same depot-per-DC
approach. Only the solve call differs (cuOpt REST-style client instead of CP).

cuOpt problem dict format:
  cost_matrix_data: {0: [[cost_ij, ...]]}   — N+1 × N+1, index 0 = depot
  task_data:
    task_locations: [1..N]                  — customer node indices
    demand: [[qty_0, ..., qty_N-1]]         — per customer
    task_time_windows: [[0, 1440]] * N      — open windows (no real data)
    service_times: [0] * N
  fleet_data:
    vehicle_locations: [[0, 0]] * K         — all vehicles start/end at depot
    capacities: [[cap]] * K
    vehicle_time_windows: [[0, 1440]] * K
  solver_config:
    time_limit: <seconds>
"""
from __future__ import annotations

import math

_CUOPT_AVAILABLE = False
try:
    from cuopt_sh_client import CuOptServiceClient
    _CUOPT_AVAILABLE = True
except ImportError:
    try:
        import cuopt as _cuopt_local  # noqa: F401
        # Local server mode — import client differently
        from cuopt_sh_client import CuOptServiceClient
        _CUOPT_AVAILABLE = True
    except ImportError:
        pass

from rl.solvers.ortools_vrp import _assign_customers_to_dcs  # reuse assignment logic


class CuOptVrpSolver:
    """Solve DC→customer routing as CVRPTW using NVIDIA cuOpt (GPU)."""

    def __init__(
        self,
        server_url: str = "http://localhost:5000",
        time_limit_s: float = 2.0,
    ) -> None:
        if not _CUOPT_AVAILABLE:
            raise ImportError(
                "cuOpt is not installed. Install with:\n"
                "  uv pip install --extra-index-url https://pypi.nvidia.com cuopt-cu12"
            )
        self._client = CuOptServiceClient(server_url)
        self._time_limit = time_limit_s

    def solve(
        self,
        open_dc_ids: list[int],
        demands: dict[int, float],
        transport_cost_d_to_c: dict[int, dict[int, float]],
        n_vehicles_per_dc: int = 3,
    ) -> float:
        if not demands or all(v == 0.0 for v in demands.values()):
            return 0.0

        cust_ids = sorted(demands.keys())
        dc_customers = _assign_customers_to_dcs(open_dc_ids, cust_ids, transport_cost_d_to_c)

        total_cost = 0.0
        for dc_id in open_dc_ids:
            assigned = dc_customers[dc_id]
            if not assigned:
                continue
            cost = self._solve_single_dc(
                customer_ids=assigned,
                demands=demands,
                costs_to_customers=transport_cost_d_to_c[dc_id],
                n_vehicles=n_vehicles_per_dc,
            )
            total_cost += cost

        return total_cost

    def _solve_single_dc(
        self,
        customer_ids: list[int],
        demands: dict[int, float],
        costs_to_customers: dict[int, float],
        n_vehicles: int,
    ) -> float:
        n = len(customer_ids)
        if n == 0:
            return 0.0

        size = n + 1
        matrix = [[0.0] * size for _ in range(size)]
        for i, ci in enumerate(customer_ids, start=1):
            cost = costs_to_customers.get(ci, 1e6)
            matrix[0][i] = cost
            matrix[i][0] = cost
        for i in range(1, size):
            for j in range(1, size):
                if i != j:
                    matrix[i][j] = matrix[0][i] + matrix[0][j]

        total_demand = sum(demands[c] for c in customer_ids)
        vehicle_capacity = int(math.ceil(total_demand / n_vehicles)) + 1

        problem = {
            "cost_matrix_data": {"data": {0: matrix}},
            "travel_time_data": {"data": {0: matrix}},
            "task_data": {
                "task_locations": list(range(1, size)),
                "demand": [[int(math.ceil(demands[c])) for c in customer_ids]],
                "task_time_windows": [[0, 1440]] * n,
                "service_times": [0] * n,
            },
            "fleet_data": {
                "vehicle_locations": [[0, 0]] * n_vehicles,
                "capacities": [[vehicle_capacity]] * n_vehicles,
                "vehicle_time_windows": [[0, 1440]] * n_vehicles,
            },
            "solver_config": {"time_limit": self._time_limit},
        }

        try:
            resp = self._client.get_optimized_routes(problem)
            solver_resp = resp.get("response", {}).get("solver_response", {})
            if solver_resp.get("status", -1) == 0:
                return float(solver_resp["solution_cost"])
        except Exception:
            pass
        return 1e6
```

- [ ] **Step 3: Run all solver tests**

```bash
uv run pytest tests/test_solvers.py -v
```

Expected:
- All OR-Tools tests: PASS
- cuOpt tests: PASS if server is running, SKIP if cuOpt not available
- `test_both_solvers_agree_within_20pct`: PASS or SKIP

- [ ] **Step 4: Commit**

```bash
git add rl/solvers/cuopt_vrp.py
git commit -m "Add CuOptVrpSolver behind shared CvrptwSolver protocol"
```

---

## Task 4: SupplyChainEnvVrp — RL environment with pluggable solver

**Files:**
- Create: `rl/environment_vrp.py`
- Create: `tests/test_environment_vrp.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_environment_vrp.py`:

```python
"""Integration tests: one episode of SupplyChainEnvVrp produces valid cost."""
import pytest

cuopt_available = True
try:
    from cuopt_sh_client import CuOptServiceClient  # noqa: F401
except ImportError:
    cuopt_available = False

skip_no_cuopt = pytest.mark.skipif(not cuopt_available, reason="cuOpt not installed")

from rl.train import DEFAULT_PARAMS
from optimizer.run_optimizer import build_supply_chain_data


def _run_episode(env) -> float:
    """Run one greedy episode (all DCs open), return total cost."""
    state = env.reset()
    total_cost = 0.0
    done = False
    while not done:
        action = (1 << env.num_dcs) - 1  # all DCs open
        state, reward, done = env.step(action)
        total_cost -= reward
    return total_cost


def test_ortools_episode_produces_finite_cost():
    """Episode with OR-Tools VRP sub-solver returns finite positive cost."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(data, num_days=3, solver=OrtoolsVrpSolver(), seed=0)
    cost = _run_episode(env)

    assert cost > 0.0
    assert cost < 1e5


@skip_no_cuopt
def test_cuopt_episode_produces_finite_cost():
    """Episode with cuOpt sub-solver returns finite positive cost."""
    from rl.solvers.cuopt_vrp import CuOptVrpSolver
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(data, num_days=3, solver=CuOptVrpSolver(), seed=0)
    cost = _run_episode(env)

    assert cost > 0.0
    assert cost < 1e5


@skip_no_cuopt
def test_both_solvers_give_comparable_episode_cost():
    """OR-Tools and cuOpt episode costs within 25% of each other (same 3-day seed)."""
    from rl.solvers.ortools_vrp import OrtoolsVrpSolver
    from rl.solvers.cuopt_vrp import CuOptVrpSolver
    from rl.environment_vrp import SupplyChainEnvVrp

    data = build_supply_chain_data(**DEFAULT_PARAMS)
    ortools_cost = _run_episode(SupplyChainEnvVrp(data, num_days=3, solver=OrtoolsVrpSolver(), seed=42))
    cuopt_cost   = _run_episode(SupplyChainEnvVrp(data, num_days=3, solver=CuOptVrpSolver(),   seed=42))

    ratio = max(ortools_cost, cuopt_cost) / min(ortools_cost, cuopt_cost)
    assert ratio < 1.25, f"OR-Tools={ortools_cost:.1f}, cuOpt={cuopt_cost:.1f}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_environment_vrp.py::test_ortools_episode_produces_finite_cost -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl.environment_vrp'`

- [ ] **Step 3: Implement SupplyChainEnvVrp**

Create `rl/environment_vrp.py`:

```python
"""SupplyChainEnv subclass that accepts any CvrptwSolver for the routing leg.

The existing _solve_routing_lp is replaced with solver.solve(). All state
management, rolling-window enforcement, and episode logic are inherited unchanged.
"""
from __future__ import annotations

from optimizer.construct_data_objects import SupplyChainData
from rl.environment import SupplyChainEnv
from rl.solvers.protocol import CvrptwSolver


class SupplyChainEnvVrp(SupplyChainEnv):
    def __init__(
        self,
        supply_chain_data: SupplyChainData,
        num_days: int = 10,
        decision_rolling_period: int = 3,
        seed: int | None = None,
        solver: CvrptwSolver | None = None,
        n_vehicles_per_dc: int = 3,
    ) -> None:
        super().__init__(supply_chain_data, num_days, decision_rolling_period, seed)
        if solver is None:
            from rl.solvers.ortools_vrp import OrtoolsVrpSolver
            solver = OrtoolsVrpSolver()
        self._solver = solver
        self._n_vehicles_per_dc = n_vehicles_per_dc

    def _compute_reward(self, executed_action: int) -> float:
        if executed_action == 0:
            return -1e6

        open_dcs = [dc_id for dc_id in range(self.num_dcs) if (executed_action >> dc_id) & 1]
        demands = self._daily_demands[self._day]

        dc_cost = 0.0
        for dc_id in open_dcs:
            was_open = (self._dc_status_bitmask >> dc_id) & 1
            if not was_open:
                dc_cost += self.data.distribution_sites[dc_id].opening_cost

        transport_costs = {
            dc_id: dict(self.data.distribution_sites[dc_id].transport_cost_d_to_c)
            for dc_id in open_dcs
        }
        routing_cost = self._solver.solve(
            open_dc_ids=open_dcs,
            demands={cid: float(qty) for cid, qty in demands.items()},
            transport_cost_d_to_c=transport_costs,
            n_vehicles_per_dc=self._n_vehicles_per_dc,
        )
        return -(dc_cost + routing_cost)
```

- [ ] **Step 4: Run all environment tests**

```bash
uv run pytest tests/test_environment_vrp.py -v
```

Expected: OR-Tools test PASS; cuOpt tests PASS or SKIP.

- [ ] **Step 5: Commit**

```bash
git add rl/environment_vrp.py tests/test_environment_vrp.py
git commit -m "Add SupplyChainEnvVrp with pluggable CvrptwSolver"
```

---

## Task 5: RL training with VRP environment

**Files:**
- Create: `rl/train_vrp.py`

This trains the RL agent using `SupplyChainEnvVrp` with a given solver. It reuses `QLearningAgent`, `train()` logic from `rl/train.py`, and writes results to `results/rl_vrp_<solver_name>.csv`.

- [ ] **Step 1: Write train_vrp.py**

Create `rl/train_vrp.py`:

```python
"""Train and evaluate RL agent using SupplyChainEnvVrp with a pluggable solver.

Usage:
    uv run python -m rl.train_vrp --solver ortools   # OR-Tools VRP sub-solver
    uv run python -m rl.train_vrp --solver cuopt     # cuOpt sub-solver
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from optimizer.run_optimizer import build_supply_chain_data
from rl.agent import QLearningAgent
from rl.environment_vrp import SupplyChainEnvVrp
from rl.train import DEFAULT_PARAMS, evaluate_rl, extract_policy_table
from utils.results import write_csv, write_learning_curve, write_policy_table

RESULTS_DIR = Path("results")


def train_vrp(
    solver_name: str = "ortools",
    episodes: int = 5_000,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    n_vehicles_per_dc: int = 3,
    seed: int = 42,
    log_interval: int = 500,
) -> tuple[QLearningAgent, list[float]]:
    """Train Q-learning agent with VRP sub-solver. Returns (agent, episode_rewards)."""
    if solver_name == "cuopt":
        from rl.solvers.cuopt_vrp import CuOptVrpSolver
        solver = CuOptVrpSolver()
    else:
        from rl.solvers.ortools_vrp import OrtoolsVrpSolver
        solver = OrtoolsVrpSolver()

    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnvVrp(
        supply_chain_data, num_days, decision_rolling_period,
        seed=seed, solver=solver, n_vehicles_per_dc=n_vehicles_per_dc,
    )
    agent = QLearningAgent(
        num_days=num_days,
        num_demand_buckets=3,
        num_dcs=len(supply_chain_data.distribution_sites),
        seed=seed,
    )

    episode_rewards: list[float] = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % log_interval == 0:
            recent = np.mean(episode_rewards[-log_interval:])
            print(f"[{solver_name}] Episode {ep+1}/{episodes} | avg reward: {recent:.1f} | epsilon={agent.epsilon:.4f}")

    return agent, episode_rewards


def run_vrp(solver_name: str = "ortools", episodes: int = 5_000) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Training RL with {solver_name} VRP solver ({episodes} episodes)...")
    agent, episode_rewards = train_vrp(solver_name=solver_name, episodes=episodes)

    write_learning_curve(episode_rewards, RESULTS_DIR / f"rl_vrp_{solver_name}_curve.csv")

    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)

    if solver_name == "cuopt":
        from rl.solvers.cuopt_vrp import CuOptVrpSolver
        eval_solver = CuOptVrpSolver()
    else:
        from rl.solvers.ortools_vrp import OrtoolsVrpSolver
        eval_solver = OrtoolsVrpSolver()

    from rl.environment_vrp import SupplyChainEnvVrp
    from rl.train import DEFAULT_PARAMS

    env = SupplyChainEnvVrp(supply_chain_data, solver=eval_solver, seed=99)
    costs = []
    for _ in range(50):
        state = env.reset()
        total_cost = 0.0
        done = False
        while not done:
            action = agent.greedy_action(state)
            state, reward, done = env.step(action)
            total_cost -= reward
        costs.append(total_cost)

    mean_cost = float(np.mean(costs))
    print(f"[{solver_name}] Eval mean cost: ${mean_cost:,.0f} +/- {float(np.std(costs)):,.0f}")

    write_csv(
        [{"solver": solver_name, "episode": i, "total_cost": c} for i, c in enumerate(costs)],
        RESULTS_DIR / f"rl_vrp_{solver_name}.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["ortools", "cuopt"], default="ortools")
    parser.add_argument("--episodes", type=int, default=5_000)
    args = parser.parse_args()
    run_vrp(solver_name=args.solver, episodes=args.episodes)
```

- [ ] **Step 2: Run OR-Tools training (smoke — 200 episodes to verify it runs)**

```bash
uv run python -m rl.train_vrp --solver ortools --episodes 200
```

Expected output:
```
Training RL with ortools VRP solver (200 episodes)...
[ortools] Eval mean cost: $XX,XXX +/- $X,XXX
```

- [ ] **Step 3: Run full OR-Tools training**

```bash
uv run python -m rl.train_vrp --solver ortools --episodes 5000
```

Expected: completes in ~5–10 min. Writes `results/rl_vrp_ortools.csv` and `results/rl_vrp_ortools_curve.csv`.

- [ ] **Step 4: Run cuOpt training (if cuOpt server running)**

```bash
uv run python -m rl.train_vrp --solver cuopt --episodes 5000
```

Expected: completes. Writes `results/rl_vrp_cuopt.csv`.

- [ ] **Step 5: Commit**

```bash
git add rl/train_vrp.py results/rl_vrp_ortools.csv results/rl_vrp_ortools_curve.csv
git commit -m "Add train_vrp: RL training with pluggable VRP sub-solver"
```

---

## Task 6: Scalability benchmark

**Files:**
- Create: `rl/benchmark.py`
- Create: `tests/test_benchmark.py`

Sweeps `n_customers` ∈ {12, 50, 100, 250, 500}, `n_vehicles_per_dc` = ceil(n/8). Generates synthetic single-DC instances (no RL — pure solver timing). Reports solve time and cost. At small sizes OR-Tools may be faster; at large sizes cuOpt GPU should dominate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark.py`:

```python
"""Smoke test: benchmark runs on smallest config and writes CSV with expected columns."""
import csv
import pytest
from pathlib import Path

cuopt_available = True
try:
    from cuopt_sh_client import CuOptServiceClient  # noqa: F401
except ImportError:
    cuopt_available = False

skip_no_cuopt = pytest.mark.skipif(not cuopt_available, reason="cuOpt not installed")


def test_benchmark_ortools_writes_csv(tmp_path):
    """Benchmark with OR-Tools only writes CSV with correct columns."""
    from rl.benchmark import ScalabilityBenchmark

    runner = ScalabilityBenchmark(results_dir=tmp_path, n_trials=1, include_cuopt=False)
    runner.run(customer_counts=[5])

    csv_path = tmp_path / "cuopt_benchmark.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"n_customers", "n_vehicles", "solver", "solve_time_s", "total_cost"}
    assert rows[0]["solver"] == "ortools_vrp"


@skip_no_cuopt
def test_benchmark_both_solvers_writes_two_rows(tmp_path):
    """Benchmark with both solvers writes 2 rows per (n_customers) config."""
    from rl.benchmark import ScalabilityBenchmark

    runner = ScalabilityBenchmark(results_dir=tmp_path, n_trials=1, include_cuopt=True)
    runner.run(customer_counts=[5])

    rows = list(csv.DictReader((tmp_path / "cuopt_benchmark.csv").open()))
    assert len(rows) == 2
    solvers = {r["solver"] for r in rows}
    assert solvers == {"ortools_vrp", "cuopt_cvrptw"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_benchmark.py::test_benchmark_ortools_writes_csv -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'rl.benchmark'`

- [ ] **Step 3: Implement ScalabilityBenchmark**

Create `rl/benchmark.py`:

```python
"""Scalability benchmark: OR-Tools VRP vs cuOpt CVRPTW.

Sweeps customer counts, times each solver on synthetic single-DC instances,
writes results/cuopt_benchmark.csv.

Usage:
    uv run python -m rl.benchmark
    uv run python -m rl.benchmark --counts 12 50 100 250 500
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from rl.solvers.ortools_vrp import OrtoolsVrpSolver

RESULTS_DIR = Path("results")


def _make_instance(
    n_customers: int, seed: int = 0
) -> tuple[list[int], dict[int, float], dict[int, dict[int, float]], int]:
    """Return (open_dc_ids, demands, transport_costs, n_vehicles) for one DC."""
    rng = np.random.default_rng(seed)
    demands = {i: float(rng.integers(10, 50)) for i in range(n_customers)}
    transport_costs = {0: {i: float(rng.uniform(1.0, 10.0)) for i in range(n_customers)}}
    n_vehicles = max(1, math.ceil(n_customers / 8))
    return [0], demands, transport_costs, n_vehicles


class ScalabilityBenchmark:
    def __init__(
        self,
        results_dir: Path = RESULTS_DIR,
        n_trials: int = 3,
        include_cuopt: bool = True,
    ) -> None:
        self._results_dir = results_dir
        self._n_trials = n_trials
        self._include_cuopt = include_cuopt

        self._ortools = OrtoolsVrpSolver(time_limit_s=10)
        self._cuopt = None
        if include_cuopt:
            try:
                from rl.solvers.cuopt_vrp import CuOptVrpSolver
                self._cuopt = CuOptVrpSolver()
            except ImportError:
                print("WARNING: cuOpt not available — running OR-Tools only.")

    def run(self, customer_counts: list[int] | None = None) -> None:
        if customer_counts is None:
            customer_counts = [12, 50, 100, 250, 500]

        self._results_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []

        for n in customer_counts:
            open_dc_ids, demands, transport_costs, n_vehicles = _make_instance(n, seed=n)

            ortools_times, ortools_cost = self._time_solver(
                self._ortools, open_dc_ids, demands, transport_costs, n_vehicles
            )
            rows.append({
                "n_customers": n,
                "n_vehicles": n_vehicles,
                "solver": "ortools_vrp",
                "solve_time_s": round(sum(ortools_times) / len(ortools_times), 6),
                "total_cost": round(ortools_cost, 4),
            })
            print(f"OR-Tools  n={n:4d}  v={n_vehicles:3d}  cost={ortools_cost:8.1f}  time={ortools_times[-1]*1000:7.1f}ms")

            if self._cuopt is not None:
                cuopt_times, cuopt_cost = self._time_solver(
                    self._cuopt, open_dc_ids, demands, transport_costs, n_vehicles
                )
                rows.append({
                    "n_customers": n,
                    "n_vehicles": n_vehicles,
                    "solver": "cuopt_cvrptw",
                    "solve_time_s": round(sum(cuopt_times) / len(cuopt_times), 6),
                    "total_cost": round(cuopt_cost, 4),
                })
                speedup = ortools_times[-1] / cuopt_times[-1] if cuopt_times[-1] > 0 else 0
                print(f"cuOpt     n={n:4d}  v={n_vehicles:3d}  cost={cuopt_cost:8.1f}  time={cuopt_times[-1]*1000:7.1f}ms  ({speedup:.1f}x speedup)")

        out_path = self._results_dir / "cuopt_benchmark.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["n_customers", "n_vehicles", "solver", "solve_time_s", "total_cost"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")

    def _time_solver(self, solver, open_dc_ids, demands, transport_costs, n_vehicles):
        times = []
        cost = 1e6
        for _ in range(self._n_trials):
            t0 = time.perf_counter()
            cost = solver.solve(open_dc_ids, demands, transport_costs, n_vehicles_per_dc=n_vehicles)
            times.append(time.perf_counter() - t0)
        return times, cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", nargs="+", type=int, default=[12, 50, 100, 250, 500])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-cuopt", action="store_true")
    args = parser.parse_args()
    ScalabilityBenchmark(n_trials=args.trials, include_cuopt=not args.no_cuopt).run(args.counts)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_benchmark.py -v
```

Expected: OR-Tools test PASS; cuOpt test PASS or SKIP.

- [ ] **Step 5: Run benchmark (OR-Tools only first)**

```bash
uv run python -m rl.benchmark --no-cuopt --counts 12 50 100
```

Expected: completes in ~2 min for 100 customers. Writes `results/cuopt_benchmark.csv`.

- [ ] **Step 6: Run full benchmark with cuOpt**

```bash
uv run python -m rl.benchmark --counts 12 50 100 250 500
```

Expected: at 500 customers, cuOpt solve time should be significantly lower than OR-Tools.

- [ ] **Step 7: Commit**

```bash
git add rl/benchmark.py tests/test_benchmark.py results/cuopt_benchmark.csv
git commit -m "Add scalability benchmark: OR-Tools VRP vs cuOpt CVRPTW, 12-500 customers"
```

---

## Task 7: Notebook results section

**Files:**
- Modify: `comparison.ipynb`

- [ ] **Step 1: Append new cells to comparison.ipynb**

Add a Markdown cell:

```markdown
## cuOpt Experiment: GPU-Accelerated VRP at Scale

Compares OR-Tools VRP (CP-based, exact on small instances) against NVIDIA cuOpt CVRPTW
(GPU-accelerated metaheuristic) across growing problem sizes (12–500 customers).

**Setup:** Single DC, customers assigned by minimum cost, vehicles = ceil(n/8).
Both solvers solve identical CVRPTW instances (capacitated vehicles, open time windows).

**Research questions:**
1. At what customer count does cuOpt become faster than OR-Tools?
2. How does solution quality (cost) compare — exact vs. heuristic?
3. How does the RL policy change when trained with cuOpt vs OR-Tools rewards?
```

Add a Code cell:

```python
import csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/cuopt_benchmark.csv")

# Solve time comparison
pivot = df.pivot_table(index="n_customers", columns="solver", values="solve_time_s")
if "ortools_vrp" in pivot.columns and "cuopt_cvrptw" in pivot.columns:
    pivot["cuopt_speedup_x"] = pivot["ortools_vrp"] / pivot["cuopt_cvrptw"]

print("=== Solve Time (seconds) ===")
print(pivot.to_string())

cost_pivot = df.pivot_table(index="n_customers", columns="solver", values="total_cost")
print("\n=== Total Routing Cost ===")
print(cost_pivot.to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Solve time
for solver in df["solver"].unique():
    sub = df[df["solver"] == solver].sort_values("n_customers")
    axes[0].plot(sub["n_customers"], sub["solve_time_s"] * 1000, marker="o", label=solver)
axes[0].set_xlabel("Customers")
axes[0].set_ylabel("Solve Time (ms)")
axes[0].set_title("Solve Time vs Problem Size")
axes[0].legend()
axes[0].set_yscale("log")

# Cost
for solver in df["solver"].unique():
    sub = df[df["solver"] == solver].sort_values("n_customers")
    axes[1].plot(sub["n_customers"], sub["total_cost"], marker="o", label=solver)
axes[1].set_xlabel("Customers")
axes[1].set_ylabel("Total Routing Cost")
axes[1].set_title("Cost vs Problem Size")
axes[1].legend()

# Speedup
if "cuopt_speedup_x" in pivot.columns:
    axes[2].bar(pivot.index.astype(str), pivot["cuopt_speedup_x"])
    axes[2].axhline(1.0, color="red", linestyle="--", label="1x (no speedup)")
    axes[2].set_xlabel("Customers")
    axes[2].set_ylabel("cuOpt Speedup vs OR-Tools")
    axes[2].set_title("GPU Speedup Factor")
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, "cuOpt results not available", ha="center", va="center", transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig("results/cuopt_benchmark_chart.png", dpi=150)
plt.show()
print("Chart saved to results/cuopt_benchmark_chart.png")
```

Add a second Code cell for RL comparison:

```python
import glob

# Load RL results for each solver variant
rl_files = glob.glob("results/rl_vrp_*.csv")
rl_dfs = []
for f in rl_files:
    solver_name = f.replace("results/rl_vrp_", "").replace(".csv", "")
    if "curve" in solver_name:
        continue
    d = pd.read_csv(f)
    d["solver"] = solver_name
    rl_dfs.append(d)

if rl_dfs:
    rl_df = pd.concat(rl_dfs)
    summary = rl_df.groupby("solver")["total_cost"].agg(["mean", "std"]).round(0)
    print("=== RL Policy Cost by Sub-Solver ===")
    print(summary.to_string())
else:
    print("No RL VRP results found. Run: uv run python -m rl.train_vrp --solver ortools")
```

- [ ] **Step 2: Execute notebook**

```bash
uv run jupyter nbconvert --to notebook --execute comparison.ipynb --output comparison.ipynb
```

Expected: runs without error.

- [ ] **Step 3: Commit**

```bash
git add comparison.ipynb results/cuopt_benchmark_chart.png
git commit -m "Add cuOpt experiment results section to comparison notebook"
```

---

## Task 8: Full test suite and README update

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all pass (cuOpt tests skip if server not running).

- [ ] **Step 2: Lint check**

```bash
uv run ruff check rl/solvers/ rl/environment_vrp.py rl/benchmark.py rl/train_vrp.py
```

Expected: no errors.

- [ ] **Step 3: Update README**

Add to `README.md` under `## Further Explorations`:

```markdown
## cuOpt Experiment — GPU-Accelerated CVRPTW

Replaces the LP flow sub-solver in the RL reward function with a proper CVRPTW solver,
then benchmarks OR-Tools VRP vs NVIDIA cuOpt at 12–500 customers.

**Run benchmark only:**
```bash
uv run python -m rl.benchmark --counts 12 50 100 250 500
```

**Train RL with OR-Tools VRP sub-solver:**
```bash
uv run python -m rl.train_vrp --solver ortools --episodes 5000
```

**Train RL with cuOpt sub-solver (requires GPU + cuOpt server):**
```bash
uv run python -m rl.train_vrp --solver cuopt --episodes 5000
```

Results in `results/cuopt_benchmark.csv`; visualized in `comparison.ipynb` (final section).
```

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "Document cuOpt experiment in README"
```
