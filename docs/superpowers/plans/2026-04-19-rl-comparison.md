# RL vs MILP Supply Chain Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tabular Q-learning agent alongside the existing MILP optimizer, producing a rigorous cost comparison with learning curves and policy insights across three methods: MILP global, MILP daily myopic, and Q-learning.

**Architecture:** The RL agent handles binary DC-open decisions; the existing LP solver computes optimal routing costs given those decisions (hierarchical decomposition). A new `rl/` package contains environment, agent, and training. A thin comparison notebook imports result CSVs and renders visualizations.

**Tech Stack:** Python 3.12, OR-Tools (existing), NumPy, pandas, matplotlib, uv, pytest

---

## File Map

**New files:**
- `rl/__init__.py`
- `rl/environment.py` — `SupplyChainEnv`: state transitions, reward, rolling window enforcement, LP routing sub-call
- `rl/agent.py` — `QLearningAgent`: Q-table, epsilon-greedy, update rule
- `rl/train.py` — training loop, policy evaluation, results export
- `utils/results.py` — CSV writing, summary table formatting
- `tests/test_environment.py`
- `tests/test_agent.py`
- `tests/test_results.py`
- `comparison.ipynb`
- `pyproject.toml`
- `README.md`
- `results/.gitkeep`

**Modified files:**
- `optimizer/run_optimizer.py` — extract `build_supply_chain_data()` helper and `run_global_milp()` / `run_daily_myopic()` functions callable from outside; remove old `__main__` block with test data
- `optimizer/construct_data_objects.py` — rename to `optimizer/data_objects.py` (cleaner name, matches spec)
- `optimizer/math_model_constraints.py` — remove dead commented-out piecewise code (lines 62–128)

---

## Task 1: Project scaffolding — pyproject.toml, results dir, cleanup

**Files:**
- Create: `pyproject.toml`
- Create: `results/.gitkeep`
- Create: `rl/__init__.py`
- Create: `tests/__init__.py`
- Modify: `optimizer/math_model_constraints.py` (remove dead code)
- Modify: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "optimize-routing"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "ortools>=9.9",
    "numpy>=1.26",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "notebook>=7.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["optimizer", "ortools_objects", "rl", "utils"]
```

- [ ] **Step 2: Create results directory and empty rl package**

```bash
mkdir -p results
touch results/.gitkeep
mkdir -p rl
touch rl/__init__.py
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 3: Remove dead piecewise code from math_model_constraints.py**

Delete lines 62–128 of `optimizer/math_model_constraints.py` (all commented-out piecewise functions). The file should end at line 59 after `distribution_shipments_equal_total_received_shipments`.

- [ ] **Step 4: Add results/ to .gitignore**

Append to `.gitignore`:
```
results/*.csv
results/*.json
```

- [ ] **Step 5: Install dependencies**

```bash
uv sync --extra dev
```
Expected: resolves and installs all dependencies including pytest.

- [ ] **Step 6: Verify existing tests still pass**

```bash
uv run pytest tests/ -v 2>/dev/null || echo "no tests yet — OK"
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml results/.gitkeep rl/__init__.py tests/__init__.py optimizer/math_model_constraints.py .gitignore
git commit -m "Scaffold project: pyproject.toml, rl package, results dir, cleanup"
```

---

## Task 2: Refactor optimizer — clean public API

**Files:**
- Modify: `optimizer/run_optimizer.py`
- Modify: `optimizer/__init__.py`

The existing `run_optimizer.py` has everything in one `optimize()` function that prints to stdout and a `__main__` block with hardcoded test data. We need clean, importable functions: `build_supply_chain_data()`, `run_global_milp()`, and `run_daily_myopic()`.

- [ ] **Step 1: Write failing test for build_supply_chain_data**

Create `tests/test_optimizer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_optimizer.py::test_build_supply_chain_data_returns_correct_counts -v
```
Expected: FAIL — `ImportError: cannot import name 'build_supply_chain_data'`

- [ ] **Step 3: Rewrite optimizer/run_optimizer.py**

```python
import logging
import numpy as np
from ortools.linear_solver import pywraplp

import utils.log as log
from optimizer.construct_data_objects import SupplyChainData, SimulationParameters
from optimizer.math_model_declaration import create_math_model
from optimizer.math_model_constraints import minimize_cost_objective
from ortools_objects.model import ORToolsCPModel


def build_supply_chain_data(
    distribution_opening_costs: list[float],
    mfg_site_capacity: list[float],
    mean_demand: list[float],
    std_dev_demand: list[float],
    transport_cost_m_to_d: list[list[float]],
    transport_cost_d_to_c: list[list[float]],
) -> SupplyChainData:
    """Build a SupplyChainData object from raw parameter lists."""
    data = SupplyChainData()
    for mf_id, cap in enumerate(mfg_site_capacity):
        data.add_manufacturing_site(site_id=mf_id, capacity=cap)
    for dist_id, cost in enumerate(distribution_opening_costs):
        data.add_distribution_site(site_id=dist_id, opening_cost=cost)
    for cust_id, (mean, std) in enumerate(zip(mean_demand, std_dev_demand)):
        data.add_customer(customer_id=cust_id, mean_demand=mean, std_dev_demand=std)
    for mf_id in range(len(mfg_site_capacity)):
        for dist_id in range(len(distribution_opening_costs)):
            data.manufacturing_sites[mf_id].set_mf_to_dist_transport_costs(
                dist_id, transport_cost_m_to_d[mf_id][dist_id]
            )
    for dist_id in range(len(distribution_opening_costs)):
        for cust_id in range(len(mean_demand)):
            data.distribution_sites[dist_id].set_dist_to_cust_transport_costs(
                cust_id, transport_cost_d_to_c[dist_id][cust_id]
            )
    return data


def _make_model(logger: logging.Logger) -> ORToolsCPModel:
    return ORToolsCPModel(
        logger=logger,
        max_time=30,
        rel_gap=0.00,
        solver_log=False,
        shallow_substitute=True,
    )


def run_global_milp(
    supply_chain_data: SupplyChainData,
    num_days: int = 10,
    decision_rolling_period: int = 3,
) -> dict:
    """Run a single global MILP solve over the full horizon.

    Returns dict with keys: total_cost, dc_decisions (list of sets of open DC ids per day),
    transport_cost_m_to_d (float), transport_cost_d_to_c (float).
    """
    logger = log.get_logger("MILP-Global")
    sim_params = SimulationParameters(num_days, 1, decision_rolling_period)
    model = _make_model(logger)
    create_math_model(model, supply_chain_data, sim_params)
    model.construct_model()
    status = model.solve_model()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("Global MILP: no feasible solution found")

    total_cost = minimize_cost_objective(model)
    dc_decisions = [
        {d for d in model.s_distribution_sites() if model.bv_distribution_on[day, d].solution_value() > 0.5}
        for day in model.s_time_indices()
    ]
    transport_m_to_d = sum(
        model.p_transport_cost_m_to_d[(m, d)] * model.v_transport_m_to_d[d, day, m].solution_value()
        for d in model.s_distribution_sites()
        for day in model.s_time_indices()
        for m in model.s_manufacturing_sites()
    )
    transport_d_to_c = sum(
        model.p_transport_cost_d_to_c[(d, c)] * model.v_transport_d_to_c[d, day, c].solution_value()
        for d in model.s_distribution_sites()
        for day in model.s_time_indices()
        for c in model.s_customers()
    )
    return {
        "total_cost": total_cost,
        "dc_decisions": dc_decisions,
        "transport_cost_m_to_d": transport_m_to_d,
        "transport_cost_d_to_c": transport_d_to_c,
    }


def run_daily_myopic(
    supply_chain_data: SupplyChainData,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    num_simulations: int = 10,
) -> dict:
    """Run daily myopic MILP solves (re-solve each day with fresh demand sample).

    Returns dict with keys: mean_total_cost, std_total_cost, costs_per_simulation (list).
    """
    logger = log.get_logger("MILP-Daily")
    costs = []
    for _ in range(num_simulations):
        sim_params = SimulationParameters(1, 1, decision_rolling_period)
        total_cost = 0.0
        for _day in range(num_days):
            model = _make_model(logger)
            create_math_model(model, supply_chain_data, sim_params)
            model.construct_model()
            status = model.solve_model()
            if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                total_cost += 1e6  # infeasibility penalty
            else:
                total_cost += minimize_cost_objective(model)
        costs.append(total_cost)
    return {
        "mean_total_cost": float(np.mean(costs)),
        "std_total_cost": float(np.std(costs)),
        "costs_per_simulation": costs,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_optimizer.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add optimizer/run_optimizer.py tests/test_optimizer.py
git commit -m "Refactor optimizer: clean public API with build_supply_chain_data, run_global_milp, run_daily_myopic"
```

---

## Task 3: RL Environment

**Files:**
- Create: `rl/environment.py`
- Create: `tests/test_environment.py`

The environment wraps the supply chain problem in a step/reset interface. At each step the agent chooses which DCs to open (as a bitmask); the environment enforces the rolling window constraint, then calls the LP solver to get the optimal routing cost for that DC configuration, and returns the negative cost as reward.

- [ ] **Step 1: Write failing tests**

Create `tests/test_environment.py`:

```python
import pytest
import numpy as np
from optimizer.run_optimizer import build_supply_chain_data
from rl.environment import SupplyChainEnv

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

@pytest.fixture
def env():
    data = build_supply_chain_data(**PARAMS)
    return SupplyChainEnv(supply_chain_data=data, num_days=10, decision_rolling_period=3, seed=42)

def test_reset_returns_valid_state(env):
    state = env.reset()
    day, demand_bucket, dc_mask = state
    assert day == 0
    assert demand_bucket in (0, 1, 2)
    assert 0 <= dc_mask <= 31

def test_step_returns_negative_reward(env):
    env.reset()
    # Open DC 0 only (bitmask = 1)
    next_state, reward, done = env.step(action=1)
    assert reward < 0
    assert not done

def test_done_on_final_day(env):
    env.reset()
    for _ in range(9):
        env.step(action=1)
    _, _, done = env.step(action=1)
    assert done

def test_rolling_window_forces_dc_open(env):
    """DC opened on day 0 must stay open for rolling_period days."""
    env.reset()
    # Open DC 0 (bitmask=1) on day 0
    env.step(action=1)
    # Try to close all DCs (action=0) on day 1 — DC 0 must remain forced open
    _, _, _ = env.step(action=0)
    assert env.forced_open_mask & 1  # bit 0 still set

def test_demand_bucket_coverage(env):
    """After many resets, all 3 demand buckets should appear."""
    buckets = set()
    for _ in range(200):
        state = env.reset()
        buckets.add(state[1])
    assert buckets == {0, 1, 2}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_environment.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rl.environment'`

- [ ] **Step 3: Implement rl/environment.py**

```python
"""Supply chain RL environment.

State: (day: int, demand_bucket: int, dc_status_bitmask: int)
  - day: 0–(num_days-1)
  - demand_bucket: 0=low, 1=med, 2=high (based on total demand percentiles)
  - dc_status_bitmask: integer 0–(2**num_dcs - 1) indicating open DCs

Action: integer 0–(2**num_dcs - 1) — desired DC open set as bitmask.
  Rolling window constraint is enforced before execution: any DC opened within
  the last rolling_period days cannot be closed.

Reward: -(dc_opening_costs + lp_routing_cost) for the executed action.
  Returns -1e6 if no DC is open (infeasible).
"""
from __future__ import annotations

import numpy as np
from ortools.linear_solver import pywraplp

from optimizer.construct_data_objects import SupplyChainData, SimulationParameters
from optimizer.math_model_declaration import create_math_model
from optimizer.math_model_constraints import minimize_cost_objective
from ortools_objects.model import ORToolsCPModel


class SupplyChainEnv:
    def __init__(
        self,
        supply_chain_data: SupplyChainData,
        num_days: int = 10,
        decision_rolling_period: int = 3,
        seed: int | None = None,
    ) -> None:
        self.data = supply_chain_data
        self.num_days = num_days
        self.rolling_period = decision_rolling_period
        self.num_dcs = len(supply_chain_data.distribution_sites)
        self.rng = np.random.default_rng(seed)

        # Compute demand percentile thresholds from distribution (1000 samples)
        samples = np.array([
            sum(
                max(0, self.rng.normal(c.mean_demand, c.std_dev_demand))
                for c in supply_chain_data.customers.values()
            )
            for _ in range(1000)
        ])
        self._low_thresh = float(np.percentile(samples, 33))
        self._high_thresh = float(np.percentile(samples, 66))

        self._day: int = 0
        self._dc_status_bitmask: int = 0   # which DCs are open
        self._open_start: dict[int, int] = {}  # dc_id -> day it was last opened
        self.forced_open_mask: int = 0
        self._daily_demands: list[dict[int, float]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[int, int, int]:
        """Reset environment. Returns initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._day = 0
        self._dc_status_bitmask = 0
        self._open_start = {}
        self.forced_open_mask = 0
        self._daily_demands = self._sample_demands()
        return self._get_state()

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        """Execute action. Returns (next_state, reward, done)."""
        if self._day >= self.num_days:
            raise RuntimeError("Episode is done. Call reset() first.")

        executed_action = self._enforce_rolling_window(action)
        reward = self._compute_reward(executed_action)
        self._update_dc_status(executed_action)
        self._day += 1
        done = self._day >= self.num_days
        return self._get_state(), reward, done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> tuple[int, int, int]:
        total_demand = sum(self._daily_demands[min(self._day, self.num_days - 1)].values())
        if total_demand <= self._low_thresh:
            bucket = 0
        elif total_demand <= self._high_thresh:
            bucket = 1
        else:
            bucket = 2
        return (self._day, bucket, self._dc_status_bitmask)

    def _sample_demands(self) -> list[dict[int, float]]:
        return [
            {
                cust_id: max(0.0, float(self.rng.normal(c.mean_demand, c.std_dev_demand)))
                for cust_id, c in self.data.customers.items()
            }
            for _ in range(self.num_days)
        ]

    def _enforce_rolling_window(self, desired_action: int) -> int:
        """Return the executed action: desired OR forced-open DCs."""
        self.forced_open_mask = 0
        for dc_id, open_day in self._open_start.items():
            if self._day < open_day + self.rolling_period:
                self.forced_open_mask |= (1 << dc_id)
        return desired_action | self.forced_open_mask

    def _update_dc_status(self, executed_action: int) -> None:
        prev = self._dc_status_bitmask
        self._dc_status_bitmask = executed_action
        for dc_id in range(self.num_dcs):
            newly_opened = (executed_action >> dc_id) & 1 and not (prev >> dc_id) & 1
            if newly_opened:
                self._open_start[dc_id] = self._day

    def _compute_reward(self, executed_action: int) -> float:
        if executed_action == 0:
            return -1e6  # no DCs open — infeasible

        open_dcs = [dc_id for dc_id in range(self.num_dcs) if (executed_action >> dc_id) & 1]
        demands = self._daily_demands[self._day]

        # DC opening cost: incurred if DC was not open in previous rolling window
        dc_cost = 0.0
        for dc_id in open_dcs:
            last_opened = self._open_start.get(dc_id, -self.rolling_period)
            if self._day >= last_opened + self.rolling_period or last_opened == -self.rolling_period:
                dc_cost += self.data.distribution_sites[dc_id].opening_cost

        # LP routing cost for this day given open DCs
        lp_cost = self._solve_routing_lp(open_dcs, demands)
        return -(dc_cost + lp_cost)

    def _solve_routing_lp(self, open_dcs: list[int], demands: dict[int, float]) -> float:
        """Solve a single-day LP to find optimal routing cost given open DCs."""
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            return 1e6

        mf_ids = list(self.data.manufacturing_sites.keys())
        cust_ids = list(self.data.customers.keys())

        # Variables: flow from m->d and d->c
        x_md = {
            (m, d): solver.NumVar(0, solver.infinity(), f"x_md_{m}_{d}")
            for m in mf_ids for d in open_dcs
        }
        x_dc = {
            (d, c): solver.NumVar(0, solver.infinity(), f"x_dc_{d}_{c}")
            for d in open_dcs for c in cust_ids
        }

        # Manufacturing capacity constraints
        for m in mf_ids:
            solver.Add(
                sum(x_md[m, d] for d in open_dcs)
                <= self.data.manufacturing_sites[m].capacity
            )

        # Customer demand constraints
        for c in cust_ids:
            solver.Add(
                sum(x_dc[d, c] for d in open_dcs) == demands[c]
            )

        # Flow balance at each DC
        for d in open_dcs:
            solver.Add(
                sum(x_md[m, d] for m in mf_ids)
                == sum(x_dc[d, c] for c in cust_ids)
            )

        # Objective: minimize transport cost
        obj = solver.Objective()
        for m in mf_ids:
            for d in open_dcs:
                cost = self.data.manufacturing_sites[m].transport_cost_m_to_d[d]
                obj.SetCoefficient(x_md[m, d], cost)
        for d in open_dcs:
            for c in cust_ids:
                cost = self.data.distribution_sites[d].transport_cost_d_to_c[c]
                obj.SetCoefficient(x_dc[d, c], cost)
        obj.SetMinimization()

        status = solver.Solve()
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return solver.Objective().Value()
        return 1e6  # infeasible routing
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_environment.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/environment.py tests/test_environment.py
git commit -m "Add SupplyChainEnv with rolling window enforcement and LP routing sub-solver"
```

---

## Task 4: Q-Learning Agent

**Files:**
- Create: `rl/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
import numpy as np
import pytest
from rl.agent import QLearningAgent

def test_q_table_shape():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    assert agent.q_table.shape == (10, 3, 32, 32)

def test_initial_q_table_is_zero():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    assert np.all(agent.q_table == 0.0)

def test_select_action_returns_valid_action():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5)
    action = agent.select_action(state=(0, 1, 0))
    assert 0 <= action <= 31

def test_epsilon_greedy_explores_at_high_epsilon():
    """With epsilon=1.0 all actions should be random (non-greedy)."""
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, epsilon=1.0, seed=0)
    # Force a clear preference in Q-table
    agent.q_table[0, 0, 0, 5] = 999.0
    actions = [agent.select_action((0, 0, 0)) for _ in range(50)]
    # With epsilon=1.0, action 5 should NOT be chosen exclusively
    assert len(set(actions)) > 1

def test_update_changes_q_value():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, alpha=0.5, gamma=0.9)
    state = (0, 1, 0)
    action = 3
    reward = -500.0
    next_state = (1, 1, 3)
    agent.update(state, action, reward, next_state, done=False)
    assert agent.q_table[0, 1, 0, 3] != 0.0

def test_update_at_terminal_state_ignores_next():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5, alpha=1.0, gamma=0.9)
    state = (9, 0, 0)
    action = 1
    reward = -300.0
    next_state = (10, 0, 0)  # beyond horizon
    agent.update(state, action, reward, next_state, done=True)
    assert agent.q_table[9, 0, 0, 1] == pytest.approx(-300.0)

def test_epsilon_decays():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5,
                           epsilon=1.0, epsilon_end=0.01, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.25)

def test_epsilon_does_not_decay_below_epsilon_end():
    agent = QLearningAgent(num_days=10, num_demand_buckets=3, num_dcs=5,
                           epsilon=0.015, epsilon_end=0.01, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_agent.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'rl.agent'`

- [ ] **Step 3: Implement rl/agent.py**

```python
"""Tabular Q-learning agent for supply chain DC-open decisions.

Q-table shape: (num_days, num_demand_buckets, dc_status_bitmask, action)
  = (10, 3, 32, 32) for a 5-DC, 10-day problem.

State: (day, demand_bucket, dc_status_bitmask)
Action: integer 0–31 — desired DC open set as bitmask.
"""
from __future__ import annotations

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        num_days: int,
        num_demand_buckets: int,
        num_dcs: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        self.num_days = num_days
        self.num_dcs = num_dcs
        self.num_actions = 2 ** num_dcs
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (num_days, num_demand_buckets, self.num_actions, self.num_actions),
            dtype=np.float64,
        )

    def select_action(self, state: tuple[int, int, int]) -> int:
        """Epsilon-greedy action selection. Returns action bitmask."""
        day, demand_bucket, dc_mask = state
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        return int(np.argmax(self.q_table[day, demand_bucket, dc_mask]))

    def update(
        self,
        state: tuple[int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int],
        done: bool,
    ) -> None:
        """Q-learning update: Q(s,a) ← Q(s,a) + α(r + γ max_a' Q(s',a') - Q(s,a))"""
        day, demand_bucket, dc_mask = state
        current_q = self.q_table[day, demand_bucket, dc_mask, action]

        if done:
            target = reward
        else:
            next_day, next_bucket, next_dc_mask = next_state
            next_max_q = np.max(self.q_table[next_day, next_bucket, next_dc_mask])
            target = reward + self.gamma * next_max_q

        self.q_table[day, demand_bucket, dc_mask, action] += self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """Multiply epsilon by decay factor, clamped to epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_action(self, state: tuple[int, int, int]) -> int:
        """Return greedy (no exploration) action for policy evaluation."""
        day, demand_bucket, dc_mask = state
        return int(np.argmax(self.q_table[day, demand_bucket, dc_mask]))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_agent.py -v
```
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rl/agent.py tests/test_agent.py
git commit -m "Add QLearningAgent with epsilon-greedy selection and Q-learning update"
```

---

## Task 5: Training loop and results export

**Files:**
- Create: `rl/train.py`
- Create: `utils/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing test for results export**

Create `tests/test_results.py`:

```python
import os
import pytest
import pandas as pd
from utils.results import write_csv, write_learning_curve, write_policy_table

def test_write_csv_creates_file(tmp_path):
    rows = [{"method": "test", "total_cost": 1234.5, "day": 0}]
    out = tmp_path / "test.csv"
    write_csv(rows, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["method", "total_cost", "day"]
    assert df.iloc[0]["total_cost"] == pytest.approx(1234.5)

def test_write_learning_curve(tmp_path):
    rewards = [-1000.0, -900.0, -800.0]
    out = tmp_path / "lc.csv"
    write_learning_curve(rewards, out, window=2)
    df = pd.read_csv(out)
    assert "episode" in df.columns
    assert "episode_reward" in df.columns
    assert "smoothed_reward" in df.columns
    assert len(df) == 3

def test_write_policy_table(tmp_path):
    # policy_map: {(day, demand_bucket): action_bitmask}
    policy_map = {(0, 0): 1, (0, 1): 3, (1, 2): 7}
    out = tmp_path / "policy.csv"
    write_policy_table(policy_map, out, num_dcs=5)
    df = pd.read_csv(out)
    assert "day" in df.columns
    assert "demand_bucket" in df.columns
    assert "action_bitmask" in df.columns
    assert "open_dcs" in df.columns
    assert len(df) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_results.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.results'`

- [ ] **Step 3: Implement utils/results.py**

```python
"""Utilities for writing result CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_learning_curve(
    episode_rewards: list[float],
    path: Path,
    window: int = 100,
) -> None:
    """Write learning curve CSV with smoothed reward column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rewards = np.array(episode_rewards)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="same")
    df = pd.DataFrame({
        "episode": np.arange(len(rewards)),
        "episode_reward": rewards,
        "smoothed_reward": smoothed,
    })
    df.to_csv(path, index=False)


def write_policy_table(
    policy_map: dict[tuple[int, int], int],
    path: Path,
    num_dcs: int,
) -> None:
    """Write policy table: (day, demand_bucket) -> action bitmask + human-readable DC list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (day, bucket), action in sorted(policy_map.items()):
        open_dcs = [dc_id for dc_id in range(num_dcs) if (action >> dc_id) & 1]
        rows.append({
            "day": day,
            "demand_bucket": bucket,
            "action_bitmask": action,
            "open_dcs": str(open_dcs),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
```

- [ ] **Step 4: Run results tests to verify they pass**

```bash
uv run pytest tests/test_results.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Implement rl/train.py**

```python
"""Training loop, policy evaluation, and results export for Q-learning agent."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from optimizer.run_optimizer import build_supply_chain_data, run_global_milp, run_daily_myopic
from rl.agent import QLearningAgent
from rl.environment import SupplyChainEnv
from utils.results import write_csv, write_learning_curve, write_policy_table

RESULTS_DIR = Path("results")

# Default supply chain parameters (matching the notebook scenario)
DEFAULT_PARAMS = dict(
    distribution_opening_costs=[350, 320, 375, 400, 550],
    mfg_site_capacity=[600000, 600000],
    mean_demand=[20, 30, 25, 40, 35, 28, 32, 50, 26, 38, 34, 27],
    std_dev_demand=[5.0] * 12,
    transport_cost_m_to_d=[
        [3.5, 2.5, 4.5, 2.5, 3.0],
        [2.5, 4.5, 5.5, 6.5, 8.5],
    ],
    transport_cost_d_to_c=[
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1],
    ],
)


def train(
    episodes: int = 15_000,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    seed: int = 42,
    log_interval: int = 500,
) -> tuple[QLearningAgent, list[float]]:
    """Train Q-learning agent. Returns (agent, episode_rewards)."""
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnv(supply_chain_data, num_days, decision_rolling_period, seed=seed)
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
            print(f"Episode {ep+1}/{episodes} | avg reward (last {log_interval}): {recent:.1f} | ε={agent.epsilon:.4f}")

    return agent, episode_rewards


def evaluate_rl(
    agent: QLearningAgent,
    num_eval_episodes: int = 100,
    num_days: int = 10,
    decision_rolling_period: int = 3,
    seed: int = 99,
) -> dict:
    """Evaluate greedy policy over multiple episodes."""
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)
    env = SupplyChainEnv(supply_chain_data, num_days, decision_rolling_period, seed=seed)
    costs = []
    for _ in range(num_eval_episodes):
        state = env.reset()
        total_cost = 0.0
        done = False
        while not done:
            action = agent.greedy_action(state)
            state, reward, done = env.step(action)
            total_cost -= reward
        costs.append(total_cost)
    return {
        "mean_total_cost": float(np.mean(costs)),
        "std_total_cost": float(np.std(costs)),
        "costs_per_episode": costs,
    }


def extract_policy_table(
    agent: QLearningAgent,
    num_days: int = 10,
) -> dict[tuple[int, int], int]:
    """Extract greedy policy as (day, demand_bucket) -> action map."""
    policy = {}
    num_actions = agent.num_actions
    for day in range(num_days):
        for bucket in range(3):
            # Use dc_mask=0 (all closed) as canonical starting state for policy table
            action = int(np.argmax(agent.q_table[day, bucket, 0]))
            policy[(day, bucket)] = action
    return policy


def run_all(
    episodes: int = 15_000,
    num_milp_simulations: int = 10,
    seed: int = 42,
) -> None:
    """Run all three methods and write results CSVs."""
    RESULTS_DIR.mkdir(exist_ok=True)
    supply_chain_data = build_supply_chain_data(**DEFAULT_PARAMS)

    print("Running MILP global solve...")
    milp_global = run_global_milp(supply_chain_data)
    write_csv(
        [{"method": "milp_global", "total_cost": milp_global["total_cost"],
          "transport_cost_m_to_d": milp_global["transport_cost_m_to_d"],
          "transport_cost_d_to_c": milp_global["transport_cost_d_to_c"]}],
        RESULTS_DIR / "milp_global.csv",
    )
    print(f"  MILP global total cost: ${milp_global['total_cost']:,.0f}")

    print("Running MILP daily myopic solves...")
    milp_daily = run_daily_myopic(supply_chain_data, num_simulations=num_milp_simulations)
    write_csv(
        [{"method": "milp_daily", "simulation": i, "total_cost": c}
         for i, c in enumerate(milp_daily["costs_per_simulation"])],
        RESULTS_DIR / "milp_daily.csv",
    )
    print(f"  MILP daily mean cost: ${milp_daily['mean_total_cost']:,.0f} ± {milp_daily['std_total_cost']:,.0f}")

    print(f"Training Q-learning agent ({episodes} episodes)...")
    agent, episode_rewards = train(episodes=episodes, seed=seed)
    write_learning_curve(episode_rewards, RESULTS_DIR / "learning_curve.csv")

    print("Evaluating RL policy...")
    rl_results = evaluate_rl(agent, seed=seed + 1)
    write_csv(
        [{"method": "rl", "episode": i, "total_cost": c}
         for i, c in enumerate(rl_results["costs_per_episode"])],
        RESULTS_DIR / "rl_policy.csv",
    )
    print(f"  RL mean cost: ${rl_results['mean_total_cost']:,.0f} ± {rl_results['std_total_cost']:,.0f}")

    policy_map = extract_policy_table(agent)
    write_policy_table(
        policy_map,
        RESULTS_DIR / "rl_policy_table.csv",
        num_dcs=len(supply_chain_data.distribution_sites),
    )

    # Print summary
    milp_opt = milp_global["total_cost"]
    rl_mean = rl_results["mean_total_cost"]
    daily_mean = milp_daily["mean_total_cost"]
    print("\n=== Summary ===")
    print(f"{'Method':<25} {'Mean Cost':>12} {'Gap vs MILP Global':>20}")
    print(f"{'MILP Global (optimal)':<25} ${milp_opt:>11,.0f} {'—':>20}")
    print(f"{'MILP Daily Myopic':<25} ${daily_mean:>11,.0f} {(daily_mean/milp_opt - 1)*100:>19.1f}%")
    print(f"{'Q-Learning':<25} ${rl_mean:>11,.0f} {(rl_mean/milp_opt - 1)*100:>19.1f}%")


if __name__ == "__main__":
    run_all()
```

- [ ] **Step 6: Smoke-test training runs without crashing**

```bash
uv run python -c "
from rl.train import train, evaluate_rl
agent, rewards = train(episodes=50, log_interval=50)
result = evaluate_rl(agent, num_eval_episodes=5)
print('RL eval mean cost:', result['mean_total_cost'])
print('SMOKE TEST PASSED')
"
```
Expected: prints mean cost and SMOKE TEST PASSED (no exceptions).

- [ ] **Step 7: Commit**

```bash
git add rl/train.py utils/results.py tests/test_results.py
git commit -m "Add training loop, policy evaluation, and results export"
```

---

## Task 6: Comparison notebook

**Files:**
- Create: `comparison.ipynb`

The notebook calls `run_all()` to generate CSVs, then renders three outputs. All computation lives in the package — the notebook only loads CSVs and plots.

- [ ] **Step 1: Create comparison.ipynb**

Create `comparison.ipynb` with the following cells. Use `jupyter nbformat` to create it programmatically, or create it manually in Jupyter:

**Cell 1 — Run all methods (skip if CSVs exist):**
```python
from pathlib import Path
from rl.train import run_all

if not Path("results/milp_global.csv").exists():
    run_all(episodes=15_000, num_milp_simulations=10)
else:
    print("Results already exist — loading from CSVs.")
```

**Cell 2 — Cost comparison table:**
```python
import pandas as pd

milp_global = pd.read_csv("results/milp_global.csv")
milp_daily = pd.read_csv("results/milp_daily.csv")
rl = pd.read_csv("results/rl_policy.csv")

milp_opt = milp_global["total_cost"].iloc[0]
daily_mean = milp_daily["total_cost"].mean()
daily_std = milp_daily["total_cost"].std()
rl_mean = rl["total_cost"].mean()
rl_std = rl["total_cost"].std()

summary = pd.DataFrame([
    {"Method": "MILP Global (optimal)", "Mean Cost ($)": f"{milp_opt:,.0f}", "Std ($)": "—", "Gap vs Optimal": "—"},
    {"Method": "MILP Daily Myopic",     "Mean Cost ($)": f"{daily_mean:,.0f}", "Std ($)": f"{daily_std:,.0f}", "Gap vs Optimal": f"{(daily_mean/milp_opt - 1)*100:.1f}%"},
    {"Method": "Q-Learning",            "Mean Cost ($)": f"{rl_mean:,.0f}",    "Std ($)": f"{rl_std:,.0f}",    "Gap vs Optimal": f"{(rl_mean/milp_opt - 1)*100:.1f}%"},
])
summary.set_index("Method")
```

**Cell 3 — Learning curve plot:**
```python
import matplotlib.pyplot as plt
import pandas as pd

lc = pd.read_csv("results/learning_curve.csv")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lc["episode"], lc["episode_reward"], alpha=0.2, color="steelblue", label="Episode reward")
ax.plot(lc["episode"], lc["smoothed_reward"], color="steelblue", linewidth=2, label="Smoothed (100-ep avg)")
ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative reward (negative cost)")
ax.set_title("Q-Learning Training Curve")
ax.legend()
plt.tight_layout()
plt.show()
```

**Cell 4 — Policy table:**
```python
import pandas as pd

policy = pd.read_csv("results/rl_policy_table.csv")
policy["demand_label"] = policy["demand_bucket"].map({0: "Low", 1: "Med", 2: "High"})
pivot = policy.pivot(index="day", columns="demand_label", values="open_dcs")
print("Learned DC policy: open_dcs per (day, demand level)")
pivot
```

- [ ] **Step 2: Verify notebook runs end-to-end**

```bash
uv run jupyter nbconvert --to notebook --execute comparison.ipynb --output comparison_executed.ipynb 2>&1 | tail -5
```
Expected: exits 0, produces `comparison_executed.ipynb`.

```bash
rm comparison_executed.ipynb
```

- [ ] **Step 3: Commit**

```bash
git add comparison.ipynb
git commit -m "Add comparison notebook with cost table, learning curve, and policy table"
```

---

## Task 7: README and Further Explorations

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

```markdown
# Supply Chain Routing Optimization — MILP vs Q-Learning

Multi-period supply chain routing problem comparing three solution approaches:
global MILP (provably optimal), daily myopic MILP, and tabular Q-learning.

**Problem:** 10-day planning horizon, 2 manufacturing sites, 5 distribution centers,
12 customers with stochastic demand. Minimize total cost: DC opening costs +
transportation costs (manufacturing → DC → customer).

## Approaches

| Method | Description | Optimality |
|---|---|---|
| **MILP Global** | Solves the full 10-day horizon as one MILP (OR-Tools/SCIP) | Provably optimal |
| **MILP Daily Myopic** | Re-solves a 1-day MILP each day with fresh demand | Suboptimal (~4% gap) |
| **Q-Learning** | Learns DC-open policy via tabular Q-learning; LP solves routing given DC decisions | Heuristic |

The RL decomposition is principled: the agent handles the combinatorial DC-open decisions
(binary, multi-period, rolling window constraint), while the LP optimally routes flow
given those decisions. This is the correct separation — routing is a solved subproblem.

## Results

Run `uv run python -m rl.train` to generate results. See `comparison.ipynb` for the
full comparison table, learning curve, and policy insights.

## Quickstart

```bash
uv sync
uv run python -m rl.train          # run all three methods, write results/
jupyter lab comparison.ipynb        # open comparison notebook
uv run pytest tests/ -v             # run test suite
```

## Project Structure

```
optimizer/          OR-Tools MILP model (sets, params, vars, constraints, objective)
ortools_objects/    Reusable OR-Tools OOP abstraction layer
rl/
  environment.py    SupplyChainEnv — state, action, reward, rolling window, LP sub-solver
  agent.py          QLearningAgent — Q-table, epsilon-greedy, Q-learning update
  train.py          Training loop, evaluation, results export
utils/
  results.py        CSV writing utilities
tests/              Pytest suite
results/            Output CSVs (gitignored)
comparison.ipynb    Comparison notebook
```

## Further Explorations

- **DQN**: Replace the tabular Q-table with a neural network to handle larger state spaces
  (100+ DCs, longer horizons). Tabular Q-learning is exact but does not scale beyond ~1,000 states.
- **Multi-agent RL**: Assign one agent per DC for decentralized policy learning in
  multi-echelon networks.
- **Constraint-aware RL**: Encode the rolling window constraint directly into the reward
  via Lagrangian relaxation instead of environment-side enforcement.
- **RL as MILP warm-start**: Use the RL policy to generate a high-quality initial solution
  for the MILP solver, reducing solve time on large instances.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add README with problem description, results, and further explorations"
```

---

## Task 8: Full test suite and final run

**Files:** No new files — verify everything works end-to-end.

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```
Expected: All tests PASS. Count should be ≥ 18 tests across test_optimizer.py, test_environment.py, test_agent.py, test_results.py.

- [ ] **Step 2: Run full comparison pipeline**

```bash
uv run python -m rl.train
```
Expected: Prints episode logs, then summary table like:
```
=== Summary ===
Method                    Mean Cost   Gap vs MILP Global
MILP Global (optimal)     $16,964                     —
MILP Daily Myopic         $17,655                  4.1%
Q-Learning                $XX,XXX                  X.X%
```

- [ ] **Step 3: Verify all result CSVs were written**

```bash
ls results/
```
Expected: `learning_curve.csv  milp_daily.csv  milp_global.csv  rl_policy.csv  rl_policy_table.csv`

- [ ] **Step 4: Delete old notebook**

```bash
git rm multi_period_routing_MILP_refactor.ipynb
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "Remove old notebook — replaced by comparison.ipynb and rl/ package"
```

- [ ] **Step 6: Push**

```bash
git push
```
