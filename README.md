# Supply Chain Routing Optimization — MILP vs Q-Learning

![CI](https://github.com/bchadburn/optimize_routing/actions/workflows/ci.yml/badge.svg)

Multi-period supply chain routing problem comparing three solution approaches:
global MILP (provably optimal), daily myopic MILP, and tabular Q-learning.

**Problem:** 10-day planning horizon, 2 manufacturing sites, 5 distribution centers,
12 customers with stochastic demand. Minimize total cost: DC opening costs +
transportation costs (manufacturing → DC → customer).

## Approaches

| Method | Description | Optimality |
|---|---|---|
| **MILP Global** | Solves the full 10-day horizon as one MILP (OR-Tools/SCIP) | Provably optimal |
| **MILP Daily Myopic** | Re-solves a 1-day MILP each day with fresh demand | Suboptimal (~14.7% gap) |
| **Q-Learning** | Learns DC-open policy via tabular Q-learning; LP solves routing given DC decisions | Heuristic |

The RL decomposition is principled: the agent handles the combinatorial DC-open decisions
(binary, multi-period, rolling window constraint), while the LP optimally routes flow
given those decisions. This is the correct separation — routing is a solved subproblem.

## Results

| Method | Avg Total Cost | Gap vs Optimal |
|---|---|---|
| MILP Global | $17,020 | — (optimal) |
| MILP Daily Myopic | $19,519 | +14.7% |
| Q-Learning (15k episodes) | $28,715 | +68.7% |

The daily myopic MILP loses 14.7% by ignoring future demand; the Q-learning agent is a
proof-of-concept showing hierarchical RL decomposition — the gap narrows significantly
with a larger state representation or function approximation.

Run `uv run python -m rl.train` to reproduce results. See `comparison.ipynb` for the
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

## cuOpt Experiment — GPU-Accelerated CVRPTW

Replaces the LP flow sub-solver in the RL environment with a proper CVRPTW solver,
then benchmarks OR-Tools VRP vs NVIDIA cuOpt at 12–500 customers.

**Current status:** OR-Tools VRP baseline complete. cuOpt GPU integration pending
WSL2 CUDA setup (see [issue #6](https://github.com/bchadburn/optimize_routing/issues/6)).

**Run benchmark (OR-Tools only):**
```bash
uv run python -m rl.benchmark
```

**Train RL with OR-Tools VRP sub-solver:**
```bash
uv run python -m rl.train_vrp --solver ortools --episodes 5000
```

**Train RL with cuOpt (requires GPU + CUDA in WSL2):**
```bash
uv run python -m rl.train_vrp --solver cuopt --episodes 5000
```

Results in `results/cuopt_benchmark.csv`; visualized in `comparison.ipynb` (final section).

**New files:**
- `rl/solvers/protocol.py` — `CvrptwSolver` protocol (shared interface)
- `rl/solvers/ortools_vrp.py` — OR-Tools Routing Library VRP solver
- `rl/solvers/cuopt_vrp.py` — cuOpt CVRPTW solver (pending GPU, see #6)
- `rl/environment_vrp.py` — `SupplyChainEnvVrp` with pluggable solver
- `rl/train_vrp.py` — RL training with VRP sub-solver
- `rl/benchmark.py` — scalability benchmark

## Further Explorations

- **DQN**: Replace the tabular Q-table with a neural network to handle larger state spaces
  (100+ DCs, longer horizons). Tabular Q-learning is exact but does not scale beyond ~1,000 states.
- **Multi-agent RL**: Assign one agent per DC for decentralized policy learning in
  multi-echelon networks.
- **Constraint-aware RL**: Encode the rolling window constraint directly into the reward
  via Lagrangian relaxation instead of environment-side enforcement.
- **RL as MILP warm-start**: Use the RL policy to generate a high-quality initial solution
  for the MILP solver, reducing solve time on large instances.
