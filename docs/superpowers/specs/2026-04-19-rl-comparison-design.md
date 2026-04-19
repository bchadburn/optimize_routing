# RL vs MILP Supply Chain Comparison — Design Spec
*Date: 2026-04-19*

---

## Goal

Add a tabular Q-learning agent to the existing supply chain routing codebase and produce a rigorous, honest comparison against the existing MILP solver (global optimal and daily myopic baselines). The RL agent learns daily DC-open policies; the comparison documents where RL approaches MILP quality and where it falls short.

---

## Package Structure

```
optimize_routing/
├── optimizer/           # existing — minor cleanup only
│   ├── __init__.py
│   ├── data_objects.py
│   ├── math_model_declaration.py
│   ├── or_tools_functions.py
│   └── run_optimizer.py
├── ortools_objects/     # existing — untouched
├── rl/                  # new
│   ├── __init__.py
│   ├── environment.py   # SupplyChainEnv — state transitions, reward, constraint enforcement
│   ├── agent.py         # QLearningAgent — Q-table, epsilon-greedy, update rule
│   └── train.py         # training loop, policy evaluation, results export
├── utils/
│   ├── log.py           # existing
│   └── results.py       # new — result formatting, CSV output, summary tables
├── tests/               # new — unit tests for env, agent, results
├── results/             # gitignored — output CSVs written here at runtime
├── comparison.ipynb     # lightweight notebook — imports results, renders comparison
├── pyproject.toml       # new — makes project installable with uv
└── README.md            # new — replaces old notebook overview
```

---

## RL Environment Design (`rl/environment.py`)

### Decomposition Principle

RL handles the **binary DC-open decisions**. The existing LP (OR-Tools) solves **optimal routing given those decisions**. This is correct — routing is a solved subproblem; the learning challenge is the DC activation policy over the multi-period horizon.

### State

`(day: int, demand_bucket: int, dc_status_bitmask: int)`

| Component | Range | Description |
|---|---|---|
| `day` | 0–9 | Current day in the planning horizon |
| `demand_bucket` | 0–2 | Today's total demand discretized: 0=low (<33rd pct), 1=med, 2=high (>66th pct) |
| `dc_status_bitmask` | 0–31 | 5-bit integer — which of 5 DCs are currently open |

Total state space: 10 × 3 × 32 = **960 states** — fully tractable for tabular Q-learning.

Demand percentile thresholds computed once from the demand distribution at environment init and held fixed across all episodes.

### Action

Integer 0–31: the **desired DC open set** as a bitmask (which DCs the agent wants open on this day).

The environment enforces the rolling window constraint before executing the action: any DC opened within the last `rolling_period` days cannot be closed. The actual executed action is the agent's desired set merged with the forced-open set. The agent receives the reward for the executed (possibly modified) action, not the desired one.

### Reward

```
reward = -(dc_opening_costs + lp_routing_cost)
```

- `dc_opening_costs`: sum of opening costs for DCs whose cost is incurred today (per rolling window logic)
- `lp_routing_cost`: optimal shipment cost returned by the existing LP solver for the chosen DC configuration
- If no DC is open and demand cannot be met: large penalty (`-1e6`), episode continues

### Episode

- 10 time steps (days 0–9)
- Demand sampled stochastically from each customer's distribution at episode start
- DC status resets to all-closed at episode start
- Returns: cumulative reward = negative total cost for the horizon

---

## Q-Learning Agent Design (`rl/agent.py`)

### Q-Table

`numpy` array of shape `(10, 3, 32, 32)` — indexed by `(day, demand_bucket, dc_status_bitmask, action)`.

Initialized to zero. Updated via standard Q-learning:

```
Q(s, a) ← Q(s, a) + α × (r + γ × max_a' Q(s', a') − Q(s, a))
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `alpha` (learning rate) | 0.1 | Standard starting point for tabular Q-learning |
| `gamma` (discount factor) | 0.95 | Encourages multi-day planning (10-day horizon) |
| `epsilon_start` | 1.0 | Full exploration at start |
| `epsilon_end` | 0.01 | Near-greedy at convergence |
| `epsilon_decay` | 0.9995 | Decays over ~10,000 episodes |
| `episodes` | 15,000 | Sufficient for 960-state space convergence |

### Exploration

Epsilon-greedy: with probability `epsilon`, pick a random valid action; otherwise pick `argmax Q(s, ·)`. "Valid" means the action is consistent with the environment's forced-open constraint.

---

## Training & Evaluation (`rl/train.py`)

### Training Loop

```
for episode in range(episodes):
    state = env.reset()              # sample demand, reset DC status
    done = False
    episode_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        episode_reward += reward
    agent.decay_epsilon()
    log episode_reward every 100 episodes
```

### Evaluation

After training, run the learned greedy policy (ε=0) over 100 episodes with fresh demand samples. Report mean and std of total cost.

### Results Export

All results written to `results/` as CSVs:

| File | Contents |
|---|---|
| `results/milp_global.csv` | Global MILP: total cost, per-day DC decisions, per-day transport cost |
| `results/milp_daily.csv` | Daily myopic MILP: same schema, averaged over 10 simulation runs |
| `results/rl_policy.csv` | RL evaluation: same schema, mean over 100 episodes |
| `results/learning_curve.csv` | Episode number, episode reward (smoothed 100-ep rolling avg) |
| `results/rl_policy_table.csv` | Policy table: (day, demand_bucket) → most-chosen DC configuration |

---

## Comparison Notebook (`comparison.ipynb`)

Thin wrapper — all computation is in the Python package. The notebook:

1. Calls `run_all()` from `rl/train.py` to generate all results CSVs (or skips if CSVs exist)
2. Renders **cost comparison table**: MILP global vs. daily myopic vs. RL (mean ± std)
3. Renders **learning curve plot**: episode vs. smoothed cumulative reward
4. Renders **policy table**: what DC configuration RL chose per (day, demand level) — key insight into learned daily rules
5. Renders **cost gap analysis**: RL vs MILP global (%), RL vs daily myopic (%)

---

## Tests (`tests/`)

| Test | What it verifies |
|---|---|
| `test_environment.py` | State transitions, rolling window constraint enforcement, reward calculation |
| `test_agent.py` | Q-table update correctness, epsilon decay, action selection |
| `test_results.py` | CSV output schema, summary table formatting |

---

## README

Replaces the old notebook overview. Covers:
- Problem description (supply chain routing, multi-period horizon)
- Three approaches: MILP global, MILP daily myopic, Q-learning
- Quickstart: `uv sync && uv run python -m rl.train`
- Results summary (filled in after implementation)
- **Further Explorations** section:
  - DQN for larger state spaces (100+ DCs)
  - Multi-agent RL for multi-echelon networks
  - Constraint-aware RL via Lagrangian relaxation
  - Real-time reoptimization: RL as a warm-start for MILP

---

## What We Are Not Building

- No DQN or neural network approximation (noted in Further Explorations)
- No Gym registration (environment is standalone, not registered with `gym.make`)
- No hyperparameter search (fixed hyperparameters, documented rationale)
- No real demand data (synthetic distribution as in existing code)
- The notebook does not re-implement any computation — it only visualizes results from the package
