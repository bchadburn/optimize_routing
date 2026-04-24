# Routing Optimization: MILP · OR-Tools · cuOpt · RL

![CI](https://github.com/bchadburn/optimize_routing/actions/workflows/ci.yml/badge.svg)

Two complementary experiments comparing fundamentally different approaches to combinatorial
routing problems — from exact solvers to GPU-accelerated heuristics to reinforcement learning.

---

## Experiment 1 — Supply Chain: MILP vs Q-Learning

**Problem:** 10-day multi-echelon supply chain. 2 plants → 5 DCs → 12 customers.
Minimize DC opening costs + transportation costs over the planning horizon.

| Solver | Avg Cost | Gap |
|--------|----------|-----|
| MILP Global (full horizon) | $17,020 | — (optimal) |
| MILP Daily Myopic | $19,519 | +14.7% |
| Q-Learning (15k episodes) | $28,715 | +68.7% |

**Key finding:** The myopic MILP loses 14.7% by ignoring future demand. Q-learning is a
proof-of-concept of hierarchical RL decomposition — the agent handles binary DC-open
decisions while an LP optimally routes flow given those decisions.

```bash
uv run python -m rl.train          # reproduce results
jupyter lab comparison.ipynb       # full comparison + learning curve
```

---

## Experiment 2 — CVRP Benchmark: MILP · OR-Tools · cuOpt · DQN · Greedy

**Problem:** Capacitated Vehicle Routing (CVRP) — single depot, n customers in a 100×100 km
grid, vehicle capacity = 50 units. Minimize total route distance (km).

### Results

| n | MILP (300s) | OR-Tools (5s) | cuOpt (10s) | Greedy | DQN |
|---|-------------|--------------|-------------|--------|-----|
| 10 | **310 km** ✓ optimal | 310 km (+0%) | 310 km (+0%) | 372 km (+20%) | 553 km (+79%) |
| 20 | 460 km (29% opt-gap) | 460 km | **460 km** (−0.1%) | 615 km (+34%) | — |
| 30 | 602 km (63% opt-gap) | 583 km | **577 km** (−1.1%) | 798 km (+37%) | — |
| 50 | — | 845 km | **810 km** (−4.2%) | 1,182 km (+40%) | 2,724 km (+222%) |
| 100 | — | 1,532 km | **1,372 km** (−10.5%) | 1,963 km (+28%) | — |
| 250 | — | 3,353 km | **2,968 km** (−11.5%) | 3,917 km (+17%) | — |
| 500 | — | 6,476 km | **5,542 km** (−14.4%) | 6,941 km (+7%) | — |

✓ = proven globally optimal by CP-SAT

**Solve time** (mean per instance):

| n | MILP | OR-Tools | cuOpt | Greedy | DQN |
|---|------|----------|-------|--------|-----|
| 10 | 43s | 5s | 16s | <1ms | 5ms |
| 20 | 300s (timeout) | 5s | 10s | <1ms | — |
| 50 | — | 5s | 12s | 3ms | 120ms |
| 100 | — | 5s | 15s | 8ms | — |
| 250 | — | 5s | 19s | 60ms | — |
| 500 | — | 5s | 27s | 88ms | — |

OR-Tools is capped at its 5s time limit at every size. cuOpt grows from 10s (small, REST overhead dominated) to 27s (n=500, compute dominated). Greedy is 3–4 orders of magnitude faster than either.

### Key Findings

**MILP (CP-SAT with circuit constraints)**
- Proves global optimality at n≤10 in ~35–43s
- At n=20: times out after 300s with a 29% optimality gap, but the incumbent matches OR-Tools exactly — all three solvers are likely within 1–2% of true optimal
- The "opt-gap" measures the weakness of the lower bound, not solution quality
- Impractical beyond n=30 even with minutes of compute

**OR-Tools Guided Local Search**
- Near-optimal quality at all sizes tested
- Consistently strong baseline; no GPU required
- 5s time limit sufficient — marginal gains beyond that

**cuOpt (NVIDIA GPU)**
- Matches exact optimum at n=10
- Beats OR-Tools (5s) by 4–14% at n=50–500
- Advantage grows with problem size — GPU parallelism pays off at scale
- ~10–27s per instance (includes REST overhead for self-hosted API)
- Best choice when a GPU is available and n≥50

**DQN (flat MLP)**
- +79% above optimal at n=10; +222% at n=50
- Root cause: flat MLP lacks permutation invariance — input encoding changes when customer order changes, but the routing problem doesn't. The network learns position-specific patterns rather than routing geometry
- Production VRP RL requires attention-based models (Kool et al. 2019)

**Greedy (nearest-neighbor)**
- 0.1–130ms across all sizes — the only real-time option
- 7–40% above optimal depending on problem size
- Gap narrows at large n as geographic structure constrains solutions

### When to Use Each Solver

| Scenario | Choice |
|----------|--------|
| n ≤ 10, need optimality proof | MILP (CP-SAT) |
| n ≤ 50, no GPU | OR-Tools GLS |
| n ≥ 50, GPU available | cuOpt |
| n > 500 or latency < 100ms required | Greedy (or attention-model RL) |

```bash
# Run benchmark (requires cuOpt server for --cuopt flag)
uv run python -m vrp_benchmark.benchmark --counts 10 20 50 100 250 500 --milp --cuopt

# Train DQN (illustrates architecture limitations)
uv run python -m vrp_benchmark.train_dqn --n 10 --episodes 200000

# Full results and plots
jupyter lab vrp_benchmark/results.ipynb
```

### cuOpt Server Setup

```bash
# Requires Docker + NVIDIA GPU
docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:26.4.0-cuda12.9-py3.13

# Install REST client (works with Python 3.12+)
uv pip install --extra-index-url https://pypi.nvidia.com cuopt-sh-client
```

---

## Project Structure

```
optimizer/              OR-Tools MILP model (supply chain, Experiment 1)
rl/                     Supply chain RL environment, Q-learning agent, training
  solvers/              Pluggable CVRP solvers used in VRP sub-problem
vrp_benchmark/          Standalone CVRP benchmark (Experiment 2)
  data.py               CVRPInstance, generate_instance (100×100 km grid)
  benchmark.py          Benchmark runner → results/vrp_benchmark.csv
  train_dqn.py          DQN training CLI
  results.ipynb         Results notebook with plots and decision matrix
  solvers/
    milp.py             CP-SAT exact solver with circuit constraints + opt-gap reporting
    ortools_vrp.py      OR-Tools Guided Local Search
    cuopt_vrp.py        NVIDIA cuOpt via self-hosted REST API
    greedy.py           Nearest-neighbor baseline
    dqn.py              DQN flat-MLP (educational — not competitive)
results/                Benchmark plots (PNG)
tests/                  Pytest suite
```

## Quickstart

```bash
uv sync
uv run pytest tests/ -v
jupyter lab vrp_benchmark/results.ipynb
```

## Next Steps

- **Solomon benchmark instances** — Load standard C1/R1/RC1 instances (100 customers,
  published optimal solutions) to measure exact % above known optimum
- **Attention Model RL** — Replace flat-MLP DQN with Kool et al. (2019) encoder-decoder;
  achieves 1–3% above OR-Tools at n=100 with <10ms inference
- **Time windows (CVRPTW)** — Solomon instances include time windows; extend solvers
  to enforce them and rerun the same comparison
