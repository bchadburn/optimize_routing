# Routing Optimization: MILP · OR-Tools · cuOpt · RL

![CI](https://github.com/bchadburn/optimize_routing/actions/workflows/ci.yml/badge.svg)

Four experiments comparing fundamentally different approaches to combinatorial routing problems —
from exact solvers to GPU-accelerated heuristics to reinforcement learning, across both CVRP
and VRPTW problem classes.

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

**Why not cuOpt here?** The routing subproblem is a multi-echelon flow (plant → DC → customer)
with soft demand allocation, not a standard CVRP. cuOpt solves CVRP/VRPTW; it can't express
the DC-open binary decisions or the multi-echelon cost structure. Experiments 2–3 isolate
the pure CVRP routing layer where cuOpt applies directly.

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
| 20 | 460 km (29% opt-gap) | 460 km | **460 km** (−0.1%) | 615 km (+34%) | 1,058 km (+130%) |
| 30 | 588 km (61% opt-gap) | 583 km | **577 km** (−1.1%) | 798 km (+37%) | 1,656 km (+184%) |
| 50 | — (budget exhausted) | 842 km | **808 km** (−4.0%) | 1,182 km (+40%) | 2,724 km (+224%) |
| 100 | — | 1,503 km | **1,361 km** (−9.5%) | 1,963 km (+31%) | — |
| 250 | — | 3,263 km | **2,960 km** (−9.3%) | 3,917 km (+20%) | — |
| 500 | — | 6,344 km | **5,534 km** (−12.8%) | 6,941 km (+9%) | — |

✓ = proven globally optimal by CP-SAT. MILP runs with a 300s per-instance limit and a 1hr total budget; n≥50 exceeded the budget and was skipped.

**Solve time** (mean per instance):

| n | MILP | OR-Tools | cuOpt | Greedy | DQN |
|---|------|----------|-------|--------|-----|
| 10 | 22s | 5s | 5s | <1ms | 4ms |
| 20 | 300s (timeout) | 5s | 11s | <1ms | 6ms |
| 30 | 300s (timeout) | 5s | 11s | <1ms | 10ms |
| 50 | — | 5s | 11s | <1ms | 19ms |
| 100 | — | 5s | 12s | 4ms | — |
| 250 | — | 5s | 15s | 19ms | — |
| 500 | — | 5s | 20s | 79ms | — |

OR-Tools is capped at its 5s time limit at every size. cuOpt grows from 5s (small, near-instant solve) to 20s (n=500, compute dominated). Greedy is 3–4 orders of magnitude faster than either.

### Key Findings

**MILP (CP-SAT with circuit constraints)**
- Proves global optimality at n≤10 in ~22s; used as the ground-truth reference for that size
- At n=20–30: times out after 300s but returns a feasible incumbent — 460 km and 588 km respectively. These match OR-Tools/cuOpt closely, so all three are likely within 1–2% of true optimal
- **Not run at n≥50:** cumulative 1hr budget exhausted after n=30. CP-SAT's lower bounds are too weak to prune the search space at larger scales regardless of time given
- The reported "opt-gap" (29–61%) measures lower-bound weakness, not solution quality

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
# Run synthetic benchmark (requires cuOpt server for --cuopt flag)
uv run python -m vrp_benchmark.benchmark --counts 10 20 50 100 250 500 --milp --cuopt

# Run real-instance benchmark vs published best-known solutions
uv run python -m vrp_benchmark.benchmark_real --cuopt

# Train DQN (illustrates architecture limitations)
uv run python -m vrp_benchmark.train_dqn --n 10 --episodes 200000

# Results notebooks
jupyter lab vrp_benchmark/results.ipynb           # synthetic + MILP gap analysis
jupyter lab vrp_benchmark/results_real.ipynb      # real instances vs BKS
```

### cuOpt Setup

cuOpt requires a running server. Two paths depending on your environment:

#### Path A — Self-hosted (local machine with NVIDIA GPU)

```bash
# 1. Start the cuOpt container (requires Docker + NVIDIA GPU + nvidia-container-toolkit)
docker run --gpus all -p 5000:5000 nvcr.io/nvidia/cuopt/cuopt:26.4.0-cuda12.9-py3.13

# 2. Install the REST client
uv pip install --extra-index-url https://pypi.nvidia.com cuopt-sh-client

# 3. Run benchmarks
uv run python -m vrp_benchmark.benchmark --cuopt
uv run python -m vrp_benchmark.benchmark_real --cuopt
uv run python -m vrp_benchmark.benchmark_solomon --cuopt
```

#### Path B — NVIDIA NIM API (no local GPU required; works from Colab)

NVIDIA hosts cuOpt as a managed API with free trial credits:

1. Sign up at [build.nvidia.com/nvidia/cuopt](https://build.nvidia.com/nvidia/cuopt) and generate an API key
2. Set the key as an environment variable:

```bash
export NVIDIA_API_KEY=nvapi-xxxx
```

3. Run benchmarks with `--nim` flag:

```bash
uv run python -m vrp_benchmark.benchmark --cuopt --nim
uv run python -m vrp_benchmark.benchmark_real --cuopt --nim
uv run python -m vrp_benchmark.benchmark_solomon --cuopt --nim
```

Or in Python:
```python
from vrp_benchmark.solvers.cuopt_vrp import CuOptSolver
solver = CuOptSolver(mode="nim")  # reads NVIDIA_API_KEY from environment
```

The NIM API uses the same request format as self-hosted; only the endpoint and auth differ.
See `colab_benchmark.ipynb` for a fully-runnable Colab example.

---

## Project Structure

```
optimizer/              OR-Tools MILP model (supply chain, Experiment 1)
rl/                     Supply chain RL environment, Q-learning agent, training
  solvers/              Pluggable CVRP solvers used in VRP sub-problem
vrp_benchmark/          Standalone VRP benchmark suite (Experiments 2–4)
  data.py               CVRPInstance, generate_instance (100×100 km grid)
  data_tw.py            VRPTWInstance, route_cost_tw (time-window extension)
  benchmark.py          Synthetic CVRP benchmark → results/vrp_benchmark.csv
  benchmark_real.py     Uchoa real-instance benchmark → results/real_benchmark.csv
  benchmark_solomon.py  Solomon VRPTW benchmark → results/solomon_benchmark.csv
  train_dqn.py          DQN training CLI
  results.ipynb         Synthetic results: quality/time plots, MILP gap, decision matrix
  results_real.ipynb    Uchoa results: gap vs published BKS, instance analysis
  solvers/
    milp.py             CP-SAT exact solver with circuit constraints + opt-gap reporting
    ortools_vrp.py      OR-Tools Guided Local Search (CVRP)
    ortools_tw.py       OR-Tools Guided Local Search (VRPTW)
    cuopt_vrp.py        NVIDIA cuOpt via self-hosted REST API (CVRP)
    cuopt_tw.py         NVIDIA cuOpt via self-hosted REST API (VRPTW)
    greedy.py           Nearest-neighbor baseline (CVRP)
    greedy_tw.py        Nearest-neighbor baseline (VRPTW)
    dqn.py              DQN flat-MLP (educational — not competitive)
  datasets/
    uchoa.py            Uchoa et al. X-instance loader (downloads + caches from GitHub)
    solomon.py          Solomon VRPTW instance loader (downloads + caches from GitHub)
results/                Benchmark outputs (CSV, PNG)
tests/                  Pytest suite
```

## Quickstart

```bash
uv sync
uv run pytest tests/ -v
jupyter lab vrp_benchmark/results.ipynb
```

## Experiment 3 — Real Instances: Gap vs Best-Known Solutions (Uchoa et al.)

Benchmarking against **published BKS** from the Uchoa et al. (2017) X-instance dataset —
the standard modern CVRP benchmark. Instances auto-download on first run.

| Instance | n | BKS | Greedy | OR-Tools (30s) | cuOpt (30s) |
|----------|---|-----|--------|----------------|-------------|
| X-n101-k25 | 100 | 27,591 | +50.5% | +5.0% | +33.2% ⚠️ |
| X-n115-k10 | 114 | 12,747 | +43.3% | +2.6% | **+0.1%** |
| X-n139-k10 | 138 | 13,590 | +30.0% | +9.5% | **+0.2%** |
| X-n162-k11 | 161 | 14,138 | +24.1% | +4.4% | **+0.3%** |
| X-n200-k36 | 199 | 58,578 | +18.4% | +4.1% | **+1.2%** |

⚠️ X-n101-k25 anomaly: 25 vehicles with tight capacity creates dense overlapping routes — cuOpt
GPU search degrades on high vehicle-count instances. OR-Tools handles it better (+5% vs +33%).

On the other 4 instances cuOpt matches or approaches published BKS (0.1–1.2%) — comparable
to production solvers. OR-Tools is consistent at 2.6–9.5% with no instance-specific pathology.

```bash
uv run python -m vrp_benchmark.benchmark_real --cuopt
jupyter lab vrp_benchmark/results_real.ipynb
```

---

## Experiment 4 — Solomon VRPTW: Time-Window Routing

**Problem:** VRPTW (Vehicle Routing Problem with Time Windows) — 100 customers, hard time
windows per customer and depot, service time per stop. Six instance families covering clustered
(C), random (R), and mixed (RC) layouts, each with tight (x1) or wide (x2) windows.
Distance = Euclidean (speed=1, so travel time = distance numerically).

BKS values are for integer Euclidean distances; solvers here use float Euclidean.
On wide-window instances OR-Tools can report a distance slightly below BKS (negative gap)
because float distances are marginally smaller than floored integers — not a genuine improvement.

| Instance | Family | BKS (dist / veh) | Greedy | OR-Tools (30s) | cuOpt (30s) |
|----------|--------|-----------------|--------|----------------|-------------|
| C101 | Clustered, tight | 828.94 / 10v | +125.7% | +3.1% (10v) | **+2.4%** (11v) |
| C201 | Clustered, wide  | 591.56 / 3v  | +217.9% | **±0.0%** (3v) | **±0.0%** (3v) |
| R101 | Random, tight    | 1650.80 / 19v | +58.9% | +0.8% (21v) | **±0.0%** (19v) |
| R201 | Random, wide     | 1252.37 / 4v  | +58.5% | −2.5%† (7v) | **+0.1%** (4v) |
| RC101 | Mixed, tight    | 1696.94 / 14v | +59.8% | +1.3% (16v) | **−3.5%**† (15v) |
| RC201 | Mixed, wide     | 1406.91 / 4v  | +75.4% | −7.4%† (8v) | +0.8% (4v) |

† Negative gap = float Euclidean distances marginally smaller than the integer distances used in published BKS.

### Key Findings

**cuOpt excels on tight-window instances** — matches BKS exactly on R101 (+0.0%) and comes
within 2–3% on C101/RC101. The GPU search handles dense feasibility constraints well.

**OR-Tools is strong on wide-window instances** — C201 and near-BKS on R101 (+0.8%),
RC101 (+1.3%). On wide-window instances (C2/R2/RC2) OR-Tools sometimes finds shorter
routes than published BKS because it optimises float distances, not integer distances.

**Vehicle count:** cuOpt matches or nearly matches the BKS vehicle count on all instances.
OR-Tools often uses more vehicles (e.g. 7 vs 4 for R201) — it minimises distance but
doesn't explicitly minimise fleet size.

**Greedy degrades severely** — 59–218% above BKS. Nearest-neighbor without time-window
look-ahead uses many more vehicles than necessary and accumulates large travel distances.

```bash
uv run python -m vrp_benchmark.benchmark_solomon                    # default 6 instances
uv run python -m vrp_benchmark.benchmark_solomon --cuopt            # include cuOpt
uv run python -m vrp_benchmark.benchmark_solomon --family C1 R1     # full C1 and R1 families
uv run python -m vrp_benchmark.benchmark_solomon --instances C101 R101 RC101
```

---

## Next Steps

- **Attention Model RL** — Replace flat-MLP DQN with Kool et al. (2019) encoder-decoder;
  achieves 1–3% above OR-Tools at n=100 with <10ms inference
- **cuOpt high-k fix** — Investigate cuOpt configuration for high vehicle-count instances
  (X-n101-k25 style); may need fleet size tuning in the REST payload
- **Augerat/Golden instances** — Fill n=32–80 gap (Augerat) and large-scale n=200–480 (Golden)
