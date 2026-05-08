# Bayesian + Robust Optimization for Multi-Echelon Supply Chain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third-and-fourth methodology track to Experiment 1 (multi-echelon supply chain) — Bayesian demand forecasting (PyMC) → Robust optimization (chance-constrained MILP and CVaR scenario-MILP) — and benchmark them head-to-head against the existing global/myopic MILP and Q-learning baselines. The result table goes from "deterministic vs RL" to "deterministic vs RL vs probabilistic-robust," which is the full landscape of how practitioners actually handle stochastic demand.

**Reference implementation:** `fico-xpress/xpress-community/StochasticEnergyPlanning/stochastic_energy_planning.py`. We mirror their PyMC + Rockafellar-Uryasev structure but swap the energy-dispatch LP for our existing multi-echelon MILP (which has binary DC-open variables, so the resulting models are MILPs not LPs).

**Architecture:** Four phases, **landed in two PRs** for review tractability:

- **PR 1 (this branch):** Phases 1 + 2 — Bayesian forecast + chance-constrained MILP. Uses the existing global MILP unchanged; only the per-day-per-customer demand input changes. The cheapest, most commonly-used robust methodology — gets you 80% of the benefit at 0% of the formulation overhead.
- **PR 2 (follow-up):** Phases 3 + 4 — CVaR scenario-MILP via Rockafellar-Uryasev + README integration. Larger lift: the existing model has a hard equality `sum_d v_transport_d_to_c == p_customer_demand`, so CVaR requires (a) relaxing that to inequality + shortage slack, (b) adding scenario-indexed `shortage[ω,i,t]` / `aux[ω]` / `var_threshold` variables through the OR-Tools wrapper, (c) modifying the objective. Self-contained as a separate PR.

**Tech Stack:** Python 3.12, uv, PyMC 5+, ArviZ, NumPy, pandas, xarray + netCDF4. MILP backend is OR-Tools `pywraplp` with SCIP (already used by Experiment 1). No FICO Xpress dependency — Xpress is what the reference uses; the methodology is solver-agnostic.

---

## Why this exists

Experiment 1's current finding:

> *"The myopic MILP loses 14.7% by ignoring future demand. Q-learning is a proof-of-concept of hierarchical RL decomposition."*

That 14.7% gap is the demand-uncertainty gap. Three ways to close it:

1. **Q-learning (existing)** — model-free, learns from sim. Currently +68.7% over global MILP. Not the right tool.
2. **CFA (forthcoming, from seminar)** — deterministic MILP + learnable penalty params, sim-based parameter update. Cheap; works without a forecast.
3. **Bayesian + Robust (this plan)** — explicit probabilistic forecast + a robust optimizer that respects it. More principled when a good posterior is available.

CFA and Bayesian+Robust are complementary, not competing. CFA absorbs uncertainty empirically; Bayesian+Robust models it explicitly. Having both in the repo makes the trade-off concrete.

---

## File Structure

| File | Purpose |
|---|---|
| `bayesian_forecast/__init__.py` | Package init |
| `bayesian_forecast/model.py` | PyMC time-series model (intercept + trend + Fourier seasonality + LogNormal noise). Per-customer fits or hierarchical pool — TBD in Phase 1. |
| `bayesian_forecast/forecast.py` | `forecast_demand()` — runs/caches NUTS sampling, returns `(scenarios, n_days, n_customers)` ndarray |
| `bayesian_forecast/diagnostics.py` | Posterior predictive plots, HDI bands, skewness check (parallels FICO's 2x2 diagnostic figure) |
| `vrp_benchmark/solvers/chance_constrained_milp.py` | `solve_chance_constrained()` — wraps existing MILP, replaces stochastic demand with per-customer 95th-percentile vector. Pure deterministic MILP. |
| `vrp_benchmark/solvers/cvar_milp.py` | `solve_cvar()` — scenario MILP with Rockafellar-Uryasev linearization. Adds `shortage[ω,i,t]`, `aux[ω]`, `var_threshold` decision vars. |
| `vrp_benchmark/utils/shortage.py` | `monte_carlo_shortage_rate()` — model-agnostic evaluator: Pr(unmet demand) under N draws from posterior |
| `vrp_benchmark/utils/pareto.py` | `pareto_sweep()` — sweep `cvar_weight` across {1, 2, 5, 10, ..., 1000} and plot cost-vs-shortage frontier |
| `tests/test_bayesian_forecast.py` | NetCDF cache hit, posterior shape, NUTS divergences below threshold |
| `tests/test_chance_constrained.py` | Percentile reduction matches `np.percentile(scenarios, 95, axis=0)` exactly; MILP accepts the demand vector unchanged |
| `tests/test_cvar.py` | Rockafellar-Uryasev objective is correct; with `cvar_weight=0` it degenerates to the deterministic MILP; with `cvar_weight=∞` it minimises shortage at any cost |
| `tests/test_pareto.py` | Frontier is monotone (higher weight ⇒ lower shortage, higher cost) within sampling noise |
| `data/ontario_supply_chain_demand_2024_2025.csv` | OR equivalent — synthetic multi-echelon SC daily demand, generated via `bayesian_forecast/synthesize.py` so the experiment is reproducible without proprietary data |
| `results/bayesian_robust_summary.csv` | Final benchmark row for the README table: `(method, avg_cost, gap_vs_oracle, shortage_rate, runtime_s)` |
| `docs/methodology/bayesian-robust.md` | User-facing methodology doc — links into existing README |

---

## Phase 1 — Bayesian forecast layer

### 1.1 Synthesize multi-echelon demand history

The FICO example has Ontario IESO data. We need an analogous synthetic dataset for multi-echelon SC demand: 12 customers × 365+ days of historical demand with weekly periodicity, mild trend, and customer-specific noise. Tunable so the seasonality is real but not overfit.

- [ ] Write `bayesian_forecast/synthesize.py` — generate per-customer time series with intercept + linear trend + weekly Fourier + LogNormal noise. Save to `data/synth_demand.csv`.
- [ ] Document the generative process so a reviewer can verify the experiment isn't tautological (we're not fitting the same model that generated the data; we add slight model misspecification to be honest).

### 1.2 PyMC model

Mirror the FICO script's structure (`stochastic_energy_planning.py:147-352`):

- [ ] `mu(t,i) = intercept_i + trend_i · t + fourier_weekly(t)` per customer
- [ ] `y(t,i) ~ LogNormal(mu(t,i), sigma_i)`
- [ ] Hierarchical priors over customers: `intercept_i ~ Normal(mu_pop, sigma_pop)` so we share statistical strength
- [ ] NUTS: 4 chains × 1000 draws, target accept = 0.9
- [ ] Out-of-sample posterior predictive for the 10-day evaluation horizon (`pm.set_data` + `pm.sample_posterior_predictive`)
- [ ] Cache to `data/forecast.nc` (NetCDF), skip sampling on cache hit

### 1.3 Diagnostics

- [ ] 2x2 figure: raw history, posterior predictive check on training data, forecast on eval horizon, per-day skewness
- [ ] NUTS divergence check (fail loudly if > 5%)

**Deliverable:** `forecast_demand()` returns `(n_scenarios=4000, n_days=10, n_customers=12)` ndarray. NetCDF cached. Diagnostic PNG written to `results/`.

---

## Phase 2 — Chance-constrained MILP (Approach 1)

### 2.1 Reduce scenarios to per-day-per-customer 95th percentile

Mirrors FICO `solve_chance_constrained` at line 368:

```python
demand_pct = np.percentile(demand_scenarios, 95, axis=0)  # (n_days, n_customers)
```

This is the cheapest possible upgrade over the deterministic MILP — collapses 4000 scenarios into a single demand vector and reuses the existing solver unchanged.

- [ ] `solve_chance_constrained(demand_scenarios, **existing_milp_kwargs)` — wraps the existing MILP solver
- [ ] No new constraints, no new variables. The robustness comes entirely from the percentile choice on the input.
- [ ] Validate via Monte Carlo: simulate the 4000 actual scenarios, count days where chosen plan can't meet demand. Should be ≤ 5%.

### 2.2 Tests

- [ ] Reduces correctly to deterministic when all 4000 scenarios are identical
- [ ] Shortage rate ≤ 5% on the held-out scenario set
- [ ] Cost is higher than deterministic MILP using the posterior mean (we're paying for robustness)

**Deliverable:** Drop-in solver, validated on synthetic data. README row added.

---

## Phase 3 — CVaR scenario-MILP (Approach 2)

### 3.1 Rockafellar-Uryasev formulation

Mirrors FICO `solve_cvar` at line 481:

- [ ] Sample 100 scenarios from the 4000-scenario posterior (or fewer if the MILP scales — energy LP did 1000; our MILP with binary DC-open will be slower)
- [ ] Add decision variables:
  - `shortage[ω, i, t] ≥ 0` for each scenario × customer × day
  - `aux[ω] ≥ 0`
  - `var_threshold ∈ ℝ` (z)
- [ ] Add constraints:
  - `shortage[ω,i,t] ≥ demand[ω,i,t] - delivered[i,t]` (linearizes max(0, ·))
  - `aux[ω] ≥ shortage_penalty · sum_{i,t} shortage[ω,i,t] - var_threshold`
- [ ] Objective: `existing_cost_terms + cvar_weight · (var_threshold + (1/(N·alpha)) · sum_ω aux[ω])`
- [ ] Keep binary DC-open and continuous flow vars exactly as today — only the shortage layer is new

### 3.2 Pareto sweep

- [ ] Sweep `cvar_weight ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}` (same as FICO)
- [ ] For each weight, solve and record `(cost, shortage_rate)`
- [ ] Plot Pareto frontier with chance-constrained as anchor point (yellow star, like FICO does)

### 3.3 Tests

- [ ] `cvar_weight=0` → degenerate to deterministic MILP using mean demand
- [ ] `cvar_weight=very_high` → shortage_rate → 0 with cost climbing
- [ ] Frontier monotone within Monte Carlo noise

**Deliverable:** CVaR solver, Pareto plot, README row.

---

## Phase 4 — Comparison & README integration

- [ ] Update `comparison.ipynb` to include the new methodologies
- [ ] Add to README's Experiment 1 table:

| Solver | Avg Cost | Gap vs oracle | Shortage rate | Compute |
|--------|----------|---------------|---------------|---------|
| MILP Global (full horizon) | $17,020 | — | 0% (oracle) | — |
| MILP Daily Myopic | $19,519 | +14.7% | — | low |
| **Chance-Constrained MILP (95th)** | **TBD** | **TBD** | **≤5% by construction** | **low** |
| **CVaR MILP (Rockafellar-Uryasev)** | **TBD (Pareto)** | **TBD** | **TBD** | **moderate** |
| Q-Learning (15k episodes) | $28,715 | +68.7% | — | very high |

- [ ] Document the methodology trade-off matrix in `docs/methodology/bayesian-robust.md`
- [ ] Cross-reference the FICO blog post (give credit) — but make clear our implementation is independent and uses open-source solvers

---

## Background: why the comparison is fair

The FICO example uses an LP because energy dispatch is continuous. Our supply chain has binary DC-open decisions, so:

- **Chance-constrained version stays a MILP** of the same complexity as today's Experiment 1 MILP. Just the demand input changes. Cheap.
- **CVaR version is a larger MILP** because the shortage layer adds N×T×C continuous variables per scenario set. This is the cost of explicit tail-risk modeling. Should still solve at our problem size (5 DCs × 12 customers × 10 days × 100 scenarios = ~60k shortage vars + 100 aux + 1 z).

Both methodologies are honest — they use the *actual* posterior to evaluate, not just to inform. The Q-learning baseline doesn't have access to a posterior at all; the CFA approach (forthcoming) skips the forecast entirely. This plan adds the "we did model the uncertainty probabilistically" data point.

---

## Out of scope

- **CFA (Cost Function Approximation) implementation.** Tracked separately — that's the seminar's primary thread. The two methodologies will share `bayesian_forecast/` only as a *contrast*: CFA explicitly does not use the posterior; Bayesian+Robust does.
- **GPU acceleration of the CVaR MILP.** cuOpt doesn't natively express scenario-CVaR; this stays CPU MILP.
- **Robust optimization with budgeted uncertainty sets.** A third paradigm (Bertsimas/Sim) — interesting but out of scope; we're picking chance-constrained + CVaR as the two most teachable approaches.
- **Inventory-coupled chance constraints** (joint chance constraints over multiple days). Single-period for now.

---

## Locked design decisions

1. **Customer pooling in the Bayesian model: hierarchical-non-centered on level/trend/noise, fully pooled on Fourier weekly seasonality.**
   - Per-customer: `intercept_i = μ_pop + σ_pop · z_i` (non-centered Matt trick to avoid the NUTS funnel); same shape for `trend_i` and `sigma_i` (the per-customer noise).
   - Shared across customers: `fourier_betas_weekly` (one pair of sin/cos coefficients used by all customers — week-of-week patterns are usually the same across a single business's customer base; matches what FICO did with their single-series Ontario model).
   - **Fallback:** if NUTS reports divergences > 1%, R-hat > 1.01 on any parameter, or ESS < 400 on any parameter, swap the hierarchical layer for independent per-customer fits. The call boundary is `forecast_demand() → ndarray` so the model internals don't leak — fallback is a one-file swap.
2. **Scenario count for CVaR: start at 100, profile, scale up.** FICO used 1000 LP-only; our MILP at 5 DCs × 12 customers × 10 days × 1000 scenarios = 600k shortage vars, which is large for an open-source MILP. 100 keeps the MILP tractable while still tracing a meaningful Pareto frontier; revisit only if the frontier looks noisy.
3. **Shortage penalty: 10× the marginal cost of the most expensive supply path, parameterizable.** FICO used $500/MWh. SC analogue = lost-sale penalty per unit of unmet demand. Domain-anchored to the existing cost matrix in Experiment 1 so the penalty scales with the rest of the problem rather than being an arbitrary number.

---

## References

- Powell, W.B. (2022) *Reinforcement Learning and Stochastic Optimization* — CFA framework
- Rockafellar, R.T. & Uryasev, S. (2000) "Optimization of Conditional Value-at-Risk" — the LP linearization used here
- FICO Xpress Community blog: *Getting Started with Optimization Under Uncertainty Using PyMC and FICO Xpress* (Vieira & Saunders, 2026)
- IESO Power Data Directory — Ontario electricity demand reference dataset (FICO example only; we synthesize for SC)
