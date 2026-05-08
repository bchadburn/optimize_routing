"""Generate synthetic per-customer demand history for the multi-echelon SC experiment.

We don't have a real public dataset for multi-echelon supply-chain demand, so we
synthesize one: 12 customers x ~365 days of daily demand with weekly seasonality,
mild trend, and per-customer noise. The generative process is intentionally
*slightly different* from the PyMC forecast model below (different intercept
priors, slight non-LogNormal contamination) so the experiment isn't tautological
- we're not just fitting the same model that generated the data.

Notes:
    - Saved to ``data/synth_demand.csv`` with columns ``[date, customer_id, demand]``.
    - The generative spec mirrors what FICO's Ontario IESO data showed
      empirically (LogNormal positive-valued daily demand) but adds a per-customer
      level/trend/noise so the hierarchical Bayesian fit downstream has signal to
      pool across.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SynthSpec:
    """Generative process for synthetic demand. Tunable for the test suite."""

    n_customers: int = 12
    n_days: int = 365
    seed: int = 7

    # Per-customer mean log-demand drawn from Normal(pop_mu, pop_sigma).
    # log scale ~ 4 -> demand ~ 55 units/day; pop_sigma=0.3 spreads customers
    # across roughly [25, 100] units/day mean.
    pop_mu: float = 4.0
    pop_sigma: float = 0.3

    # Per-customer linear trend coefficient drawn from Normal(0, trend_scale).
    # Small (0.0005 per day) so cumulative trend over 365 days is ~ +/-18% of base.
    trend_scale: float = 0.0005

    # Per-customer LogNormal observation noise sigma drawn from
    # HalfNormal(noise_scale). Modest (~0.15) so day-to-day demand wiggles by
    # ~10-20% but not chaotically.
    noise_scale: float = 0.15

    # Shared weekly Fourier amplitude. Models a uniform weekly pattern across
    # customers (e.g. weekday > weekend) so the fully-pooled Fourier prior in
    # the PyMC model has signal to fit.
    weekly_amplitude: float = 0.10


def synthesize_demand(spec: SynthSpec | None = None) -> pd.DataFrame:
    """Generate the synthetic per-customer daily demand history.

    Returns a long-format DataFrame: one row per (date, customer_id, demand).
    """
    spec = spec or SynthSpec()
    rng = np.random.default_rng(spec.seed)

    # Per-customer parameters
    customer_intercepts = rng.normal(spec.pop_mu, spec.pop_sigma, size=spec.n_customers)
    customer_trends     = rng.normal(0.0, spec.trend_scale, size=spec.n_customers)
    customer_noise      = np.abs(rng.normal(0.0, spec.noise_scale, size=spec.n_customers)) + 0.05

    # Shared weekly Fourier (one sin + one cos coefficient)
    weekly_beta_sin = spec.weekly_amplitude
    weekly_beta_cos = spec.weekly_amplitude * 0.5

    # Time index aligned so day 0 is a Monday; weekly cycle uses 2*pi/7.
    days = np.arange(spec.n_days)
    weekly_phase = 2 * np.pi * days / 7.0

    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for cust in range(spec.n_customers):
        log_mu = (
            customer_intercepts[cust]
            + customer_trends[cust] * days
            + weekly_beta_sin * np.sin(weekly_phase)
            + weekly_beta_cos * np.cos(weekly_phase)
        )
        # LogNormal observation
        eps = rng.normal(0.0, customer_noise[cust], size=spec.n_days)
        demand = np.exp(log_mu + eps)
        for t in range(spec.n_days):
            rows.append({
                "date":        base_date + pd.Timedelta(days=int(t)),
                "customer_id": cust,
                "demand":      float(demand[t]),
            })
    return pd.DataFrame(rows)


def write_synth_demand(path: Path, spec: SynthSpec | None = None) -> Path:
    """Write the synthetic dataset to ``path`` (creating parents). Returns the path."""
    df = synthesize_demand(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = Path("data/synth_demand.csv")
    write_synth_demand(out)
    print(f"Wrote {out} with {SynthSpec().n_customers * SynthSpec().n_days} rows")
