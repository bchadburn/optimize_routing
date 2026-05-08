"""PyMC time-series model for per-customer demand with hierarchical pooling.

Model structure (per customer ``i``, day ``t``):

    intercept_i  = mu_pop + sigma_pop * z_intercept_i           # non-centered
    trend_i      = mu_trend + sigma_trend * z_trend_i           # non-centered
    sigma_i      = abs(mu_sigma + sigma_sigma * z_sigma_i)      # non-centered HalfNormal-ish
    fourier_t    = beta_sin * sin(2*pi*t/7) + beta_cos * cos(2*pi*t/7)   # fully pooled
    log_mu(t,i)  = intercept_i + trend_i * t + fourier_t
    y(t,i)       ~ LogNormal(log_mu(t,i), sigma_i)

Why this shape:

- **Hierarchical-non-centered on level/trend/noise.** Per-customer level and
  trend genuinely vary; partial pooling shrinks low-history customers toward
  the population while letting high-history customers be themselves. Non-
  centered parameterization (Matt trick) avoids the NUTS funnel pathology when
  ``sigma_pop`` is small.
- **Fully pooled weekly Fourier.** Week-of-week patterns (weekday vs. weekend)
  are usually shared across customers in the same business. One pair of
  coefficients keeps the model identifiable on small data and matches what
  FICO's Ontario aggregate model did with its single Fourier component.
- **LogNormal likelihood.** Demand is positive-valued; this matches FICO's
  Ontario IESO model and the synthetic generative process.

If NUTS reports trouble (divergences > 1%, R-hat > 1.01, or ESS < 400 on any
parameter), swap the hierarchical layer for independent per-customer fits via
``build_independent_model``. The interface (a ``pymc.Model``) is identical, so
``forecast.py`` doesn't change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

# Weekly Fourier period in days (used by both training and out-of-sample).
WEEKLY_PERIOD_DAYS = 7.0


def _weekly_design(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sin/cos pair for a 7-day cycle. Returns ``(sin_term, cos_term)``."""
    phase = 2.0 * np.pi * t / WEEKLY_PERIOD_DAYS
    return np.sin(phase), np.cos(phase)


def build_hierarchical_model(history: pd.DataFrame) -> pm.Model:
    """Build the hierarchical-non-centered demand model from a long-form history.

    Args:
        history: DataFrame with columns ``[date, customer_id, demand]``.

    Returns:
        A ``pymc.Model`` ready for ``pm.sample()``. Coords are exposed so
        ``pm.set_data`` + ``pm.sample_posterior_predictive`` can re-run the
        model on out-of-sample dates.
    """
    df = history.sort_values(["customer_id", "date"]).reset_index(drop=True)
    customers = sorted(df["customer_id"].unique().tolist())
    dates     = sorted(df["date"].unique())
    n_cust    = len(customers)
    n_days    = len(dates)

    # Reshape to (day, customer) for vectorized broadcasting in PyMC.
    pivot = df.pivot(index="date", columns="customer_id", values="demand").to_numpy()
    assert pivot.shape == (n_days, n_cust), f"unexpected pivot shape {pivot.shape}"

    t = np.arange(n_days, dtype=float)
    sin_t, cos_t = _weekly_design(t)

    coords = {"customer": customers, "date": [str(d) for d in dates]}

    with pm.Model(coords=coords) as model:
        # ── Mutable data so out-of-sample forecasting can swap dates ─────────
        trend_data    = pm.Data("trend_data",    t,        dims="date")
        sin_data      = pm.Data("sin_data",      sin_t,    dims="date")
        cos_data      = pm.Data("cos_data",      cos_t,    dims="date")
        y_data        = pm.Data("y_data",        pivot,    dims=("date", "customer"))

        # ── Population-level priors for level / trend ───────────────────────
        mu_pop      = pm.Normal("mu_pop",       mu=4.0, sigma=1.0)
        sigma_pop   = pm.HalfNormal("sigma_pop", sigma=0.5)

        # Trend priors are intentionally tight: the synthetic generative
        # process uses ``trend_scale=0.0005`` per day (see synthesize.py), so
        # ``sigma=0.001`` here gives the model headroom of ~2x without
        # admitting unrealistically steep growth that would dominate a 10-day
        # forecast horizon. Loosen if applying to longer horizons or noisier
        # real-world data where seasonal trend is genuinely larger.
        mu_trend    = pm.Normal("mu_trend",     mu=0.0, sigma=0.001)
        sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.001)

        # ── Non-centered per-customer level + trend ─────────────────────────
        # Intercept is the load-bearing level term so we use the Matt trick to
        # avoid the NUTS funnel when sigma_pop is small. Trend follows the same
        # pattern.
        z_intercept = pm.Normal("z_intercept", 0.0, 1.0, dims="customer")
        z_trend     = pm.Normal("z_trend",     0.0, 1.0, dims="customer")

        intercept = pm.Deterministic(
            "intercept", mu_pop + sigma_pop * z_intercept, dims="customer"
        )
        trend = pm.Deterministic(
            "trend", mu_trend + sigma_trend * z_trend, dims="customer"
        )

        # ── Per-customer noise: hierarchical HalfNormal ──────────────────────
        # HalfNormal's bounded support makes the funnel pathology far milder than
        # for unbounded Normals, so we keep this centered: each customer's noise
        # scale is HalfNormal with a learned population scale. Avoids needing
        # softplus (not in pm.math) or abs (not strictly half-normal).
        mu_sigma   = pm.HalfNormal("mu_sigma", sigma=0.5)
        sigma_cust = pm.HalfNormal("sigma_cust", sigma=mu_sigma, dims="customer")

        # ── Fully pooled weekly Fourier ──────────────────────────────────────
        beta_sin = pm.Normal("beta_sin", mu=0.0, sigma=0.25)
        beta_cos = pm.Normal("beta_cos", mu=0.0, sigma=0.25)
        fourier  = beta_sin * sin_data + beta_cos * cos_data  # shape (n_days,)

        # ── Linear predictor and likelihood ──────────────────────────────────
        # log_mu has shape (n_days, n_customers): per-customer level/trend +
        # shared Fourier broadcast across customers.
        log_mu = (
            intercept[None, :]                        # (1, n_cust)
            + trend[None, :] * trend_data[:, None]    # (n_days, n_cust)
            + fourier[:, None]                        # (n_days, 1)
        )

        pm.LogNormal(
            "y",
            mu=log_mu,
            sigma=sigma_cust[None, :],
            observed=y_data,
            dims=("date", "customer"),
        )

    return model


def build_independent_model(history: pd.DataFrame) -> pm.Model:
    """Fallback: per-customer independent fits, no pooling.

    Used when NUTS diagnostics on the hierarchical model fail (divergences
    > 1%, R-hat > 1.01, or ESS < 400 on any parameter). Same return type so
    the rest of the pipeline (NetCDF caching, posterior predictive) doesn't
    change. Trades information sharing for sampler stability.
    """
    df = history.sort_values(["customer_id", "date"]).reset_index(drop=True)
    customers = sorted(df["customer_id"].unique().tolist())
    dates     = sorted(df["date"].unique())
    pivot     = df.pivot(index="date", columns="customer_id", values="demand").to_numpy()

    t = np.arange(len(dates), dtype=float)
    sin_t, cos_t = _weekly_design(t)
    coords = {"customer": customers, "date": [str(d) for d in dates]}

    with pm.Model(coords=coords) as model:
        trend_data    = pm.Data("trend_data",    t,        dims="date")
        sin_data      = pm.Data("sin_data",      sin_t,    dims="date")
        cos_data      = pm.Data("cos_data",      cos_t,    dims="date")
        y_data        = pm.Data("y_data",        pivot,    dims=("date", "customer"))

        # Independent per-customer params with weak priors — no population pool
        intercept  = pm.Normal("intercept",  mu=4.0, sigma=1.0,    dims="customer")
        trend      = pm.Normal("trend",      mu=0.0, sigma=0.001,  dims="customer")
        sigma_cust = pm.HalfNormal("sigma_cust", sigma=0.5,        dims="customer")

        beta_sin = pm.Normal("beta_sin", mu=0.0, sigma=0.25)
        beta_cos = pm.Normal("beta_cos", mu=0.0, sigma=0.25)
        fourier  = beta_sin * sin_data + beta_cos * cos_data

        log_mu = (
            intercept[None, :]
            + trend[None, :] * trend_data[:, None]
            + fourier[:, None]
        )

        pm.LogNormal(
            "y", mu=log_mu, sigma=sigma_cust[None, :],
            observed=y_data, dims=("date", "customer"),
        )

    return model
