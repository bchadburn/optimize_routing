"""Run NUTS sampling, produce out-of-sample posterior predictive demand scenarios,
cache to NetCDF.

Public entry point: ``forecast_demand(history, horizon_days, cache_path)`` returns
``(scenarios, customers)`` where ``scenarios`` is shape ``(n_scenarios, horizon, n_customers)``.

The cache key is the ``cache_path`` itself - delete the file to force a re-fit.
That mirrors FICO's ``stochastic_energy_planning.forecast()`` which uses
``NC_FILE.exists()`` as the only cache invariant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from bayesian_forecast.model import (
    _weekly_design,
    build_hierarchical_model,
    build_independent_model,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplerSpec:
    """NUTS knobs. Defaults match FICO's reference (4 chains x 1000 draws)."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 7


@dataclass(frozen=True)
class DiagnosticResult:
    """Output of ``check_diagnostics``. Used to decide whether to fall back."""

    divergence_rate: float
    max_r_hat: float
    min_ess: float
    healthy: bool


def check_diagnostics(idata: xr.Dataset, threshold_div: float = 0.01,
                      threshold_r_hat: float = 1.01,
                      threshold_ess: float = 400.0) -> DiagnosticResult:
    """Compute a quick health score from an ArviZ inference data object.

    Returns a ``DiagnosticResult`` with ``healthy`` set when all thresholds pass.
    Documented thresholds:

    - divergence_rate <= 1%
    - max R-hat <= 1.01
    - min ESS >= 400

    These are the canonical PyMC defaults; failing any of them is a strong
    signal to fall back to ``build_independent_model``.
    """
    import arviz as az

    n_total = int(idata.posterior.sizes["chain"]) * int(idata.posterior.sizes["draw"])
    divergent = int(idata.sample_stats.diverging.sum().item())
    div_rate = divergent / max(n_total, 1)

    summary = az.summary(idata, kind="diagnostics")
    max_r_hat = float(summary["r_hat"].max())
    min_ess   = float(summary["ess_bulk"].min())

    healthy = (
        div_rate <= threshold_div
        and max_r_hat <= threshold_r_hat
        and min_ess >= threshold_ess
    )
    return DiagnosticResult(
        divergence_rate=div_rate,
        max_r_hat=max_r_hat,
        min_ess=min_ess,
        healthy=healthy,
    )


def _sample(model: pm.Model, sampler: SamplerSpec) -> xr.Dataset:
    """Run NUTS, return an ArviZ inference data."""
    with model:
        idata = pm.sample(
            draws=sampler.draws,
            tune=sampler.tune,
            chains=sampler.chains,
            target_accept=sampler.target_accept,
            random_seed=sampler.seed,
            progressbar=False,
        )
    return idata


def _posterior_predictive_for_horizon(
    model: pm.Model,
    idata: xr.Dataset,
    n_train_days: int,
    horizon_days: int,
    n_customers: int,
    forecast_start: pd.Timestamp,
    sampler: SamplerSpec,
) -> xr.Dataset:
    """Run posterior predictive on out-of-sample dates by swapping ``pm.set_data``.

    Mirrors the FICO pattern: keep the same model, re-point the trend/sin/cos/
    y_data buffers to the future window, sample posterior predictive.
    """
    t_future = np.arange(n_train_days, n_train_days + horizon_days, dtype=float)
    sin_future, cos_future = _weekly_design(t_future)
    blank_y = np.zeros((horizon_days, n_customers), dtype=float)

    future_dates = pd.date_range(forecast_start, periods=horizon_days, freq="D")
    new_coords = {"date": [str(d) for d in future_dates]}

    with model:
        pm.set_data(
            new_data={
                "trend_data": t_future,
                "sin_data":   sin_future,
                "cos_data":   cos_future,
                "y_data":     blank_y,
            },
            coords=new_coords,
        )
        forecast_idata = pm.sample_posterior_predictive(
            idata,
            random_seed=sampler.seed,
            progressbar=False,
        )
    return forecast_idata


def forecast_demand(
    history: pd.DataFrame,
    horizon_days: int,
    cache_path: Path,
    forecast_start: pd.Timestamp | None = None,
    sampler: SamplerSpec | None = None,
    force: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Run the Bayesian forecast pipeline. Cached as NetCDF on first run.

    Args:
        history: long-form demand history with columns ``[date, customer_id, demand]``.
        horizon_days: number of out-of-sample days to forecast.
        cache_path: NetCDF path. If the file exists and ``force=False``, the
            posterior predictive samples are loaded from disk and NUTS is not
            re-run.
        forecast_start: first date of the out-of-sample forecast. Defaults to
            ``max(history.date) + 1 day``.
        sampler: NUTS configuration; defaults to ``SamplerSpec()``.
        force: when True, re-fit even if the cache exists.

    Returns:
        ``(scenarios, customers)`` where ``scenarios`` is shape
        ``(n_scenarios, horizon_days, n_customers)`` and ``customers`` is the
        list of customer ids in the column order of the last axis.
    """
    sampler = sampler or SamplerSpec()
    cache_path = Path(cache_path)

    customers = sorted(history["customer_id"].unique().tolist())
    n_customers = len(customers)

    if cache_path.exists() and not force:
        log.info("Loading cached posterior predictive from %s", cache_path)
        with xr.open_dataset(cache_path, engine="netcdf4") as ds:
            y_samples = ds["y"].values  # (chain, draw, day, customer)
        # Reshape (chain, draw, day, cust) -> (chain*draw, day, cust)
        scenarios = y_samples.reshape(-1, y_samples.shape[-2], y_samples.shape[-1])
        return scenarios, customers

    # First run: fit, posterior-predict, cache.
    log.info(
        "No cached forecast at %s - fitting hierarchical model (chains=%d, draws=%d)",
        cache_path, sampler.chains, sampler.draws,
    )
    n_train_days = history["date"].nunique()
    forecast_start = forecast_start or (
        pd.Timestamp(history["date"].max()) + pd.Timedelta(days=1)
    )

    model = build_hierarchical_model(history)
    idata = _sample(model, sampler)

    diag = check_diagnostics(idata)
    if not diag.healthy:
        log.warning(
            "Hierarchical fit unhealthy (div=%.3f, max_rhat=%.3f, min_ess=%.0f) - "
            "falling back to independent fits per the plan doc.",
            diag.divergence_rate, diag.max_r_hat, diag.min_ess,
        )
        model = build_independent_model(history)
        idata = _sample(model, sampler)

    forecast_idata = _posterior_predictive_for_horizon(
        model, idata, n_train_days, horizon_days, n_customers, forecast_start, sampler,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_idata.posterior_predictive.to_netcdf(cache_path)
    log.info("Cached posterior predictive to %s", cache_path)

    y = forecast_idata.posterior_predictive["y"].values
    scenarios = y.reshape(-1, y.shape[-2], y.shape[-1])
    return scenarios, customers
