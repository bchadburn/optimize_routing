"""End-to-end Bayesian forecast tests.

NUTS is slow — these tests use small data + minimal draws to keep the suite
under a minute. The `slow` marker is added so a developer running
``pytest -m "not slow"`` skips them; CI runs everything.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from bayesian_forecast.forecast import (
    SamplerSpec,
    check_diagnostics,
    forecast_demand,
)
from bayesian_forecast.synthesize import SynthSpec, synthesize_demand

# All forecast tests run NUTS — mark them slow so users can skip with
# ``pytest -m "not slow"``.
pytestmark = pytest.mark.slow


# Small spec keeps NUTS fast enough for CI: 3 customers x 56 days, 2 chains x 100 draws.
TINY_HISTORY_SPEC = SynthSpec(n_customers=3, n_days=56, seed=11)
TINY_SAMPLER      = SamplerSpec(draws=100, tune=200, chains=2, target_accept=0.9, seed=11)


def test_forecast_returns_expected_shape(tmp_path, caplog):
    history = synthesize_demand(TINY_HISTORY_SPEC)
    cache_path = tmp_path / "forecast.nc"

    with caplog.at_level(logging.INFO):
        scenarios, customers = forecast_demand(
            history,
            horizon_days=10,
            cache_path=cache_path,
            sampler=TINY_SAMPLER,
        )

    # 2 chains x 100 draws = 200 scenarios; 10 days; 3 customers
    assert scenarios.shape == (200, 10, 3)
    assert customers == [0, 1, 2]
    # All demand strictly positive (LogNormal)
    assert (scenarios > 0).all()
    # Cache file exists
    assert cache_path.exists()


def test_forecast_cache_short_circuits_second_call(tmp_path):
    """Second call with the same cache must NOT re-sample."""
    history = synthesize_demand(TINY_HISTORY_SPEC)
    cache_path = tmp_path / "forecast.nc"

    # First run populates the cache
    s1, c1 = forecast_demand(
        history, horizon_days=10, cache_path=cache_path, sampler=TINY_SAMPLER,
    )
    mtime_1 = cache_path.stat().st_mtime

    # Second run loads from cache - should be fast and produce identical samples
    s2, c2 = forecast_demand(
        history, horizon_days=10, cache_path=cache_path, sampler=TINY_SAMPLER,
    )
    mtime_2 = cache_path.stat().st_mtime

    np.testing.assert_array_equal(s1, s2)
    assert c1 == c2
    # Cache file untouched on second call
    assert mtime_1 == mtime_2


def test_forecast_force_rerun_overwrites_cache(tmp_path):
    history = synthesize_demand(TINY_HISTORY_SPEC)
    cache_path = tmp_path / "forecast.nc"

    forecast_demand(history, horizon_days=10, cache_path=cache_path, sampler=TINY_SAMPLER)
    mtime_1 = cache_path.stat().st_mtime

    # Force rerun
    forecast_demand(history, horizon_days=10, cache_path=cache_path, sampler=TINY_SAMPLER, force=True)
    mtime_2 = cache_path.stat().st_mtime
    assert mtime_2 > mtime_1


def test_check_diagnostics_flags_unhealthy_run():
    """Construct a fake idata-shaped object whose diagnostics fail and confirm
    ``check_diagnostics`` returns ``healthy=False``."""
    # Real call below also exercises the healthy path; this test focuses on the
    # threshold logic via a synthetic dataset.
    pytest.importorskip("arviz")
    import arviz as az

    # 4 chains x 50 draws of a single highly-correlated parameter -> low ESS
    rng = np.random.default_rng(0)
    samples = np.cumsum(rng.standard_normal((4, 50)), axis=1)
    idata = az.from_dict(
        posterior={"theta": samples},
        sample_stats={"diverging": np.zeros((4, 50), dtype=bool)},
    )

    diag = check_diagnostics(idata)
    # Random walk -> R-hat blows up, ESS tiny - either flag should trigger
    assert not diag.healthy


def test_forecast_includes_parameter_uncertainty(tmp_path):
    """The whole point of the Bayesian path: variance across scenarios should
    exceed the per-customer observation noise alone (otherwise we're not
    capturing parameter uncertainty)."""
    history = synthesize_demand(SynthSpec(n_customers=3, n_days=120, seed=2))
    cache_path = tmp_path / "forecast.nc"

    scenarios, _ = forecast_demand(
        history, horizon_days=14, cache_path=cache_path, sampler=TINY_SAMPLER,
    )

    # For each customer, variance across scenarios on day 0 reflects both
    # parameter uncertainty AND observation noise. We expect it to be visibly
    # > 0 - a tight assertion would be flaky on this small sample, but a
    # sanity check that scenarios actually vary catches a class of bugs where
    # the cache or set_data path collapses to a single trajectory.
    for cust in range(scenarios.shape[2]):
        spread = scenarios[:, 0, cust].std() / scenarios[:, 0, cust].mean()
        assert spread > 0.01, (
            f"customer {cust}: scenario coefficient of variation collapsed to "
            f"{spread:.4f} - posterior predictive may not be flowing through"
        )
