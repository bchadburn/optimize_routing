"""Synthesizer is pure NumPy — fast tests, no PyMC dependency."""
from __future__ import annotations

import numpy as np
import pytest

from bayesian_forecast.synthesize import SynthSpec, synthesize_demand


def test_synthesize_returns_long_format_dataframe():
    spec = SynthSpec(n_customers=3, n_days=14, seed=1)
    df = synthesize_demand(spec)
    assert list(df.columns) == ["date", "customer_id", "demand"]
    assert len(df) == 3 * 14
    # All demand values strictly positive (LogNormal)
    assert (df["demand"] > 0).all()
    # Customers 0..n-1 present
    assert sorted(df["customer_id"].unique().tolist()) == [0, 1, 2]


def test_synthesize_seed_is_deterministic():
    spec = SynthSpec(n_customers=2, n_days=10, seed=42)
    df_a = synthesize_demand(spec)
    df_b = synthesize_demand(spec)
    np.testing.assert_array_equal(df_a["demand"].to_numpy(), df_b["demand"].to_numpy())


def test_synthesize_customer_means_diverge_with_population_sigma():
    """Different customers should have visibly different mean demands when
    pop_sigma > 0 — that's the whole point of the per-customer level."""
    spec = SynthSpec(n_customers=10, n_days=200, seed=1, pop_sigma=0.5)
    df = synthesize_demand(spec)
    means = df.groupby("customer_id")["demand"].mean()
    # Coefficient of variation across customer means should be substantial
    cv = means.std() / means.mean()
    assert cv > 0.1, (
        f"customer means too similar ({cv=:.3f}); "
        "synthesizer's population sigma may not be flowing through"
    )


def test_synthesize_weekly_seasonality_is_visible():
    """Day-of-week aggregate should not be flat — the shared Fourier creates a
    weekly bump."""
    spec = SynthSpec(
        n_customers=5,
        n_days=4 * 7,           # 4 full weeks
        seed=1,
        weekly_amplitude=0.3,   # boost so the test isn't fragile
        noise_scale=0.05,       # quiet down LogNormal noise
    )
    df = synthesize_demand(spec)
    df = df.assign(dow=df["date"].dt.dayofweek)
    by_dow = df.groupby("dow")["demand"].mean()
    # Range across day-of-week means should be > 5% of the overall mean
    overall = df["demand"].mean()
    spread = (by_dow.max() - by_dow.min()) / overall
    assert spread > 0.05, f"weekly spread too small ({spread=:.3f})"


def test_synthesize_handles_zero_trend_without_negative_demand():
    """Edge case: zero trend, zero noise spec should still yield positive demand."""
    spec = SynthSpec(n_customers=2, n_days=5, seed=1, trend_scale=0.0, noise_scale=0.0)
    df = synthesize_demand(spec)
    assert (df["demand"] > 0).all()


@pytest.mark.parametrize("n_days", [1, 7, 30])
def test_synthesize_supports_short_horizons(n_days):
    df = synthesize_demand(SynthSpec(n_customers=2, n_days=n_days, seed=1))
    assert len(df) == 2 * n_days
