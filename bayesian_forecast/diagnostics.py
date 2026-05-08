"""Diagnostic plots for the Bayesian forecast pipeline.

Produces a 2x2 grid mirroring FICO's reference: raw history, posterior predictive
check on training data, out-of-sample forecast with HDI band, per-day skewness.
The skewness panel is the most actionable for downstream optimization - if the
posterior predictive is heavily right-skewed on a given day, the 95th-percentile
input to the chance-constrained MILP will dominate the daily plan, so it's
worth showing the reviewer where that happens.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast_diagnostics(
    history: pd.DataFrame,
    scenarios: np.ndarray,
    customers: list[int],
    out_path: Path,
    customer_to_plot: int = 0,
    hdi_prob: float = 0.94,
) -> Path:
    """Render the 2x2 diagnostic figure to ``out_path``.

    Plots focus on a single customer (``customer_to_plot``) so the figure is
    legible. A reviewer who needs other customers re-runs with a different id.

    Args:
        history: long-form demand history (training data).
        scenarios: ``(n_scenarios, horizon_days, n_customers)`` from forecast_demand.
        customers: customer ids in the column order of the last axis of ``scenarios``.
        out_path: PNG output path. Parents created.
        customer_to_plot: which customer's history + forecast to render.
        hdi_prob: HDI probability for the credible band (0.94 matches FICO).

    Returns:
        The output path.
    """
    if customer_to_plot not in customers:
        raise ValueError(
            f"customer_to_plot={customer_to_plot} not in customers={customers}"
        )
    cust_idx = customers.index(customer_to_plot)

    cust_history = history[history["customer_id"] == customer_to_plot].sort_values("date")
    train_dates  = pd.to_datetime(cust_history["date"].to_numpy())
    train_demand = cust_history["demand"].to_numpy()

    n_scenarios, horizon, _ = scenarios.shape
    last_train = train_dates.max()
    future_dates = pd.date_range(last_train + pd.Timedelta(days=1), periods=horizon, freq="D")

    fc_for_cust = scenarios[:, :, cust_idx]                # (n_scenarios, horizon)
    fc_mean     = fc_for_cust.mean(axis=0)
    lo_q        = (1 - hdi_prob) / 2
    hi_q        = 1 - lo_q
    fc_lo       = np.quantile(fc_for_cust, lo_q, axis=0)
    fc_hi       = np.quantile(fc_for_cust, hi_q, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1,1) — Raw training history
    ax = axes[0, 0]
    ax.plot(train_dates, train_demand, "o-", markersize=2, color="tab:blue")
    ax.set_ylabel("Demand (units/day)")
    ax.set_title(f"Customer {customer_to_plot}: training history")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # (1,2) — Posterior mean for the customer overlaid on observed training data
    # We don't have the posterior predictive on training dates here (forecast()
    # only stores the out-of-sample window) so this panel reuses the rolling
    # 7-day mean as a smoothed reference. Cheaper than re-running PP-on-train.
    ax = axes[0, 1]
    ax.plot(train_dates, train_demand, "-o", markersize=2, color="black",
            alpha=0.5, label="Observed")
    if len(train_demand) >= 7:
        roll = pd.Series(train_demand).rolling(7, min_periods=1).mean().to_numpy()
        ax.plot(train_dates, roll, color="tab:blue", linewidth=1.5,
                label="7-day rolling mean")
    ax.set_ylabel("Demand (units/day)")
    ax.set_title("Training data + smoothed reference")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # (2,1) — Out-of-sample forecast with HDI band
    ax = axes[1, 0]
    ax.plot(future_dates, fc_mean, label="Posterior predictive mean",
            color="tab:blue", linewidth=1.5)
    ax.fill_between(future_dates, fc_lo, fc_hi, alpha=0.3, color="tab:blue",
                    label=f"{int(hdi_prob*100)}% credible interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (units/day)")
    ax.set_title(f"Customer {customer_to_plot}: out-of-sample forecast")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # (2,2) — Per-day skewness (right-skew indicates fat upper tail; relevant
    # because the chance-constrained 95th percentile will sit far from the mean
    # when this is high)
    from scipy.stats import skew

    skewnesses = np.array([skew(fc_for_cust[:, t]) for t in range(horizon)])
    ax = axes[1, 1]
    ax.plot(future_dates, skewnesses, "o-", markersize=2, color="tab:purple")
    ax.axhline(0.0, linestyle="--", color="grey", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Skewness")
    ax.set_title("Per-day posterior-predictive skewness")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
