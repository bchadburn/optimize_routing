"""Utilities for writing result CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_learning_curve(
    episode_rewards: list[float],
    path: Path,
    window: int = 100,
) -> None:
    """Write learning curve CSV with smoothed reward column."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rewards = np.array(episode_rewards)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="same")
    df = pd.DataFrame({
        "episode": np.arange(len(rewards)),
        "episode_reward": rewards,
        "smoothed_reward": smoothed,
    })
    df.to_csv(path, index=False)


def write_policy_table(
    policy_map: dict[tuple[int, int], int],
    path: Path,
    num_dcs: int,
) -> None:
    """Write policy table: (day, demand_bucket) -> action bitmask + human-readable DC list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (day, bucket), action in sorted(policy_map.items()):
        open_dcs = [dc_id for dc_id in range(num_dcs) if (action >> dc_id) & 1]
        rows.append({
            "day": day,
            "demand_bucket": bucket,
            "action_bitmask": action,
            "open_dcs": str(open_dcs),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
