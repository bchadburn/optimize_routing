"""Shared utilities for benchmark runner scripts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_cuopt_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard --cuopt / --nim / --cuopt-time flags to an argparse parser."""
    parser.add_argument("--cuopt", action="store_true", help="Include cuOpt solver")
    parser.add_argument(
        "--nim",
        action="store_true",
        help="Use NVIDIA NIM cloud API instead of self-hosted (set NVIDIA_API_KEY)",
    )
    parser.add_argument("--cuopt-time", type=int, default=30, help="cuOpt time limit (s)")


def init_cuopt_solver(
    include: bool,
    nim_mode: bool,
    time_s: int,
    solver_class: type,
) -> object | None:
    """Instantiate a cuOpt solver with error handling. Returns None if unavailable."""
    if not include:
        return None
    try:
        return solver_class(time_limit_s=time_s, mode="nim" if nim_mode else "self-hosted")
    except Exception as e:
        print(f"WARNING: cuOpt unavailable: {e}")
        return None
