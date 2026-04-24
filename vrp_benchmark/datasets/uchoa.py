"""Uchoa et al. (2017) X-instance CVRP benchmark loader.

Downloads instances on first use and caches them locally. These are the
standard modern CVRP benchmark instances with published best-known solutions
(BKS). Unlike Solomon instances (VRPTW), these are pure CVRP — matching our
existing solver interface exactly.

Reference:
    Uchoa et al. (2017) "New benchmark instances for the Capacitated Vehicle
    Routing Problem." European Journal of Operational Research 257(3): 845–858.

Best-known solutions are from the CVRPLIB website and HGS-CVRP paper (Vidal 2022).
Only instances with confirmed optimal or near-optimal BKS are included.

Instance naming: X-nN-kK where N = customers+1 (includes depot), K = vehicles.
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrp_benchmark.data import CVRPInstance

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "uchoa"
_BASE_URL = "https://raw.githubusercontent.com/vidalt/HGS-CVRP/main/Instances/CVRP"

# Selected instances covering small → large scale, with their best-known costs.
# BKS values from CVRPLIB (http://vrp.galgos.inf.puc-rio.br/index.php/en/) and Vidal (2022).
INSTANCES: dict[str, int] = {
    "X-n101-k25":  27591,
    "X-n106-k14":  26362,
    "X-n110-k13":  14971,
    "X-n115-k10":  12747,
    "X-n120-k6":   13332,
    "X-n125-k30":  55539,
    "X-n129-k18":  28940,
    "X-n134-k13":  10916,
    "X-n139-k10":  13590,
    "X-n143-k7":   15700,
    "X-n153-k22":  21220,
    "X-n157-k13":  16876,
    "X-n162-k11":  14138,
    "X-n167-k10":  20557,
    "X-n172-k51":  45607,
    "X-n176-k26":  47812,
    "X-n181-k23":  25569,
    "X-n186-k15":  24145,
    "X-n190-k8":   16980,
    "X-n195-k51":  44225,
    "X-n200-k36":  58578,
}

# Subset to use by default in benchmark runs (one per rough size bucket)
DEFAULT_INSTANCES = [
    "X-n101-k25",   # ~100 customers
    "X-n115-k10",   # ~115
    "X-n139-k10",   # ~139
    "X-n162-k11",   # ~162
    "X-n200-k36",   # ~200
]


@dataclass
class BenchmarkInstance:
    name: str
    instance: CVRPInstance
    bks: int  # best-known solution cost (integer, using floor distances)


def _download(name: str) -> Path:
    """Download instance file to cache if not present. Returns local path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.vrp"
    if not path.exists():
        url = f"{_BASE_URL}/{name}.vrp"
        logger.info("Downloading %s ...", name)
        urllib.request.urlretrieve(url, path)
    return path


def _parse_vrplib(path: Path) -> CVRPInstance:
    """Parse a VRPLIB-format CVRP instance. Returns CVRPInstance."""
    text = path.read_text()
    lines = text.splitlines()

    capacity = 0
    coords: list[tuple[float, float]] = []
    demands: list[int] = []

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip().split()[0])
        elif line == "NODE_COORD_SECTION":
            section = "coords"
        elif line == "DEMAND_SECTION":
            section = "demands"
        elif line in ("DEPOT_SECTION", "EOF"):
            section = None
        elif section == "coords":
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
        elif section == "demands":
            parts = line.split()
            if len(parts) >= 2:
                demands.append(int(parts[1]))

    # Node 0 is depot; customers are 1..n
    depot = np.array(coords[0])
    cust_coords = np.array(coords[1:])
    cust_demands = np.array(demands[1:], dtype=float)
    n_customers = len(cust_coords)

    n_vehicles = int(np.ceil(cust_demands.sum() / capacity)) + 2

    return CVRPInstance(
        n_customers=n_customers,
        depot=depot,
        coords=cust_coords,
        demands=cust_demands,
        capacity=capacity,
        n_vehicles=n_vehicles,
    )


def load(name: str) -> BenchmarkInstance:
    """Load a named Uchoa instance. Downloads on first use."""
    if name not in INSTANCES:
        raise ValueError(f"Unknown instance {name!r}. Available: {list(INSTANCES)}")
    path = _download(name)
    instance = _parse_vrplib(path)
    return BenchmarkInstance(name=name, instance=instance, bks=INSTANCES[name])


def load_default() -> list[BenchmarkInstance]:
    """Load the default benchmark set (one instance per size bucket)."""
    return [load(name) for name in DEFAULT_INSTANCES]
