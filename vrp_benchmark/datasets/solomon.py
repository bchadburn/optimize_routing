"""Solomon (1987) VRPTW benchmark loader.

Downloads instances on first use and caches them locally. These are the
standard time-window VRP benchmark instances. Best-known solutions (BKS)
are from the VRPLIB/Sintef leaderboard.

Reference:
    Solomon, M.M. (1987) "Algorithms for the Vehicle Routing and Scheduling
    Problems with Time Window Constraints." Operations Research 35(2): 254–265.

Instance naming: CXxx (clustered), RXxx (random), RCXxx (mixed).
C1/R1/RC1 = tight time windows; C2/R2/RC2 = wide time windows.
All instances have 100 customers. Distance = Euclidean (speed = 1, so travel
time equals distance numerically).
"""
from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrp_benchmark.data import CVRPInstance
from vrp_benchmark.data_tw import VRPTWInstance

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache" / "solomon"
_BASE_URL = "https://raw.githubusercontent.com/gaopan812/VRPTW_solomon_instances/main/solomon_100"

# Best-known solutions: (distance, n_vehicles)
# Sources: CVRPLIB (http://vrp.galgos.inf.puc-rio.br) and Sintef (2010 leaderboard).
BKS: dict[str, tuple[float, int]] = {
    # C1 — clustered, tight windows
    "C101": (828.94, 10), "C102": (828.94, 10), "C103": (828.06, 10),
    "C104": (824.78, 10), "C105": (828.94, 10), "C106": (828.94, 10),
    "C107": (828.94, 10), "C108": (828.94, 10), "C109": (828.94, 10),
    # C2 — clustered, wide windows
    "C201": (591.56, 3), "C202": (591.56, 3), "C203": (591.17, 3),
    "C204": (590.60, 3), "C205": (588.88, 3), "C206": (588.49, 3),
    "C207": (588.29, 3), "C208": (588.32, 3),
    # R1 — random, tight windows
    "R101": (1650.80, 19), "R102": (1486.12, 17), "R103": (1292.68, 13),
    "R104": (1007.24, 9),  "R105": (1377.11, 14), "R106": (1251.98, 12),
    "R107": (1104.66, 10), "R108": (960.88, 9),   "R109": (1194.73, 11),
    "R110": (1118.59, 10), "R111": (1096.72, 10), "R112": (982.14, 9),
    # R2 — random, wide windows
    "R201": (1252.37, 4), "R202": (1191.70, 3), "R203": (939.50, 3),
    "R204": (825.52, 2),  "R205": (994.42, 3),  "R206": (906.14, 3),
    "R207": (890.61, 2),  "R208": (726.82, 2),  "R209": (909.16, 3),
    "R210": (939.34, 3),  "R211": (892.71, 2),
    # RC1 — mixed, tight windows
    "RC101": (1696.94, 14), "RC102": (1554.75, 12), "RC103": (1261.67, 11),
    "RC104": (1135.48, 10), "RC105": (1629.44, 13), "RC106": (1424.73, 11),
    "RC107": (1230.48, 11), "RC108": (1139.82, 10),
    # RC2 — mixed, wide windows
    "RC201": (1406.91, 4), "RC202": (1365.65, 3), "RC203": (1049.62, 3),
    "RC204": (798.46, 3),  "RC205": (1297.65, 4), "RC206": (1146.32, 3),
    "RC207": (1061.14, 3), "RC208": (828.14, 3),
}

# One representative per family for quick runs
DEFAULT_INSTANCES = ["C101", "C201", "R101", "R201", "RC101", "RC201"]

FAMILIES: dict[str, list[str]] = {
    "C1":  [f"C10{i}" for i in range(1, 10)],
    "C2":  [f"C20{i}" for i in range(1, 9)],
    "R1":  [f"R10{i}" if i < 10 else f"R1{i}" for i in range(1, 13)],
    "R2":  [f"R20{i}" if i < 10 else f"R2{i}" for i in range(1, 12)],
    "RC1": [f"RC10{i}" for i in range(1, 9)],
    "RC2": [f"RC20{i}" for i in range(1, 9)],
}


@dataclass
class SolomonBenchmarkInstance:
    name: str
    instance: VRPTWInstance
    bks_distance: float
    bks_vehicles: int


def _download(name: str) -> Path:
    """Download instance JSON to cache if not present. Returns local path."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.json"
    if not path.exists():
        url = f"{_BASE_URL}/{name}.json"
        logger.info("Downloading %s ...", name)
        urllib.request.urlretrieve(url, path)
    return path


def _parse_solomon(path: Path) -> VRPTWInstance:
    """Parse Solomon JSON format. Returns VRPTWInstance.

    JSON structure:
        {"vehicle": {"number": N, "capacity": C},
         "customers": [{"id": 0, "x": ..., "y": ..., "demand": ...,
                        "ready_time": ..., "due_time": ..., "service_time": ...}, ...]}
    Index 0 is depot; indices 1..n are customers.
    """
    import json

    data = json.loads(path.read_text())
    n_vehicles = data["vehicle"]["number"]
    capacity = data["vehicle"]["capacity"]

    customers = sorted(data["customers"], key=lambda c: c["id"])
    xs = np.array([c["x"] for c in customers], dtype=float)
    ys = np.array([c["y"] for c in customers], dtype=float)
    demands = np.array([c["demand"] for c in customers], dtype=float)
    ready_times = np.array([c["ready_time"] for c in customers], dtype=float)
    due_dates = np.array([c["due_time"] for c in customers], dtype=float)
    service_times = np.array([c["service_time"] for c in customers], dtype=float)

    depot = np.array([xs[0], ys[0]])
    cust_coords = np.column_stack([xs[1:], ys[1:]])
    n_customers = len(cust_coords)

    cvrp = CVRPInstance(
        n_customers=n_customers,
        depot=depot,
        coords=cust_coords,
        demands=demands[1:],
        capacity=float(capacity),
        n_vehicles=n_vehicles,
    )

    return VRPTWInstance(
        cvrp=cvrp,
        ready_times=ready_times,
        due_dates=due_dates,
        service_times=service_times,
    )


def load(name: str) -> SolomonBenchmarkInstance:
    """Load a named Solomon instance. Downloads on first use."""
    if name not in BKS:
        raise ValueError(f"Unknown instance {name!r}. Available: {sorted(BKS)}")
    path = _download(name)
    instance = _parse_solomon(path)
    bks_dist, bks_veh = BKS[name]
    return SolomonBenchmarkInstance(
        name=name,
        instance=instance,
        bks_distance=bks_dist,
        bks_vehicles=bks_veh,
    )


def load_family(family: str) -> list[SolomonBenchmarkInstance]:
    """Load all instances in a family (e.g. 'C1', 'R2')."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}. Available: {sorted(FAMILIES)}")
    return [load(name) for name in FAMILIES[family]]
