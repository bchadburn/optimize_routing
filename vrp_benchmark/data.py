"""CVRP instance generator and data types.

Generates random instances in the unit square with uniform demand.
All instances use a single depot at (0.5, 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CVRPInstance:
    """A single Capacitated VRP instance.

    Attributes:
        n_customers: Number of customers (excluding depot).
        depot: (x, y) coordinates of the depot.
        coords: (n_customers, 2) array of customer coordinates.
        demands: (n_customers,) array of integer demands.
        capacity: Vehicle capacity (same for all vehicles).
        n_vehicles: Maximum vehicles available.
    """

    n_customers: int
    depot: np.ndarray
    coords: np.ndarray
    demands: np.ndarray
    capacity: int
    n_vehicles: int

    # Precomputed distance matrix including depot at index 0
    _dist: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        all_coords = np.vstack([self.depot, self.coords])  # (n+1, 2)
        diff = all_coords[:, None, :] - all_coords[None, :, :]
        self._dist = np.sqrt((diff ** 2).sum(axis=-1))

    def dist(self, i: int, j: int) -> float:
        """Distance between node i and j (0 = depot, 1..n = customers)."""
        return float(self._dist[i, j])

    @property
    def dist_matrix(self) -> np.ndarray:
        """Full (n+1) × (n+1) distance matrix."""
        return self._dist


def generate_instance(
    n_customers: int,
    capacity: int = 50,
    demand_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> CVRPInstance:
    """Generate a random CVRP instance in the unit square.

    Args:
        n_customers: Number of customers.
        capacity: Vehicle capacity.
        demand_range: (min, max) inclusive for customer demands.
        seed: Random seed for reproducibility.

    Returns:
        A CVRPInstance with n_customers customers.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(n_customers, 2))
    demands = rng.integers(demand_range[0], demand_range[1] + 1, size=n_customers)
    depot = np.array([0.5, 0.5])
    # Minimum vehicles needed: ceil(total_demand / capacity)
    n_vehicles = int(np.ceil(demands.sum() / capacity)) + 1
    return CVRPInstance(
        n_customers=n_customers,
        depot=depot,
        coords=coords,
        demands=demands,
        capacity=capacity,
        n_vehicles=n_vehicles,
    )


def route_cost(instance: CVRPInstance, routes: list[list[int]]) -> float:
    """Compute total distance for a solution (routes are lists of customer indices 1..n).

    Each route implicitly starts and ends at the depot (index 0).
    """
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0  # depot
        for node in route:
            total += instance.dist(prev, node)
            prev = node
        total += instance.dist(prev, 0)  # return to depot
    return total
