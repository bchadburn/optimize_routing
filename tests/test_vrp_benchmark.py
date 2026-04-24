"""Tests for vrp_benchmark module."""
from __future__ import annotations

import numpy as np
import pytest

from vrp_benchmark.data import CVRPInstance, generate_instance, route_cost
from vrp_benchmark.solvers.greedy import GreedySolver
from vrp_benchmark.solvers.ortools_vrp import ORToolsSolver


# ── Data tests ───────────────────────────────────────────────────────────────

def test_generate_instance_shape():
    inst = generate_instance(10, seed=0)
    assert inst.n_customers == 10
    assert inst.coords.shape == (10, 2)
    assert inst.demands.shape == (10,)
    assert inst.capacity > 0


def test_dist_matrix_symmetric():
    inst = generate_instance(5, seed=1)
    d = inst.dist_matrix
    assert d.shape == (6, 6)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_route_cost_single_route():
    inst = generate_instance(3, seed=2)
    routes = [[1, 2, 3]]
    cost = route_cost(inst, routes)
    expected = (
        inst.dist(0, 1) + inst.dist(1, 2) + inst.dist(2, 3) + inst.dist(3, 0)
    )
    assert abs(cost - expected) < 1e-9


def test_route_cost_empty_routes():
    inst = generate_instance(3, seed=3)
    assert route_cost(inst, []) == 0.0
    assert route_cost(inst, [[]]) == 0.0


# ── Greedy solver tests ──────────────────────────────────────────────────────

def test_greedy_visits_all_customers():
    inst = generate_instance(15, seed=4)
    solver = GreedySolver()
    routes, cost = solver.solve(inst)
    visited = [node for r in routes for node in r]
    assert sorted(visited) == list(range(1, inst.n_customers + 1))


def test_greedy_respects_capacity():
    inst = generate_instance(20, seed=5)
    solver = GreedySolver()
    routes, _ = solver.solve(inst)
    for route in routes:
        load = sum(inst.demands[n - 1] for n in route)
        assert load <= inst.capacity, f"Route load {load} exceeds capacity {inst.capacity}"


def test_greedy_cost_matches_route_cost():
    inst = generate_instance(10, seed=6)
    solver = GreedySolver()
    routes, reported_cost = solver.solve(inst)
    computed_cost = route_cost(inst, routes)
    assert abs(reported_cost - computed_cost) < 1e-6


# ── OR-Tools solver tests ─────────────────────────────────────────────────────

def test_ortools_visits_all_customers():
    inst = generate_instance(15, seed=7)
    solver = ORToolsSolver(time_limit_s=5)
    routes, cost = solver.solve(inst)
    assert cost < 1e8, "OR-Tools should find a solution for n=15"
    visited = [node for r in routes for node in r]
    assert sorted(visited) == list(range(1, inst.n_customers + 1))


def test_ortools_better_than_greedy():
    """OR-Tools with local search should beat pure greedy on average."""
    greedy = GreedySolver()
    ortools = ORToolsSolver(time_limit_s=5)
    wins = 0
    for seed in range(10):
        inst = generate_instance(30, seed=seed + 100)
        _, gc = greedy.solve(inst)
        _, oc = ortools.solve(inst)
        if oc < gc:
            wins += 1
    assert wins >= 7, f"OR-Tools should beat greedy in at least 7/10 instances, got {wins}"


def test_ortools_respects_capacity():
    inst = generate_instance(20, seed=8)
    solver = ORToolsSolver(time_limit_s=5)
    routes, _ = solver.solve(inst)
    for route in routes:
        load = sum(inst.demands[n - 1] for n in route)
        assert load <= inst.capacity
