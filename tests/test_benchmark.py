"""Smoke test: benchmark runs on smallest config and writes CSV with correct columns."""
import csv
from pathlib import Path


def test_benchmark_ortools_writes_csv(tmp_path):
    """Benchmark with OR-Tools writes CSV with expected columns."""
    from rl.benchmark import ScalabilityBenchmark

    runner = ScalabilityBenchmark(results_dir=tmp_path, n_trials=1, include_cuopt=False)
    runner.run(customer_counts=[5])

    csv_path = tmp_path / "cuopt_benchmark.csv"
    assert csv_path.exists()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"n_customers", "n_vehicles", "solver", "solve_time_s", "total_cost"}
    assert rows[0]["solver"] == "ortools_vrp"
    assert float(rows[0]["total_cost"]) > 0.0
