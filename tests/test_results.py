import pytest
import pandas as pd
from utils.results import write_csv, write_learning_curve, write_policy_table

def test_write_csv_creates_file(tmp_path):
    rows = [{"method": "test", "total_cost": 1234.5, "day": 0}]
    out = tmp_path / "test.csv"
    write_csv(rows, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["method", "total_cost", "day"]
    assert df.iloc[0]["total_cost"] == pytest.approx(1234.5)

def test_write_learning_curve(tmp_path):
    rewards = [-1000.0, -900.0, -800.0]
    out = tmp_path / "lc.csv"
    write_learning_curve(rewards, out, window=2)
    df = pd.read_csv(out)
    assert "episode" in df.columns
    assert "episode_reward" in df.columns
    assert "smoothed_reward" in df.columns
    assert len(df) == 3

def test_write_policy_table(tmp_path):
    policy_map = {(0, 0): 1, (0, 1): 3, (1, 2): 7}
    out = tmp_path / "policy.csv"
    write_policy_table(policy_map, out, num_dcs=5)
    df = pd.read_csv(out)
    assert "day" in df.columns
    assert "demand_bucket" in df.columns
    assert "action_bitmask" in df.columns
    assert "open_dcs" in df.columns
    assert len(df) == 3
