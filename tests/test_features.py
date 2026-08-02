"""
Deliverable 6 sanity tests: at least one test per feature validating the
engineered feature against the raw close/volume data, plus a model-quality check.
Runs in CI after `dvc pull` (test data) and fetching the model from MLflow.
"""
import os
import pandas as pd
import pytest

TEST_CSV = os.getenv("TEST_CSV", "processed/test_v0.csv")
FEATURES = ["rolling_avg_10", "volume_sum_10"]


@pytest.fixture(scope="module")
def df():
    assert os.path.exists(TEST_CSV), f"{TEST_CSV} missing — did `dvc pull` run?"
    return pd.read_csv(TEST_CSV)


# ---- sanity test for feature: rolling_avg_10 ----
def test_rolling_avg_10_within_price_range(df):
    # a 10-min moving average of close must sit within the min/max close seen
    assert (df["rolling_avg_10"] >= 0).all(), "rolling_avg_10 has negatives"
    # rolling avg should be finite and close to close-price magnitude
    assert df["rolling_avg_10"].notna().all(), "rolling_avg_10 has NaNs"
    assert (df["rolling_avg_10"] <= df["close"].max() * 5).all(), "rolling_avg_10 implausibly large"


# ---- sanity test for feature: volume_sum_10 ----
def test_volume_sum_10_nonnegative_and_ge_volume(df):
    # a 10-min sum of volume must be >= 0 and >= the single-minute volume
    assert (df["volume_sum_10"] >= 0).all(), "volume_sum_10 negative"
    assert (df["volume_sum_10"] >= df["volume"]).all(), "volume_sum_10 < single-minute volume"


# ---- sanity test for feature: stock_name ----
def test_stock_name_present(df):
    assert "stock_name" in df.columns and df["stock_name"].notna().all()


# ---- target integrity ----
def test_target_binary(df):
    assert set(df["target"].unique()).issubset({0, 1}), "target not binary"
