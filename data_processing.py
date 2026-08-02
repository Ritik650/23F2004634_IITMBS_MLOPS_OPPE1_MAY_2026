"""
Process NSE minute-level stock CSVs into a training-ready feature set.

Generalizes data_processing_reference.ipynb to ALL stocks in a data folder:
  - forward-fills missing values
  - rolling_avg_10  : 10-minute rolling mean of close (per stock)
  - volume_sum_10   : 10-minute rolling sum of volume (per stock)
  - target          : 1 if close is higher 5 minutes later, else 0
  - holds out the last 20 rows per stock as the test set

Subset-first: --nrows caps rows read PER FILE so you can validate the pipeline
on a small slice before running on the full data.

Usage:
    python data_processing.py --data-dir StockAnalyticaData/v0 --version v0 --nrows 20000
    python data_processing.py --data-dir StockAnalyticaData/v0 --version v0            # full
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

FEATURES = ["rolling_avg_10", "volume_sum_10"]
TARGET = "target"


def process_stock(path, nrows=None):
    stock = os.path.basename(path).split("__")[0]
    df = pd.read_csv(path, nrows=nrows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["stock_name"] = stock
    df = df.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")
    df.ffill(inplace=True)

    df["rolling_avg_10"] = df["close"].rolling("10min", min_periods=1).mean()
    df["volume_sum_10"] = df["volume"].rolling("10min", min_periods=1).sum()
    df.dropna(subset=["rolling_avg_10", "volume_sum_10"], inplace=True)

    df["close_5min_future"] = df["close"].shift(-5)
    df["target"] = (df["close_5min_future"] > df["close"]).astype(int)
    df.drop(["close_5min_future"], axis=1, inplace=True)
    df = df.dropna(subset=["target"]).copy()

    test = df.tail(20).copy()
    train = df.iloc[:-20].copy()
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--version", required=True, help="label for outputs, e.g. v0/v1")
    ap.add_argument("--nrows", type=int, default=None, help="cap rows per file (subset mode)")
    ap.add_argument("--out-dir", default="processed")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    print(f"Processing {len(files)} stock file(s) from {args.data_dir}"
          f"{' (subset '+str(args.nrows)+' rows/file)' if args.nrows else ' (full)'}")

    trains, tests = [], []
    for f in files:
        tr, te = process_stock(f, nrows=args.nrows)
        trains.append(tr); tests.append(te)
        print(f"  {os.path.basename(f).split('__')[0]:<10} train={len(tr):>7} test={len(te)}")

    train = pd.concat(trains).reset_index()
    test = pd.concat(tests).reset_index()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, f"train_{args.version}.csv")
    test_path = os.path.join(args.out_dir, f"test_{args.version}.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"\nWrote {train_path} ({len(train)} rows) and {test_path} ({len(test)} rows)")
    print(f"Target balance: {train['target'].mean():.2%} up")


if __name__ == "__main__":
    main()
