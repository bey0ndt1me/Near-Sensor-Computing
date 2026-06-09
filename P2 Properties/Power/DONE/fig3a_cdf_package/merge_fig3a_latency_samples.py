#!/usr/bin/env python3
"""Merge multiple per-platform Fig. 3a latency sample CSVs into one panel-a template CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED = ["platform", "frame_id", "latency_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Per-platform *_fig3a_latency_samples.csv files.")
    ap.add_argument("--out", default="fig3a_latency_cdf_template.csv")
    args = ap.parse_args()
    dfs = []
    for f in args.inputs:
        df = pd.read_csv(f)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns: {missing}")
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        df = df.dropna(subset=["latency_ms"])
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Merged {len(out)} latency samples -> {args.out}")


if __name__ == "__main__":
    main()
