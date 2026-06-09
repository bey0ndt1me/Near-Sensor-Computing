#!/usr/bin/env python3
"""Merge multiple per-platform Fig. 3a latency sample CSVs into one panel-a template CSV."""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import pandas as pd

REQUIRED = ["platform", "frame_id", "latency_ms"]


def expand_inputs(inputs):
    files = []
    unmatched = []
    for item in inputs:
        if item == "\\":
            continue

        matches = sorted(glob.glob(item))
        if matches:
            files.extend(matches)
        else:
            unmatched.append(item)

    if unmatched:
        raise FileNotFoundError(
            "No files matched these input path(s): "
            + ", ".join(unmatched)
            + "\nPowerShell line continuation uses a backtick (`), not backslash (\\)."
        )
    if not files:
        raise FileNotFoundError("No input CSV files were provided.")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Per-platform *_fig3a_latency_samples.csv files.")
    ap.add_argument("--out", default="fig3a_latency_cdf_template.csv")
    args = ap.parse_args()
    dfs = []
    for f in expand_inputs(args.inputs):
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
