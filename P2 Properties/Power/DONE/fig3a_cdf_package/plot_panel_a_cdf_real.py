#!/usr/bin/env python3
"""Plot Fig. 3a CDF directly from raw latency samples. No synthetic data is generated."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['axes.linewidth'] = 0.8

COLORS = {
    'FPGA pipeline (this work)': '#1f5aa6',
    'This work': '#1f5aa6',
    'Jetson Orin NX': '#e67e22',
    'RTX 4090': '#3182bd',
    'GPU workstation (RTX 4090)': '#3182bd',
    'CPU i7-12700': '#b22222',
    'CPU workstation (i7-12700)': '#b22222',
}
ORDER = [
    'FPGA pipeline (this work)',
    'This work',
    'Jetson Orin NX',
    'RTX 4090',
    'GPU workstation (RTX 4090)',
    'CPU i7-12700',
    'CPU workstation (i7-12700)',
]


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Merged fig3a latency sample CSV.')
    ap.add_argument('--out-prefix', default='Fig3a_real_cdf')
    ap.add_argument('--xmin', type=float, default=0.1)
    ap.add_argument('--xmax', type=float, default=None)
    ap.add_argument('--fpga-latency-ms', type=float, default=0.21149, help='Optional fixed FPGA latency line if FPGA samples are absent. Use <=0 to disable.')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if 'platform' not in df.columns or 'latency_ms' not in df.columns:
        raise ValueError('Input must contain platform and latency_ms columns.')
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df = df.dropna(subset=['platform', 'latency_ms'])

    if args.fpga_latency_ms > 0 and not df['platform'].astype(str).isin(['FPGA pipeline (this work)', 'This work']).any():
        extra = pd.DataFrame({
            'platform': ['FPGA pipeline (this work)'] * 1000,
            'latency_ms': [args.fpga_latency_ms] * 1000,
        })
        df = pd.concat([df, extra], ignore_index=True)

    fig, ax = plt.subplots(figsize=(88/25.4, 58/25.4))
    used = set()
    for name in ORDER + sorted(set(df['platform'].astype(str))):
        if name in used:
            continue
        used.add(name)
        g = df[df['platform'].astype(str) == name]
        if g.empty:
            continue
        x, y = ecdf(g['latency_ms'].to_numpy())
        label = name
        ax.step(x, y, where='post', lw=1.3, color=COLORS.get(name, None), label=label)
        p99 = np.percentile(x, 99)
        ax.plot([p99], [0.99], marker='o', ms=2.5, color=COLORS.get(name, None))

    ax.set_xscale('log')
    ax.set_xlim(args.xmin, args.xmax or max(df['latency_ms'].quantile(0.999) * 1.3, args.xmin * 10))
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('Per-frame latency CDF (raw measured samples)', loc='left', fontweight='bold')
    ax.grid(axis='y', color='#dadada', lw=0.6)
    ax.axhline(0.50, color='0.45', linestyle=(0, (4, 4)), lw=0.75)
    ax.axhline(0.99, color='0.45', linestyle=(0, (4, 4)), lw=0.75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='0.75', loc='lower right')

    out_pdf = Path(args.out_prefix).with_suffix('.pdf')
    out_png = Path(args.out_prefix).with_suffix('.png')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(out_pdf)
    print(out_png)


if __name__ == '__main__':
    main()
