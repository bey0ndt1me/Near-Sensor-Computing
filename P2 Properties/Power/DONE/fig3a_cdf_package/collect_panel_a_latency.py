#!/usr/bin/env python3
"""
Collect per-frame latency samples for Fig. 3a real CDF.

Important:
  - The CDF must be drawn from raw per-frame latency_ms samples, not only p50/p95/p99.
  - Default workload is a PyTorch Poisson/DST reconstruction matching the CPU/GPU/Jetson baseline.
  - To benchmark your own pipeline, replace run_one_frame_hook() and use --workload hook.

Examples:
  Jetson end-to-end GPU path including H2D + compute + D2H:
    python collect_panel_a_latency.py --platform "Jetson Orin NX" --backend cuda \
      --measurement-scope h2d_compute_d2h --width 128 --height 128 --num-iters 1000 --warmup 100 \
      --out-dir fig3a_jetson

  RTX 4090 compute-only CUDA latency:
    python collect_panel_a_latency.py --platform "RTX 4090" --backend cuda \
      --measurement-scope compute_only --width 128 --height 128 --num-iters 1000 --warmup 100 \
      --out-dir fig3a_rtx4090

  CPU baseline:
    python collect_panel_a_latency.py --platform "CPU i7-12700" --backend cpu \
      --num-threads 20 --width 128 --height 128 --num-iters 1000 --warmup 100 \
      --out-dir fig3a_cpu
"""
from __future__ import annotations

import argparse
import csv
import math
import platform as py_platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def percentile_summary(lat_ms: np.ndarray) -> Dict[str, float]:
    lat_ms = np.asarray(lat_ms, dtype=float)
    return {
        "latency_p50_ms": float(np.percentile(lat_ms, 50)),
        "latency_p95_ms": float(np.percentile(lat_ms, 95)),
        "latency_p99_ms": float(np.percentile(lat_ms, 99)),
        "latency_mean_ms": float(np.mean(lat_ms)),
        "latency_std_ms": float(np.std(lat_ms, ddof=1)) if len(lat_ms) > 1 else 0.0,
        "latency_min_ms": float(np.min(lat_ms)),
        "latency_max_ms": float(np.max(lat_ms)),
    }


# ------------------------- replace this for real pipeline ----------------------
def run_one_frame_hook() -> None:
    """Replace this function with one call to your true reconstruction pipeline."""
    raise NotImplementedError(
        "--workload hook selected, but run_one_frame_hook() has not been implemented. "
        "Put exactly one frame/batch of your reconstruction pipeline inside this function."
    )


def run_external_command(cmd: str) -> None:
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"External command failed with code {result.returncode}: {cmd}")


# -------------------------- PyTorch Poisson baseline --------------------------
def require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")


def dst_torch(x, norm: str | None = "ortho", axis: int = -1):
    n = x.shape[axis]
    x_ext = torch.cat([x, -x.flip([axis])], dim=axis)
    idx = torch.arange(1, n + 1, device=x.device)
    result = torch.fft.fft(x_ext, dim=axis).imag.index_select(axis, idx)
    if norm == "ortho":
        result.mul_(math.sqrt(2.0 / n))
        sl = [slice(None)] * result.dim()
        sl[axis] = 0
        result[tuple(sl)].mul_(math.sqrt(2.0))
    return result


def idst_torch(x, norm: str | None = "ortho", axis: int = -1):
    n = x.shape[axis]
    if norm == "ortho":
        x = x * math.sqrt(2.0 / (n + 1))
    else:
        x = x * 2.0
    ext_shape = list(x.shape)
    ext_shape[axis] = 2 * (n + 1)
    ext = torch.zeros(ext_shape, dtype=x.dtype, device=x.device)
    sl = [slice(None)] * x.dim()
    sl[axis] = slice(1, n + 1)
    ext[tuple(sl)] = -x
    sl[axis] = slice(n + 2, 2 * n + 2)
    ext[tuple(sl)] = x.flip([axis])
    out = torch.fft.ifft(ext, dim=axis)
    sl[axis] = slice(1, n + 1)
    return out[tuple(sl)].imag


class PoissonReconstructor:
    def __init__(self, height: int, width: int, batch_size: int, device):
        inner_h = height - 2
        inner_w = width - 2
        y, x = np.ogrid[1:inner_h + 1, 1:inner_w + 1]
        denom = (2 * np.cos(math.pi * x / (inner_w + 1)) - 2) + (2 * np.cos(math.pi * y / (inner_h + 1)) - 2)
        self.denom = torch.tensor(denom, dtype=torch.float32, device=device).unsqueeze(0)

    def __call__(self, grady, gradx, boundary):
        gyy = grady[:, 1:, :-1] - grady[:, :-1, :-1]
        gxx = gradx[:, :-1, 1:] - gradx[:, :-1, :-1]
        f = torch.zeros_like(boundary)
        f[:, :-1, 1:] += gxx
        f[:, 1:, :-1] += gyy
        b = boundary.clone()
        b[:, 1:-1, 1:-1] = 0
        f_bp = -4 * b[:, 1:-1, 1:-1] + b[:, 1:-1, 2:] + b[:, 1:-1, :-2] + b[:, 2:, 1:-1] + b[:, :-2, 1:-1]
        f_inner = f[:, 1:-1, 1:-1] - f_bp
        tmp = dst_torch(f_inner, norm="ortho", axis=-1)
        fsin = dst_torch(tmp.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)
        solved = fsin / self.denom
        tmp = idst_torch(solved, norm="ortho", axis=-1)
        img = idst_torch(tmp.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)
        out = b
        out[:, 1:-1, 1:-1] += img
        return out


def cuda_sync(device) -> None:
    if torch is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def make_torch_inputs(batch: int, height: int, width: int, device, cpu_source: bool):
    source_device = torch.device("cpu") if cpu_source else device
    grady = torch.randn(batch, height, width, dtype=torch.float32, device=source_device)
    gradx = torch.randn(batch, height, width, dtype=torch.float32, device=source_device)
    boundary = torch.zeros(batch, height, width, dtype=torch.float32, device=source_device)
    return grady, gradx, boundary


def prepare_torch_workload(args):
    require_torch()
    device = torch.device(args.backend)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    if device.type == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    include_transfers = (device.type == "cuda" and args.measurement_scope == "h2d_compute_d2h")
    recon = PoissonReconstructor(args.height, args.width, args.batch_size, device=device)
    if include_transfers:
        grady_cpu, gradx_cpu, boundary_cpu = make_torch_inputs(args.batch_size, args.height, args.width, device, cpu_source=True)
        grady_dev = torch.empty_like(grady_cpu, device=device)
        gradx_dev = torch.empty_like(gradx_cpu, device=device)
        boundary_dev = torch.empty_like(boundary_cpu, device=device)
    else:
        grady_dev, gradx_dev, boundary_dev = make_torch_inputs(args.batch_size, args.height, args.width, device, cpu_source=False)
        grady_cpu = gradx_cpu = boundary_cpu = None

    def one_frame():
        nonlocal grady_dev, gradx_dev, boundary_dev
        if include_transfers:
            grady_dev.copy_(grady_cpu, non_blocking=False)
            gradx_dev.copy_(gradx_cpu, non_blocking=False)
            boundary_dev.copy_(boundary_cpu, non_blocking=False)
        out = recon(grady_dev, gradx_dev, boundary_dev)
        if include_transfers:
            _ = out.detach().cpu()

    return one_frame, device


def benchmark(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.workload == "torch_poisson":
        one_frame, device = prepare_torch_workload(args)
        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else py_platform.processor()
    elif args.workload == "hook":
        one_frame = run_one_frame_hook
        device = None
        device_name = "user_hook"
    elif args.workload == "command":
        if not args.command:
            raise ValueError("--workload command requires --command '...'")
        one_frame = lambda: run_external_command(args.command)
        device = None
        device_name = "external_command"
    else:
        raise ValueError(args.workload)

    # Warm-up samples are intentionally discarded.
    for _ in range(args.warmup):
        one_frame()
        if device is not None:
            cuda_sync(device)

    latencies_ms = []
    timestamps_s = []
    t_global = time.perf_counter()
    for i in range(args.num_iters):
        if device is not None:
            cuda_sync(device)
        t0 = time.perf_counter()
        one_frame()
        if device is not None:
            cuda_sync(device)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies_ms.append(lat_ms)
        timestamps_s.append(t1 - t_global)
        if args.print_every and (i + 1) % args.print_every == 0:
            st = percentile_summary(np.array(latencies_ms))
            print(f"{i+1:5d}/{args.num_iters}: p50={st['latency_p50_ms']:.4f} ms, p95={st['latency_p95_ms']:.4f} ms, p99={st['latency_p99_ms']:.4f} ms")

    lat = np.asarray(latencies_ms, dtype=float)
    stats = percentile_summary(lat)
    include_transfers = args.measurement_scope == "h2d_compute_d2h"
    run_id = args.run_id or f"{args.platform.replace(' ', '_')}_{args.width}x{args.height}_{time.strftime('%Y%m%d_%H%M%S')}"

    raw_rows = []
    for idx, (lat_ms, ts) in enumerate(zip(latencies_ms, timestamps_s), start=1):
        raw_rows.append({
            "platform": args.platform,
            "display_name": args.display_name or args.platform,
            "run_id": run_id,
            "frame_id": idx,
            "latency_ms": lat_ms,
            "resolution_width": args.width,
            "resolution_height": args.height,
            "fps_target": args.fps_target,
            "batch_size": args.batch_size,
            "backend": args.backend,
            "device_name": device_name,
            "measurement_scope": args.measurement_scope,
            "include_transfers": include_transfers,
            "timestamp_s": ts,
            "warmup": args.warmup,
            "num_iters": args.num_iters,
            "notes": args.notes,
        })
    raw_df = pd.DataFrame(raw_rows)
    summary = dict(
        platform=args.platform,
        display_name=args.display_name or args.platform,
        run_id=run_id,
        width=args.width,
        height=args.height,
        pixels=args.width * args.height,
        fps_target=args.fps_target,
        batch_size=args.batch_size,
        backend=args.backend,
        device_name=device_name,
        measurement_scope=args.measurement_scope,
        include_transfers=include_transfers,
        num_iters=args.num_iters,
        warmup=args.warmup,
        notes=args.notes,
        **stats,
    )
    return raw_df, pd.DataFrame([summary])


def parse_args():
    p = argparse.ArgumentParser(description="Collect raw per-frame latency samples for Fig. 3a CDF.")
    p.add_argument("--platform", required=True, help="Platform label, e.g. 'Jetson Orin NX', 'RTX 4090', 'CPU i7-12700'.")
    p.add_argument("--display-name", default=None, help="Optional display name used by plotting script.")
    p.add_argument("--backend", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--workload", default="torch_poisson", choices=["torch_poisson", "hook", "command"])
    p.add_argument("--command", default=None, help="External command to time once per iteration when --workload command is used.")
    p.add_argument("--measurement-scope", default="compute_only", choices=["compute_only", "h2d_compute_d2h"], help="For CUDA: compute only or H2D+compute+D2H wall-clock latency.")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fps-target", type=float, default=180.0)
    p.add_argument("--num-iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--num-threads", type=int, default=0, help="CPU only. 0 keeps PyTorch default.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", default="fig3a_latency")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df, summary_df = benchmark(args)
    run_id = summary_df.iloc[0]["run_id"]
    raw_path = out_dir / f"{run_id}_fig3a_latency_samples.csv"
    summary_path = out_dir / f"{run_id}_latency_summary.csv"
    template_path = out_dir / "fig3a_latency_cdf_filled.csv"
    raw_df.to_csv(raw_path, index=False, quoting=csv.QUOTE_MINIMAL)
    raw_df.to_csv(template_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)

    row = summary_df.iloc[0]
    print("\nFig. 3a latency summary")
    print(f"platform          : {row['platform']}")
    print(f"resolution        : {int(row['width'])}x{int(row['height'])}")
    print(f"measurement_scope : {row['measurement_scope']}")
    print(f"latency_p50_ms    : {row['latency_p50_ms']:.6f}")
    print(f"latency_p95_ms    : {row['latency_p95_ms']:.6f}")
    print(f"latency_p99_ms    : {row['latency_p99_ms']:.6f}")
    print(f"raw_samples_csv   : {raw_path}")
    print(f"summary_csv       : {summary_path}")
    print(f"panel_a_csv       : {template_path}")


if __name__ == "__main__":
    main()
