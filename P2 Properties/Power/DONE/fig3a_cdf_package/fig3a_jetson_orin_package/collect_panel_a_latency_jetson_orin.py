#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect real per-frame latency samples for Fig.3a CDF on Jetson Orin.

Adapted from keep_running.py:
- same PyTorch Poisson reconstruction kernel structure;
- CUDA tensors are created once for compute-only benchmarking;
- CUDA warm-up is performed before measurement;
- torch.cuda.synchronize() is used around every timed iteration;
- NVML power logging is optional.

Important for Jetson:
Jetson Orin is an integrated SoC. CPU and GPU share system DRAM, so H2D/D2H are
not discrete PCIe transfers as on an RTX workstation. Use:
  --measurement-scope compute_only
for pure CUDA-resident kernel latency, or:
  --measurement-scope app_e2e
for CPU-side tensor preparation/copy + GPU compute + output materialization.

Outputs:
  <platform>_fig3a_latency_samples.csv
  <platform>_latency_summary.csv
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.fft

try:
    import pynvml
except Exception:
    pynvml = None


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


def init_nvml(index: int = 0):
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(index)
    except Exception as e:
        print(f"[warning] NVML unavailable: {e}", file=sys.stderr)
        return None


def shutdown_nvml():
    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_gpu_power_w(handle):
    if handle is None:
        return float("nan")
    try:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        return float("nan")


def dst_torch(x, type=2, n=None, axis=-1, norm=None):
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device if torch.is_tensor(x) else None)
    if n is not None:
        x = torch.nn.functional.pad(x, (0, max(0, n - x.shape[axis])))
        x = x.narrow(axis, 0, n)

    N = x.shape[axis]
    if type == 2:
        x = torch.cat([x, -x.flip([axis])], dim=axis)
        idx = torch.arange(1, N + 1, device=x.device)
        result = torch.fft.fft(x, dim=axis).imag.index_select(axis, idx)
        if norm == "ortho":
            result.mul_(math.sqrt(2 / N))
            result[..., 0].mul_(math.sqrt(2))
        return result
    raise ValueError(f"DST type-{type} is not implemented")


def idst_torch(input_tensor, norm=None, axis=-1):
    n = input_tensor.shape[axis]
    if norm == "ortho":
        input_tensor = input_tensor * math.sqrt(2 / (n + 1))
    else:
        input_tensor = input_tensor * 2

    extended_shape = list(input_tensor.shape)
    extended_shape[axis] = 2 * (n + 1)
    extended = torch.zeros(extended_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    slices = [slice(None)] * input_tensor.dim()
    slices[axis] = slice(1, n + 1)
    extended[tuple(slices)] = -input_tensor

    slices[axis] = slice(n + 2, 2 * n + 2)
    extended[tuple(slices)] = input_tensor.flip([axis])

    ifft_result = torch.fft.ifft(extended, dim=axis)

    slices[axis] = slice(1, n + 1)
    return ifft_result[tuple(slices)].imag


def make_denominator(h: int, w: int, dtype: torch.dtype, device: torch.device):
    y, x = np.ogrid[1:h + 1, 1:w + 1]
    denom_np = (2 * np.cos(math.pi * x / (w + 1)) - 2) + (2 * np.cos(math.pi * y / (h + 1)) - 2)
    return torch.tensor(denom_np, dtype=dtype, device=device).unsqueeze(0)


def poisson_reconstruct_pytorch(grady, gradx, boundarysrc, denom=None):
    if boundarysrc is None:
        boundarysrc = torch.zeros_like(grady)

    gyy = grady[:, 1:, :-1] - grady[:, :-1, :-1]
    gxx = gradx[:, :-1, 1:] - gradx[:, :-1, :-1]
    f = torch.zeros_like(boundarysrc)
    f[:, :-1, 1:] += gxx
    f[:, 1:, :-1] += gyy

    boundary = boundarysrc.clone()
    boundary[:, 1:-1, 1:-1] = 0
    f_bp = (
        -4 * boundary[:, 1:-1, 1:-1]
        + boundary[:, 1:-1, 2:]
        + boundary[:, 1:-1, :-2]
        + boundary[:, 2:, 1:-1]
        + boundary[:, :-2, 1:-1]
    )
    f = f[:, 1:-1, 1:-1] - f_bp

    tt = dst_torch(f, norm="ortho", axis=-1)
    fsin = dst_torch(tt.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)

    h, w = f.shape[1], f.shape[2]
    if denom is None:
        denom = make_denominator(h, w, f.dtype, f.device)

    f = fsin / denom

    tt = idst_torch(f, norm="ortho", axis=-1)
    img_tt = idst_torch(tt.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)

    result = boundary
    result[:, 1:-1, 1:-1] += img_tt
    return result


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This Jetson script requires CUDA-enabled PyTorch.")

    device = torch.device("cuda:0")
    h, w = args.height, args.width
    b = args.batch_size

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    nvml_handle = init_nvml(args.gpu_index) if args.log_power else None

    # Precompute denominator once to avoid repeatedly timing invariant CPU->GPU setup.
    denom = make_denominator(h - 2, w - 2, torch.float32, device)

    # Compute-only path: inputs already resident on GPU, matching keep_running.py style.
    gradx_gpu = torch.randn(b, h, w, device=device, dtype=torch.float32)
    grady_gpu = torch.randn(b, h, w, device=device, dtype=torch.float32)
    boundary_gpu = torch.zeros(b, h, w, device=device, dtype=torch.float32)

    # App E2E path: CPU-side source tensors. On Jetson this is shared-DRAM SoC movement,
    # not discrete PCIe H2D/D2H.
    gradx_cpu = torch.randn(b, h, w, dtype=torch.float32)
    grady_cpu = torch.randn(b, h, w, dtype=torch.float32)
    boundary_cpu = torch.zeros(b, h, w, dtype=torch.float32)

    def one_iter_compute_only():
        return poisson_reconstruct_pytorch(grady_gpu, gradx_gpu, boundary_gpu, denom=denom)

    def one_iter_app_e2e():
        # Simulate application-visible path: CPU tensors become CUDA tensors,
        # GPU computation runs, and one scalar is materialized on CPU to force output readiness.
        gx = gradx_cpu.to(device, non_blocking=False)
        gy = grady_cpu.to(device, non_blocking=False)
        bd = boundary_cpu.to(device, non_blocking=False)
        out = poisson_reconstruct_pytorch(gy, gx, bd, denom=denom)
        return out[0, 0, 0].detach().cpu().item()

    if args.measurement_scope == "compute_only":
        fn = one_iter_compute_only
    elif args.measurement_scope == "app_e2e":
        fn = one_iter_app_e2e
    else:
        raise ValueError(args.measurement_scope)

    print(f"Platform: {args.platform}")
    print(f"Resolution: {w}x{h}")
    print(f"Batch size: {b}")
    print(f"Measurement scope: {args.measurement_scope}")
    print(f"Warmup: {args.warmup}, measured iterations: {args.num_iters}")

    for _ in range(args.warmup):
        _ = fn()
    torch.cuda.synchronize()

    samples = []
    for i in range(args.num_iters):
        torch.cuda.synchronize()
        p0 = get_gpu_power_w(nvml_handle)
        t0 = time.perf_counter_ns()

        _ = fn()

        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        p1 = get_gpu_power_w(nvml_handle)

        latency_ms = (t1 - t0) / 1e6
        samples.append({
            "platform": args.platform,
            "run_id": args.run_id,
            "frame_id": i,
            "latency_ms": latency_ms,
            "resolution_width": w,
            "resolution_height": h,
            "fps_target": args.fps_target,
            "batch_size": b,
            "timestamp_s": time.time(),
            "measurement_method": f"jetson_torch_cuda_{args.measurement_scope}",
            "power_w_before": p0,
            "power_w_after": p1,
            "notes": "Jetson integrated SoC; host-device movement is not discrete PCIe H2D/D2H."
        })

    shutdown_nvml()

    lat = np.array([r["latency_ms"] for r in samples], dtype=np.float64)
    summary = {
        "platform": args.platform,
        "width": w,
        "height": h,
        "pixels": w * h,
        "fps": args.fps_target,
        "batch_size": b,
        "latency_ms": float(np.mean(lat)),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "latency_p99_ms": percentile(lat, 99),
        "latency_mean_ms": float(np.mean(lat)),
        "latency_std_ms": float(np.std(lat, ddof=1)) if len(lat) > 1 else 0.0,
        "latency_min_ms": float(np.min(lat)),
        "latency_max_ms": float(np.max(lat)),
        "measurement_scope": args.measurement_scope,
        "num_iters": args.num_iters,
        "warmup": args.warmup,
        "notes": "Use samples CSV for Fig.3a CDF; use summary CSV for p50/p95/p99 fields."
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_name(args.platform)}_{w}x{h}_{args.measurement_scope}"

    samples_path = out_dir / f"{stem}_fig3a_latency_samples.csv"
    summary_path = out_dir / f"{stem}_latency_summary.csv"

    with samples_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\nLatency summary")
    print(f"latency_p50_ms = {summary['latency_p50_ms']:.6f}")
    print(f"latency_p95_ms = {summary['latency_p95_ms']:.6f}")
    print(f"latency_p99_ms = {summary['latency_p99_ms']:.6f}")
    print(f"mean_ms        = {summary['latency_mean_ms']:.6f}")
    print(f"std_ms         = {summary['latency_std_ms']:.6f}")
    print(f"samples_csv    = {samples_path}")
    print(f"summary_csv    = {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Collect Jetson Orin latency samples for Fig.3a CDF.")
    p.add_argument("--platform", default="Jetson Orin NX")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--fps-target", type=float, default=180.0)
    p.add_argument("--measurement-scope", choices=["compute_only", "app_e2e"], default="app_e2e")
    p.add_argument("--out-dir", default="fig3a_jetson_latency")
    p.add_argument("--run-id", default="run001")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-power", action="store_true")
    p.add_argument("--gpu-index", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
