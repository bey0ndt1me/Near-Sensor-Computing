#!/usr/bin/env python3
"""
Jetson panel-d measurement using jtop total power only.

Updated protocol
----------------
For each target FPS:
  1. Idle settle for 10 s.
  2. Measure average idle total power for the next 10 s.
  3. Run workload for 10 s as active settle.
  4. Run workload for the next 10 s and measure average active total power.

Then:
  full_chip_power_w = active_total_power_w
  incremental_power_w = active_total_power_w - idle_total_power_w
  full_chip_energy_per_frame_mj = 1000 * full_chip_power_w / processed_fps
  incremental_energy_per_frame_mj = 1000 * incremental_power_w / processed_fps

No core-power measurement.
No latency measurement.

Default target FPS:
  30, 45, 60, 90, 120, 180

Run:
  python3 measure_panel_d_jetson.py --out-dir panel_d_measurements
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from jtop import jtop
except Exception as e:
    raise RuntimeError(
        "Failed to import jtop. Install/enable jetson-stats first:\n"
        "  sudo -H pip3 install -U jetson-stats\n"
        "  sudo systemctl restart jtop.service\n"
        "  sudo reboot\n"
    ) from e


JETSON_TARGET_FPS = [30, 45, 60, 90, 120, 180]


# -----------------------------------------------------------------------------
# Synthetic Poisson reconstruction workload
# -----------------------------------------------------------------------------


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PoissonReconstructor:
    def __init__(self, height: int, width: int, batch_size: int, device: torch.device):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device

        ky = torch.fft.fftfreq(height, d=1.0, device=device).reshape(height, 1)
        kx = torch.fft.fftfreq(width, d=1.0, device=device).reshape(1, width)
        denom = (2.0 * np.pi) ** 2 * (kx * kx + ky * ky)
        denom[0, 0] = 1.0
        self.inv_laplacian = 1.0 / denom
        self.inv_laplacian[0, 0] = 0.0

    def __call__(self, grady: torch.Tensor, gradx: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        div = torch.zeros_like(boundary)
        div[:, :, :-1] += gradx[:, :, :-1]
        div[:, :, 1:] -= gradx[:, :, :-1]
        div[:, :-1, :] += grady[:, :-1, :]
        div[:, 1:, :] -= grady[:, :-1, :]
        div = div + 0.01 * boundary

        div_hat = torch.fft.fft2(div)
        depth_hat = div_hat * self.inv_laplacian
        depth = torch.fft.ifft2(depth_hat).real
        return depth


def make_inputs(batch_size: int, resolution: int, seed: int, pin_memory: bool):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    kwargs = dict(dtype=torch.float32, device="cpu")
    if pin_memory and torch.cuda.is_available():
        kwargs["pin_memory"] = True

    grady = torch.randn(batch_size, resolution, resolution, generator=gen, **kwargs)
    gradx = torch.randn(batch_size, resolution, resolution, generator=gen, **kwargs)
    boundary = torch.zeros(batch_size, resolution, resolution, **kwargs)
    return grady, gradx, boundary


def prepare_workload(args):
    device = torch.device("cuda:0")
    recon = PoissonReconstructor(args.resolution, args.resolution, args.batch_size, device)

    host = make_inputs(
        batch_size=args.batch_size,
        resolution=args.resolution,
        seed=args.seed,
        pin_memory=(not args.no_pin_memory),
    )

    if args.include_transfers:
        grady_h, gradx_h, boundary_h = host
        dev = (
            torch.empty_like(grady_h, device=device),
            torch.empty_like(gradx_h, device=device),
            torch.empty_like(boundary_h, device=device),
        )
    else:
        dev = tuple(x.to(device) for x in host)
        host = None

    return device, recon, host, dev


def run_one_frame(args, recon, host, dev) -> None:
    if args.include_transfers:
        grady_h, gradx_h, boundary_h = host
        grady_d, gradx_d, boundary_d = dev
        grady_d.copy_(grady_h, non_blocking=False)
        gradx_d.copy_(gradx_h, non_blocking=False)
        boundary_d.copy_(boundary_h, non_blocking=False)
    else:
        grady_d, gradx_d, boundary_d = dev

    out = recon(grady_d, gradx_d, boundary_d)

    if args.include_transfers:
        _ = out.detach().cpu()


# -----------------------------------------------------------------------------
# jtop total power sampling
# -----------------------------------------------------------------------------


def read_total_power(jetson) -> Dict[str, float]:
    """Read total Jetson power using the same style as keep_running.py."""
    total_power_mw = jetson.power.get("tot", {}).get("power", 0)
    try:
        mem_used_mb = float(jetson.memory["RAM"]["used"]) / 1024.0
    except Exception:
        mem_used_mb = float("nan")

    return {
        "timestamp_s": time.perf_counter(),
        "total_power_w": float(total_power_mw) / 1000.0,
        "mem_used_mb": mem_used_mb,
    }


def sample_idle_phase(jetson, args, duration_s: float, all_samples: List[dict], phase: str) -> Tuple[float, float]:
    t0 = time.perf_counter()
    next_sample = t0

    while True:
        now = time.perf_counter()
        if now - t0 >= duration_s:
            break
        if now >= next_sample:
            s = read_total_power(jetson)
            s.update({"phase": phase, "target_fps": ""})
            all_samples.append(s)
            next_sample += args.power_sample_interval_s
        time.sleep(min(0.005, max(0.0, next_sample - time.perf_counter())))

    t1 = time.perf_counter()
    return t0, t1


def run_active_phase(
    jetson,
    args,
    target_fps: int,
    duration_s: float,
    device,
    recon,
    host,
    dev,
    all_samples: List[dict],
    phase: str,
    count_frames: bool,
) -> Tuple[float, float, int]:
    t0 = time.perf_counter()
    next_power_sample = t0
    frames = 0
    period_s = 1.0 / float(target_fps)

    while True:
        if time.perf_counter() - t0 >= duration_s:
            break

        iter_t0 = time.perf_counter()

        run_one_frame(args, recon, host, dev)
        sync_if_cuda(device)

        if count_frames:
            frames += args.batch_size

        now = time.perf_counter()
        if now >= next_power_sample:
            s = read_total_power(jetson)
            s.update({"phase": phase, "target_fps": target_fps})
            all_samples.append(s)
            next_power_sample += args.power_sample_interval_s

        sleep_s = period_s - (time.perf_counter() - iter_t0)
        if sleep_s > 0:
            sleep_end = time.perf_counter() + sleep_s
            while time.perf_counter() < sleep_end:
                now = time.perf_counter()
                if now >= next_power_sample:
                    s = read_total_power(jetson)
                    s.update({"phase": phase, "target_fps": target_fps})
                    all_samples.append(s)
                    next_power_sample += args.power_sample_interval_s
                time.sleep(min(0.002, max(0.0, sleep_end - time.perf_counter())))

    t1 = time.perf_counter()
    return t0, t1, frames


def mean_power(samples: List[dict], t0: float, t1: float) -> float:
    vals = [
        float(s["total_power_w"])
        for s in samples
        if t0 <= float(s["timestamp_s"]) <= t1 and np.isfinite(float(s["total_power_w"]))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def measure_one_target(args, jetson, target_fps: int, device, recon, host, dev, all_samples: List[dict]) -> dict:
    # Optional warmup before the formal protocol; not used for statistics.
    for _ in range(args.warmup):
        run_one_frame(args, recon, host, dev)
    sync_if_cuda(device)

    print(f"  idle settle {args.idle_settle_s:.1f} s")
    idle_settle_t0, idle_settle_t1 = sample_idle_phase(
        jetson, args, args.idle_settle_s, all_samples, phase="idle_settle"
    )

    print(f"  idle measure {args.idle_measure_s:.1f} s")
    idle_measure_t0, idle_measure_t1 = sample_idle_phase(
        jetson, args, args.idle_measure_s, all_samples, phase="idle_measure"
    )

    print(f"  active settle {args.active_settle_s:.1f} s")
    active_settle_t0, active_settle_t1, _ = run_active_phase(
        jetson, args, target_fps, args.active_settle_s,
        device, recon, host, dev, all_samples,
        phase="active_settle", count_frames=False
    )

    print(f"  active measure {args.active_measure_s:.1f} s")
    active_measure_t0, active_measure_t1, frames = run_active_phase(
        jetson, args, target_fps, args.active_measure_s,
        device, recon, host, dev, all_samples,
        phase="active_measure", count_frames=True
    )

    idle_power_w = mean_power(all_samples, idle_measure_t0, idle_measure_t1)
    active_power_w = mean_power(all_samples, active_measure_t0, active_measure_t1)
    processed_fps = frames / max(active_measure_t1 - active_measure_t0, 1e-12)
    incremental_power_w = active_power_w - idle_power_w
    full_chip_power_w = active_power_w
    full_chip_energy_per_frame_mj = 1000.0 * full_chip_power_w / processed_fps
    incremental_energy_per_frame_mj = 1000.0 * incremental_power_w / processed_fps

    return {
        "platform": args.platform,
        "point_id": f"{args.platform} {target_fps} FPS",
        "target_fps": target_fps,
        "processed_fps": processed_fps,
        "active_power_w": active_power_w,
        "idle_power_w": idle_power_w,
        "full_chip_power_w": full_chip_power_w,
        "incremental_power_w": incremental_power_w,
        "full_chip_energy_per_frame_mj": full_chip_energy_per_frame_mj,
        "incremental_energy_per_frame_mj": incremental_energy_per_frame_mj,
        "bubble_size_w": full_chip_power_w,
        "energy_basis": "full-chip and incremental",
        "meets_180_fps": "Yes" if processed_fps >= 180.0 else "No",
        "notes": (
            "Jetson jtop total power only; "
            f"idle {args.idle_settle_s:.0f}s settle + {args.idle_measure_s:.0f}s measure; "
            f"active {args.active_settle_s:.0f}s settle + {args.active_measure_s:.0f}s measure; "
            "panel-d excludes latency."
        ),
        "n_frames": frames,
        "idle_settle_start_s": idle_settle_t0,
        "idle_settle_end_s": idle_settle_t1,
        "idle_measure_start_s": idle_measure_t0,
        "idle_measure_end_s": idle_measure_t1,
        "active_settle_start_s": active_settle_t0,
        "active_settle_end_s": active_settle_t1,
        "active_measure_start_s": active_measure_t0,
        "active_measure_end_s": active_measure_t1,
    }


def main():
    parser = argparse.ArgumentParser(description="Jetson panel-d total-power measurement using jtop.")
    parser.add_argument("--platform", default="Jetson Orin NX")
    parser.add_argument("--target-fps-list", default=",".join(map(str, JETSON_TARGET_FPS)))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)

    parser.add_argument("--idle-settle-s", type=float, default=10.0)
    parser.add_argument("--idle-measure-s", type=float, default=10.0)
    parser.add_argument("--active-settle-s", type=float, default=10.0)
    parser.add_argument("--active-measure-s", type=float, default=10.0)

    parser.add_argument("--power-sample-interval-s", type=float, default=0.05)
    parser.add_argument("--out-dir", default="panel_d_measurements")
    parser.add_argument("--include-transfers", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")

    targets = [int(x) for x in args.target_fps_list.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, recon, host, dev = prepare_workload(args)

    rows: List[dict] = []
    raw_samples: List[dict] = []

    print("--- Jetson panel-d measurement via jtop total power ---")
    print(f"Targets: {targets}")
    print(
        f"Protocol per FPS: idle settle {args.idle_settle_s}s, "
        f"idle measure {args.idle_measure_s}s, "
        f"active settle {args.active_settle_s}s, "
        f"active measure {args.active_measure_s}s"
    )
    print("Power metric: jetson.power['tot']['power'] only")

    with jtop() as jetson:
        time.sleep(1.0)
        first = read_total_power(jetson)
        print(f"Initial total power: {first['total_power_w']:.3f} W")

        for target in targets:
            print(f"[Jetson] target {target} FPS")
            row = measure_one_target(args, jetson, target, device, recon, host, dev, raw_samples)
            rows.append(row)
            print(
                f"  processed={row['processed_fps']:.2f} FPS | "
                f"idle={row['idle_power_w']:.3f} W | "
                f"active={row['active_power_w']:.3f} W | "
                f"incr={row['incremental_power_w']:.3f} W | "
                f"full-chip E/frame={row['full_chip_energy_per_frame_mj']:.3f} mJ | "
                f"incr E/frame={row['incremental_energy_per_frame_mj']:.3f} mJ"
            )

    raw_fields = ["timestamp_s", "phase", "target_fps", "total_power_w", "mem_used_mb"]
    summary_fields = [
        "platform", "point_id", "target_fps", "processed_fps",
        "active_power_w", "idle_power_w", "full_chip_power_w", "incremental_power_w",
        "full_chip_energy_per_frame_mj", "incremental_energy_per_frame_mj", "bubble_size_w", "energy_basis",
        "meets_180_fps", "notes", "n_frames",
        "idle_settle_start_s", "idle_settle_end_s",
        "idle_measure_start_s", "idle_measure_end_s",
        "active_settle_start_s", "active_settle_end_s",
        "active_measure_start_s", "active_measure_end_s",
    ]
    template_fields = [
        "platform", "point_id", "target_fps", "processed_fps",
        "active_power_w", "idle_power_w", "full_chip_power_w", "incremental_power_w",
        "full_chip_energy_per_frame_mj", "incremental_energy_per_frame_mj", "bubble_size_w", "energy_basis",
        "meets_180_fps", "notes",
    ]

    write_csv(out_dir / "jetson_panel_d_raw_total_power_jtop.csv", raw_samples, raw_fields)
    write_csv(out_dir / "jetson_panel_d_summary_total_jtop.csv", rows, summary_fields)
    write_csv(out_dir / "jetson_panel_d_for_template.csv", rows, template_fields)

    print("Done.")
    print("Wrote:", out_dir / "jetson_panel_d_summary_total_jtop.csv")
    print("Wrote:", out_dir / "jetson_panel_d_for_template.csv")
    print("Wrote:", out_dir / "jetson_panel_d_raw_total_power_jtop.csv")


if __name__ == "__main__":
    main()
