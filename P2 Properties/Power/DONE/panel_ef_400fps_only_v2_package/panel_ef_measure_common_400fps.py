#!/usr/bin/env python3
"""
Panel e/f real-time latency-limited benchmark, v2.

# 400-FPS-only override for CPU/GPU panel e/f tests
CPU_TARGET_FPS = [400]
GPU_TARGET_FPS = [400]

Key change versus the previous target-FPS sweep:
  * No fixed-FPS sweep is used to decide whether a platform is "suitable".
  * For every resolution, the script first preheats the device and measures
    back-to-back single-frame latency with no artificial sleep.
  * latency_limited_fps is computed from the selected latency percentile
    (default p99): FPS = 1000 / latency_p99_ms.
  * A real-time operating FPS is then selected as the largest candidate FPS
    not exceeding latency_limited_fps and the platform cap.
  * Power is measured after 10 s idle settle + 10 s idle measure +
    10 s active settle + 10 s active measure.

This avoids the CPU DVFS artifact where a low target-FPS loop introduces sleep,
reduces CPU frequency, and makes latency appear just barely good enough for a
low target rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.fft

RESOLUTIONS_DEFAULT = [64, 128, 256]
CANDIDATE_FPS_DEFAULT = [400]

CSV_HEADER = [
    "platform", "device", "resolution", "label", "pixels",
    "latency_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "latency_percentile_for_fps", "latency_limited_fps",
    "selected_realtime_fps", "active_fps", "missed_deadline_rate",
    "active_power_w", "idle_power_w", "incremental_power_w",
    "energy_per_frame_mj", "measure_mode", "notes",
]


# -----------------------------
# Poisson reconstruction workload
# -----------------------------

def dst_torch(x: torch.Tensor, norm: Optional[str] = "ortho", axis: int = -1) -> torch.Tensor:
    """DST-II implemented with FFT, matching the structure used in keep_running.py."""
    n = x.shape[axis]
    x_rev = torch.flip(x, dims=[axis])
    y = torch.cat([x, -x_rev], dim=axis)
    Y = torch.fft.fft(y, dim=axis)
    out = -Y.imag.narrow(axis, 1, n)
    if norm == "ortho":
        out = out * math.sqrt(1.0 / (2.0 * (n + 1)))
        # Orthonormal DST-II has a first-mode correction in strict definitions.
        # We keep this lightweight implementation consistent across platforms.
    return out


def idst_torch(x: torch.Tensor, norm: Optional[str] = "ortho", axis: int = -1) -> torch.Tensor:
    """Use DST as its inverse under the same normalization for benchmarking."""
    return dst_torch(x, norm=norm, axis=axis)


def poisson_reconstruct_pytorch(grady: torch.Tensor, gradx: torch.Tensor, boundarysrc: torch.Tensor) -> torch.Tensor:
    """Full per-frame Poisson reconstruction path: divergence -> boundary -> DST -> division -> IDST."""
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

    h, w = f.shape[-2], f.shape[-1]
    yy = torch.arange(1, h + 1, device=f.device, dtype=f.dtype).view(1, h, 1)
    xx = torch.arange(1, w + 1, device=f.device, dtype=f.dtype).view(1, 1, w)
    denom = 2.0 * torch.cos(math.pi * yy / (h + 1)) + 2.0 * torch.cos(math.pi * xx / (w + 1)) - 4.0

    u = fsin / denom

    tt = idst_torch(u, norm="ortho", axis=-1)
    img_tt = idst_torch(tt.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)

    out = boundarysrc.clone()
    out[:, 1:-1, 1:-1] = img_tt
    return out


class PoissonWorkload:
    def __init__(self, resolution: int, batch: int, device: torch.device, seed: int = 0) -> None:
        self.resolution = int(resolution)
        self.batch = int(batch)
        self.device = device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + self.resolution)

        shape = (self.batch, self.resolution, self.resolution)
        self.grady = torch.randn(shape, generator=gen, dtype=torch.float32).to(device)
        self.gradx = torch.randn(shape, generator=gen, dtype=torch.float32).to(device)
        self.boundary = torch.zeros(shape, dtype=torch.float32, device=device)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def step(self) -> None:
        _ = poisson_reconstruct_pytorch(self.grady, self.gradx, self.boundary)

    def step_timed_ms(self) -> float:
        t0 = time.perf_counter()
        self.step()
        self.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    def run_continuous_for_seconds(self, duration_s: float) -> int:
        end_t = time.perf_counter() + float(duration_s)
        frames = 0
        while time.perf_counter() < end_t:
            self.step()
            self.synchronize()
            frames += self.batch
        return frames

    def run_paced_for_seconds(self, duration_s: float, target_fps: float) -> Tuple[int, int]:
        period = 1.0 / float(target_fps)
        end_t = time.perf_counter() + float(duration_s)
        next_deadline = time.perf_counter()
        frames = 0
        missed = 0
        while time.perf_counter() < end_t:
            t0 = time.perf_counter()
            self.step()
            self.synchronize()
            t1 = time.perf_counter()
            frames += self.batch
            if (t1 - t0) > period:
                missed += self.batch

            next_deadline += period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If already late, do not accumulate infinitely growing delay.
                next_deadline = time.perf_counter()
        return frames, missed


# -----------------------------
# Statistics and CSV helpers
# -----------------------------

def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def ensure_header(path: Path, header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(header)


def append_rows(path: Path, rows: Sequence[Dict[str, object]], header: Sequence[str] = CSV_HEADER) -> None:
    ensure_header(path, header)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def write_rows(path: Path, rows: Sequence[Dict[str, object]], header: Sequence[str] = CSV_HEADER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def choose_realtime_fps(latency_limited_fps: float, candidates: Sequence[float], platform_cap: float) -> float:
    # 400-FPS-only protocol:
    # Latency is still measured and written to CSV, but active-power measurement
    # is always paced at 400 FPS for CPU/GPU comparison.
    return 400.0


# -----------------------------
# Power readers
# -----------------------------

class PowerReader:
    def read_power_w(self) -> float:
        raise NotImplementedError

    def measure_idle_power_w(self, duration_s: float, sample_interval_s: float) -> float:
        vals = []
        t_end = time.perf_counter() + float(duration_s)
        while time.perf_counter() < t_end:
            vals.append(float(self.read_power_w()))
            time.sleep(float(sample_interval_s))
        return sum(vals) / max(1, len(vals))

    def measure_active_power_w(
        self,
        duration_s: float,
        sample_interval_s: float,
        workload: PoissonWorkload,
        target_fps: float,
    ) -> Tuple[float, float, float]:
        vals = []
        frames = 0
        missed = 0
        t_start = time.perf_counter()
        t_end = t_start + float(duration_s)
        period = 1.0 / float(target_fps)
        next_deadline = time.perf_counter()

        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            workload.step()
            workload.synchronize()
            t1 = time.perf_counter()
            frames += workload.batch
            if (t1 - t0) > period:
                missed += workload.batch

            vals.append(float(self.read_power_w()))

            next_deadline += period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, max(0.0, float(sample_interval_s))))
                # If there is still slack, sleep the remaining part.
                rest = next_deadline - time.perf_counter()
                if rest > 0:
                    time.sleep(rest)
            else:
                next_deadline = time.perf_counter()

        elapsed = max(1e-9, time.perf_counter() - t_start)
        active_fps = frames / elapsed
        miss_rate = missed / max(1, frames)
        power = sum(vals) / max(1, len(vals))
        return power, active_fps, miss_rate


class RAPLPackageReader(PowerReader):
    def __init__(self, rapl_path: Optional[str] = None) -> None:
        if rapl_path:
            base = Path(rapl_path)
        else:
            root = Path("/sys/class/powercap/intel-rapl")
            candidates = sorted(root.glob("intel-rapl:*"))
            if not candidates:
                raise RuntimeError("No Intel RAPL domain found under /sys/class/powercap/intel-rapl.")
            base = candidates[0]
        self.energy_path = base / "energy_uj"
        if not self.energy_path.exists():
            raise RuntimeError(f"RAPL energy file not found: {self.energy_path}")
        # Test permission early.
        try:
            _ = self.energy_path.read_text()
        except PermissionError as e:
            raise RuntimeError(
                f"Permission denied when reading {self.energy_path}. "
                "Run with sudo, or use: sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
            ) from e

    def read_energy_j(self) -> float:
        return float(self.energy_path.read_text().strip()) / 1e6

    def read_power_w(self) -> float:
        e0 = self.read_energy_j()
        t0 = time.perf_counter()
        time.sleep(0.05)
        e1 = self.read_energy_j()
        t1 = time.perf_counter()
        return (e1 - e0) / max(1e-9, t1 - t0)

    def measure_idle_power_w(self, duration_s: float, sample_interval_s: float) -> float:
        e0 = self.read_energy_j()
        t0 = time.perf_counter()
        time.sleep(float(duration_s))
        e1 = self.read_energy_j()
        t1 = time.perf_counter()
        return (e1 - e0) / max(1e-9, t1 - t0)

    def measure_active_power_w(
        self,
        duration_s: float,
        sample_interval_s: float,
        workload: PoissonWorkload,
        target_fps: float,
    ) -> Tuple[float, float, float]:
        e0 = self.read_energy_j()
        t_start = time.perf_counter()

        frames = 0
        missed = 0
        period = 1.0 / float(target_fps)
        next_deadline = time.perf_counter()
        t_end = t_start + float(duration_s)

        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            workload.step()
            workload.synchronize()
            t1 = time.perf_counter()
            frames += workload.batch
            if (t1 - t0) > period:
                missed += workload.batch

            next_deadline += period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_deadline = time.perf_counter()

        e1 = self.read_energy_j()
        t1 = time.perf_counter()
        power = (e1 - e0) / max(1e-9, t1 - t_start)
        active_fps = frames / max(1e-9, t1 - t_start)
        miss_rate = missed / max(1, frames)
        return power, active_fps, miss_rate


class NVMLReader(PowerReader):
    def __init__(self, gpu_index: int = 0) -> None:
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize NVML. Install nvidia-ml-py/pynvml and run on a discrete NVIDIA GPU."
            ) from e

    def read_power_w(self) -> float:
        return float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0


class JetsonJtopTotalReader(PowerReader):
    def __init__(self) -> None:
        try:
            from jtop import jtop
        except Exception as e:
            raise RuntimeError("jtop is required on Jetson. Install with: sudo -H pip install -U jetson-stats") from e
        self._jtop_cls = jtop
        self.jetson = jtop()
        self.jetson.start()
        time.sleep(0.5)

    def close(self) -> None:
        try:
            self.jetson.close()
        except Exception:
            pass

    def read_power_w(self) -> float:
        stats = self.jetson.stats
        # Preferred: jtop stats usually exposes total power as power cur in mW.
        for key in ("Power TOT", "power cur", "POM_5V_IN", "VDD_IN"):
            if key in stats:
                val = float(stats[key])
                return val / 1000.0 if val > 100 else val
        # Fallback: direct power dict.
        p = getattr(self.jetson, "power", {})
        tot = p.get("tot", {}) if isinstance(p, dict) else {}
        if "power" in tot:
            val = float(tot["power"])
            return val / 1000.0 if val > 100 else val
        raise RuntimeError("Could not locate Jetson total power in jtop stats/power.")


# -----------------------------
# Benchmark protocol
# -----------------------------

def set_torch_threads(n: Optional[int]) -> None:
    if n and n > 0:
        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(max(1, min(4, int(n))))


def measure_back_to_back_latency_ms(workload: PoissonWorkload, n_frames: int) -> List[float]:
    lat = []
    for _ in range(int(n_frames)):
        lat.append(workload.step_timed_ms())
    return lat


def benchmark_resolution(
    *,
    platform: str,
    device_name: str,
    torch_device: torch.device,
    power_reader: PowerReader,
    resolution: int,
    batch: int,
    seed: int,
    latency_preheat_s: float,
    latency_frames: int,
    latency_percentile_for_fps: str,
    platform_fps_cap: float,
    candidate_fps: Sequence[float],
    idle_settle_s: float,
    idle_measure_s: float,
    active_settle_s: float,
    active_measure_s: float,
    sample_interval_s: float,
) -> Dict[str, object]:
    workload = PoissonWorkload(resolution, batch, torch_device, seed=seed)

    # Warm up / preheat with no sleep. This is the key anti-DVFS step for CPU.
    print(f"[{platform}] {resolution}×{resolution}: latency preheat {latency_preheat_s:.1f}s, no sleep")
    workload.run_continuous_for_seconds(latency_preheat_s)

    print(f"[{platform}] {resolution}×{resolution}: measuring {latency_frames} back-to-back latencies")
    latencies = measure_back_to_back_latency_ms(workload, latency_frames)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    pct_map = {"p50": p50, "p95": p95, "p99": p99}
    key = latency_percentile_for_fps.lower()
    if key not in pct_map:
        raise ValueError("--latency-percentile-for-fps must be p50, p95 or p99")
    selected_latency = pct_map[key]
    latency_limited_fps = 1000.0 / max(selected_latency, 1e-9)
    selected_fps = choose_realtime_fps(latency_limited_fps, candidate_fps, platform_fps_cap)

    print(
        f"[{platform}] {resolution}×{resolution}: "
        f"p50={p50:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms, "
        f"latency-limited={latency_limited_fps:.2f} FPS, selected={selected_fps:.2f} FPS"
    )

    print(f"[{platform}] {resolution}×{resolution}: idle settle {idle_settle_s:.1f}s")
    time.sleep(idle_settle_s)
    print(f"[{platform}] {resolution}×{resolution}: idle measure {idle_measure_s:.1f}s")
    idle_power = power_reader.measure_idle_power_w(idle_measure_s, sample_interval_s)

    print(f"[{platform}] {resolution}×{resolution}: active settle {active_settle_s:.1f}s @ {selected_fps:.2f} FPS")
    workload.run_paced_for_seconds(active_settle_s, selected_fps)

    print(f"[{platform}] {resolution}×{resolution}: active measure {active_measure_s:.1f}s @ {selected_fps:.2f} FPS")
    active_power, active_fps, miss_rate = power_reader.measure_active_power_w(
        active_measure_s, sample_interval_s, workload, selected_fps
    )

    incremental = active_power - idle_power
    # Panel e/f full-device operating-point energy. Incremental can be recomputed from CSV if needed.
    energy_mj = active_power / max(active_fps, 1e-9) * 1000.0

    return {
        "platform": platform,
        "device": device_name,
        "resolution": int(resolution),
        "label": f"{int(resolution)}×{int(resolution)}",
        "pixels": int(resolution) * int(resolution),
        "latency_ms": selected_latency,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "latency_percentile_for_fps": key,
        "latency_limited_fps": latency_limited_fps,
        "selected_realtime_fps": selected_fps,
        "active_fps": active_fps,
        "missed_deadline_rate": miss_rate,
        "active_power_w": active_power,
        "idle_power_w": idle_power,
        "incremental_power_w": incremental,
        "energy_per_frame_mj": energy_mj,
        "measure_mode": "real_time_latency_limited_fps_v2",
        "notes": "latency measured back-to-back after no-sleep preheat; no 96/192 resolution",
    }


def build_parser(platform: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Panel e/f real-time latency-limited benchmark for {platform}")
    p.add_argument("--out-dir", type=str, default="panel_ef_measurements")
    p.add_argument("--template-dir", type=str, default="template")
    p.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS_DEFAULT)
    p.add_argument("--candidate-fps", type=float, nargs="+", default=CANDIDATE_FPS_DEFAULT)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latency-preheat-s", type=float, default=10.0)
    p.add_argument("--latency-frames", type=int, default=300)
    p.add_argument("--latency-percentile-for-fps", type=str, default="p99", choices=["p50", "p95", "p99"])
    p.add_argument("--idle-settle-s", type=float, default=10.0)
    p.add_argument("--idle-measure-s", type=float, default=10.0)
    p.add_argument("--active-settle-s", type=float, default=10.0)
    p.add_argument("--active-measure-s", type=float, default=10.0)
    p.add_argument("--sample-interval-s", type=float, default=0.2)
    p.add_argument("--platform-fps-cap", type=float, default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--rapl-path", type=str, default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    return p


def run_suite(
    *,
    platform: str,
    device_name: str,
    torch_device: torch.device,
    power_reader: PowerReader,
    args: argparse.Namespace,
    default_fps_cap: float,
    stem: str,
) -> None:
    set_torch_threads(args.torch_threads)
    fps_cap = float(args.platform_fps_cap) if args.platform_fps_cap is not None else float(default_fps_cap)

    rows: List[Dict[str, object]] = []
    try:
        for res in args.resolutions:
            row = benchmark_resolution(
                platform=platform,
                device_name=device_name,
                torch_device=torch_device,
                power_reader=power_reader,
                resolution=int(res),
                batch=int(args.batch),
                seed=int(args.seed),
                latency_preheat_s=float(args.latency_preheat_s),
                latency_frames=int(args.latency_frames),
                latency_percentile_for_fps=str(args.latency_percentile_for_fps),
                platform_fps_cap=fps_cap,
                candidate_fps=[float(x) for x in args.candidate_fps],
                idle_settle_s=float(args.idle_settle_s),
                idle_measure_s=float(args.idle_measure_s),
                active_settle_s=float(args.active_settle_s),
                active_measure_s=float(args.active_measure_s),
                sample_interval_s=float(args.sample_interval_s),
            )
            rows.append(row)
    finally:
        if hasattr(power_reader, "close"):
            try:
                power_reader.close()
            except Exception:
                pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_platform_csv = out_dir / f"{stem}_panel_ef_400fps.csv"
    write_rows(per_platform_csv, rows, CSV_HEADER)

    template_dir = Path(args.template_dir)
    # Both panel e and f consume the same measured rows, but keeping two files
    # preserves the current Fig.3 script interface.
    write_rows(template_dir / "fig3e_pareto_points_template.csv", rows, CSV_HEADER)
    write_rows(template_dir / "fig3f_resolution_scaling_template.csv", rows, CSV_HEADER)

    print(f"\nSaved: {per_platform_csv}")
    print(f"Updated: {template_dir / 'fig3e_pareto_points_template.csv'}")
    print(f"Updated: {template_dir / 'fig3f_resolution_scaling_template.csv'}")
