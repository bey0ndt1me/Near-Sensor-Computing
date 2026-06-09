#!/usr/bin/env python3
"""
Panel-d CPU/GPU staged power measurement common code.

Protocol per target FPS:
  1. idle settle
  2. idle measure
  3. active settle
  4. active measure

No latency is measured or exported.
Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch


CPU_GPU_TARGET_FPS = [30, 45, 60, 90, 120, 180, 240, 320, 400]


# -----------------------------------------------------------------------------
# Poisson workload, Python 3.8 compatible
# -----------------------------------------------------------------------------


def dst_torch(x: torch.Tensor, norm: Optional[str] = "ortho", axis: int = -1) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32, device=x.device)
    n = x.shape[axis]
    x_ext = torch.cat([x, -x.flip([axis])], dim=axis)
    idx = torch.arange(1, n + 1, device=x.device)
    out = torch.fft.fft(x_ext, dim=axis).imag.index_select(axis, idx)
    if norm == "ortho":
        out.mul_(math.sqrt(2.0 / n))
        first = [slice(None)] * out.dim()
        first[axis] = 0
        out[tuple(first)].mul_(math.sqrt(2.0))
    return out


def idst_torch(x: torch.Tensor, norm: Optional[str] = "ortho", axis: int = -1) -> torch.Tensor:
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
    def __init__(self, height: int, width: int, batch_size: int, device: torch.device):
        if height < 4 or width < 4:
            raise ValueError("height and width must be at least 4")
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device

        inner_h = height - 2
        inner_w = width - 2
        y, x = np.ogrid[1:inner_h + 1, 1:inner_w + 1]
        denom = (
            (2 * np.cos(math.pi * x / (inner_w + 1)) - 2)
            + (2 * np.cos(math.pi * y / (inner_h + 1)) - 2)
        )
        self.denom = torch.tensor(denom, dtype=torch.float32, device=device).unsqueeze(0)

    def __call__(self, grady: torch.Tensor, gradx: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        gyy = grady[:, 1:, :-1] - grady[:, :-1, :-1]
        gxx = gradx[:, :-1, 1:] - gradx[:, :-1, :-1]

        f = torch.zeros_like(boundary)
        f[:, :-1, 1:] += gxx
        f[:, 1:, :-1] += gyy

        b = boundary.clone()
        b[:, 1:-1, 1:-1] = 0
        f_bp = (
            -4 * b[:, 1:-1, 1:-1]
            + b[:, 1:-1, 2:]
            + b[:, 1:-1, :-2]
            + b[:, 2:, 1:-1]
            + b[:, :-2, 1:-1]
        )
        f_inner = f[:, 1:-1, 1:-1] - f_bp

        tmp = dst_torch(f_inner, norm="ortho", axis=-1)
        fsin = dst_torch(tmp.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)
        solved = fsin / self.denom
        tmp = idst_torch(solved, norm="ortho", axis=-1)
        img = idst_torch(tmp.transpose(-1, -2), norm="ortho", axis=-1).transpose(-1, -2)

        result = b
        result[:, 1:-1, 1:-1] += img
        return result


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(batch: int, height: int, width: int, device: torch.device, seed: int = 1234, pin_memory: bool = False):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    kwargs = dict(dtype=torch.float32, device="cpu")
    if pin_memory and torch.cuda.is_available():
        kwargs["pin_memory"] = True
    gradx = torch.randn(batch, height, width, generator=gen, **kwargs)
    grady = torch.randn(batch, height, width, generator=gen, **kwargs)
    boundary = torch.zeros(batch, height, width, **kwargs)

    if device.type == "cuda":
        return grady, gradx, boundary
    return grady.to(device), gradx.to(device), boundary.to(device)


# -----------------------------------------------------------------------------
# Power readers
# -----------------------------------------------------------------------------


class RAPLPackageReader:
    """Intel CPU package energy reader via /sys/class/powercap/intel-rapl."""
    def __init__(self, energy_path: Optional[str] = None):
        if energy_path is None:
            roots = [Path("/sys/class/powercap/intel-rapl"), Path("/sys/class/powercap")]
            candidates = []
            for root in roots:
                if root.exists():
                    candidates.extend(sorted(root.glob("intel-rapl:*/energy_uj")))
                    candidates.extend(sorted(root.glob("*/energy_uj")))
            preferred = [c for c in candidates if "intel-rapl:0" in str(c)]
            candidates = preferred or candidates
            if not candidates:
                raise RuntimeError(
                    "Could not find RAPL energy_uj. Try --rapl-energy-path "
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
                )
            energy_path = str(candidates[0])

        self.energy_path = Path(energy_path)
        if not self.energy_path.exists():
            raise RuntimeError("RAPL energy file not found: %s" % self.energy_path)

        max_path = self.energy_path.parent / "max_energy_range_uj"
        self.max_energy_j = None
        if max_path.exists():
            try:
                self.max_energy_j = float(max_path.read_text().strip()) / 1e6
            except Exception:
                self.max_energy_j = None

    def read_energy_j(self) -> float:
        return float(self.energy_path.read_text().strip()) / 1e6

    def energy_delta_j(self, e0: float, e1: float) -> float:
        delta = e1 - e0
        if delta < 0 and self.max_energy_j is not None:
            delta += self.max_energy_j
        return delta

    def measure_sleep_power_w(self, duration_s: float) -> Tuple[float, float]:
        t0 = time.perf_counter()
        e0 = self.read_energy_j()
        while time.perf_counter() - t0 < duration_s:
            time.sleep(0.05)
        elapsed = time.perf_counter() - t0
        e1 = self.read_energy_j()
        return self.energy_delta_j(e0, e1) / elapsed, elapsed


class NvmlReader:
    """Desktop NVIDIA GPU power reader via NVML."""
    def __init__(self, index: int = 0):
        self.pynvml = None
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize NVML: %s. Install nvidia-ml-py/pynvml and run on a desktop NVIDIA GPU."
                % e
            )

    def read_power_w(self) -> float:
        return float(self.pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0

    def shutdown(self) -> None:
        if self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


class PowerSampler:
    def __init__(self, reader: NvmlReader, interval_s: float = 0.02, phase: str = "", target_fps: float = 0):
        self.reader = reader
        self.interval_s = float(interval_s)
        self.phase = phase
        self.target_fps = target_fps
        self.samples: List[dict] = []
        self._stop = threading.Event()
        self._t0 = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                t = time.perf_counter()
                p = float(self.reader.read_power_w())
                if self._t0 is not None:
                    self.samples.append({
                        "sample_time_s": t - self._t0,
                        "power_w": p,
                        "phase": self.phase,
                        "target_fps": self.target_fps,
                    })
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def mean_power_w(self) -> float:
        arr = np.asarray([row["power_w"] for row in self.samples], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")


# -----------------------------------------------------------------------------
# Benchmark logic
# -----------------------------------------------------------------------------


@dataclass
class SuiteConfig:
    platform: str
    device: str
    target_fps_list: List[int]
    resolution: int = 128
    batch_size: int = 1
    warmup: int = 50

    idle_settle_s: float = 10.0
    idle_measure_s: float = 10.0
    active_settle_s: float = 10.0
    active_measure_s: float = 10.0

    out_dir: str = "panel_d_measurements"
    include_transfers: bool = True
    no_pin_memory: bool = False
    seed: int = 1234
    num_threads: int = 0
    rapl_energy_path: Optional[str] = None
    power_sample_interval_s: float = 0.02


def prepare_cuda_workload(cfg: SuiteConfig, device: torch.device):
    host = make_inputs(
        cfg.batch_size,
        cfg.resolution,
        cfg.resolution,
        torch.device("cpu"),
        cfg.seed,
        pin_memory=not cfg.no_pin_memory,
    )
    if cfg.include_transfers:
        grady_cpu, gradx_cpu, boundary_cpu = host
        dev = (
            torch.empty_like(grady_cpu, device=device),
            torch.empty_like(gradx_cpu, device=device),
            torch.empty_like(boundary_cpu, device=device),
        )
        return host, dev
    return None, tuple(t.to(device) for t in host)


def run_one_cpu_batch(recon: PoissonReconstructor, tensors) -> None:
    _ = recon(*tensors)


def run_one_cuda_batch(recon: PoissonReconstructor, host_tensors, dev_tensors, include_transfers: bool) -> None:
    grady_dev, gradx_dev, boundary_dev = dev_tensors
    if include_transfers:
        grady_cpu, gradx_cpu, boundary_cpu = host_tensors
        grady_dev.copy_(grady_cpu, non_blocking=False)
        gradx_dev.copy_(gradx_cpu, non_blocking=False)
        boundary_dev.copy_(boundary_cpu, non_blocking=False)

    out = recon(grady_dev, gradx_dev, boundary_dev)

    if include_transfers:
        _ = out.detach().cpu()
    sync_if_cuda(grady_dev.device)


def run_cpu_workload_for_duration(recon, tensors, cfg: SuiteConfig, target_fps: float, duration_s: float, count_frames: bool) -> Tuple[int, float]:
    frames_done = 0
    target_period = cfg.batch_size / target_fps if target_fps > 0 else 0.0
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < duration_s:
        iter0 = time.perf_counter()
        run_one_cpu_batch(recon, tensors)
        if count_frames:
            frames_done += cfg.batch_size

        sleep_s = target_period - (time.perf_counter() - iter0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    elapsed = time.perf_counter() - t0
    return frames_done, elapsed


def run_cuda_workload_for_duration(recon, host_tensors, dev_tensors, cfg: SuiteConfig, target_fps: float, duration_s: float, count_frames: bool) -> Tuple[int, float]:
    frames_done = 0
    target_period = cfg.batch_size / target_fps if target_fps > 0 else 0.0
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < duration_s:
        iter0 = time.perf_counter()
        run_one_cuda_batch(recon, host_tensors, dev_tensors, cfg.include_transfers)
        if count_frames:
            frames_done += cfg.batch_size

        sleep_s = target_period - (time.perf_counter() - iter0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    elapsed = time.perf_counter() - t0
    return frames_done, elapsed


def make_result_row(cfg: SuiteConfig, target_fps: float, processed_fps: float, active_power: float, idle_power: float, frames_done: int, notes: str) -> dict:
    incremental = active_power - idle_power if math.isfinite(active_power) and math.isfinite(idle_power) else float("nan")
    full_chip_energy_mj = 1000.0 * active_power / processed_fps if processed_fps > 0 and math.isfinite(active_power) else float("nan")
    incremental_energy_mj = 1000.0 * incremental / processed_fps if processed_fps > 0 and math.isfinite(incremental) else float("nan")
    return {
        "platform": cfg.platform,
        "point_id": f"{cfg.platform} {target_fps:g} FPS",
        "target_fps": target_fps,
        "processed_fps": processed_fps,
        "active_power_w": active_power,
        "idle_power_w": idle_power,
        "full_chip_power_w": active_power,
        "incremental_power_w": incremental,
        "full_chip_energy_per_frame_mj": full_chip_energy_mj,
        "incremental_energy_per_frame_mj": incremental_energy_mj,
        "bubble_size_w": active_power,
        "energy_basis": "full-chip and incremental",
        "meets_180_fps": "Yes" if processed_fps >= 180.0 else "No",
        "notes": notes,
        "n_frames": frames_done,
        "idle_settle_s": cfg.idle_settle_s,
        "idle_measure_s": cfg.idle_measure_s,
        "active_settle_s": cfg.active_settle_s,
        "active_measure_s": cfg.active_measure_s,
    }


def benchmark_point_cpu(cfg: SuiteConfig, recon: PoissonReconstructor, tensors, rapl: RAPLPackageReader, target_fps: float) -> dict:
    for _ in range(cfg.warmup):
        run_one_cpu_batch(recon, tensors)

    print("  idle settle %.1f s" % cfg.idle_settle_s)
    time.sleep(cfg.idle_settle_s)

    print("  idle measure %.1f s" % cfg.idle_measure_s)
    idle_power, _ = rapl.measure_sleep_power_w(cfg.idle_measure_s)

    print("  active settle %.1f s" % cfg.active_settle_s)
    run_cpu_workload_for_duration(recon, tensors, cfg, target_fps, cfg.active_settle_s, count_frames=False)

    print("  active measure %.1f s" % cfg.active_measure_s)
    t0 = time.perf_counter()
    e0 = rapl.read_energy_j()
    frames_done, elapsed = run_cpu_workload_for_duration(recon, tensors, cfg, target_fps, cfg.active_measure_s, count_frames=True)
    e1 = rapl.read_energy_j()
    elapsed = time.perf_counter() - t0

    active_power = rapl.energy_delta_j(e0, e1) / elapsed
    processed_fps = frames_done / max(elapsed, 1e-12)

    return make_result_row(
        cfg,
        target_fps,
        processed_fps,
        active_power,
        idle_power,
        frames_done,
        "CPU Intel RAPL package power; staged 10s settle + 10s measure; no latency exported.",
    )


def benchmark_point_cuda(cfg: SuiteConfig, recon: PoissonReconstructor, host_tensors, dev_tensors, nvml: NvmlReader, target_fps: float):
    for _ in range(cfg.warmup):
        run_one_cuda_batch(recon, host_tensors, dev_tensors, cfg.include_transfers)

    print("  idle settle %.1f s" % cfg.idle_settle_s)
    time.sleep(cfg.idle_settle_s)

    print("  idle measure %.1f s" % cfg.idle_measure_s)
    with PowerSampler(nvml, cfg.power_sample_interval_s, phase="idle_measure", target_fps=target_fps) as idle_sampler:
        time.sleep(cfg.idle_measure_s)
    idle_power = idle_sampler.mean_power_w

    print("  active settle %.1f s" % cfg.active_settle_s)
    run_cuda_workload_for_duration(recon, host_tensors, dev_tensors, cfg, target_fps, cfg.active_settle_s, count_frames=False)

    print("  active measure %.1f s" % cfg.active_measure_s)
    with PowerSampler(nvml, cfg.power_sample_interval_s, phase="active_measure", target_fps=target_fps) as active_sampler:
        frames_done, elapsed = run_cuda_workload_for_duration(
            recon, host_tensors, dev_tensors, cfg, target_fps, cfg.active_measure_s, count_frames=True
        )

    active_power = active_sampler.mean_power_w
    processed_fps = frames_done / max(elapsed, 1e-12)

    row = make_result_row(
        cfg,
        target_fps,
        processed_fps,
        active_power,
        idle_power,
        frames_done,
        "NVML device power; staged 10s settle + 10s measure; no latency exported.",
    )

    samples = []
    samples.extend(idle_sampler.samples)
    samples.extend(active_sampler.samples)
    return row, samples


def _write_dict_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(cfg: SuiteConfig, rows: List[dict], raw_rows: List[dict], stem: str):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / ("%s_panel_d_summary_staged.csv" % stem)
    raw_path = out_dir / ("%s_panel_d_power_samples_staged.csv" % stem)
    template_path = out_dir / ("%s_panel_d_for_template.csv" % stem)

    summary_cols = [
        "platform", "point_id", "target_fps", "processed_fps",
        "active_power_w", "idle_power_w", "full_chip_power_w", "incremental_power_w",
        "full_chip_energy_per_frame_mj", "incremental_energy_per_frame_mj",
        "bubble_size_w", "meets_180_fps", "energy_basis", "notes",
        "n_frames", "idle_settle_s", "idle_measure_s", "active_settle_s", "active_measure_s",
    ]
    raw_cols = ["platform", "target_fps", "phase", "sample_time_s", "power_w"]
    template_cols = [
        "platform", "point_id", "target_fps", "processed_fps",
        "active_power_w", "idle_power_w", "full_chip_power_w", "incremental_power_w",
        "full_chip_energy_per_frame_mj", "incremental_energy_per_frame_mj",
        "bubble_size_w", "meets_180_fps", "energy_basis", "notes",
    ]

    _write_dict_csv(summary_path, rows, summary_cols)
    _write_dict_csv(raw_path, raw_rows, raw_cols)
    _write_dict_csv(template_path, rows, template_cols)
    return summary_path, raw_path, template_path


def run_cpu_suite(cfg: SuiteConfig):
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)

    device = torch.device("cpu")
    recon = PoissonReconstructor(cfg.resolution, cfg.resolution, cfg.batch_size, device)
    tensors = make_inputs(cfg.batch_size, cfg.resolution, cfg.resolution, device, cfg.seed)
    rapl = RAPLPackageReader(cfg.rapl_energy_path)

    rows, raw_rows = [], []
    for target in cfg.target_fps_list:
        print("[%s] target %s FPS ..." % (cfg.platform, target))
        row = benchmark_point_cpu(cfg, recon, tensors, rapl, target)
        rows.append(row)
        print(
            "  processed_fps=%.2f, active=%.3f W, idle=%.3f W, incr=%.3f W, full-chip E/frame=%.4f mJ, incr E/frame=%.4f mJ"
            % (
                row["processed_fps"],
                row["active_power_w"],
                row["idle_power_w"],
                row["incremental_power_w"],
                row["full_chip_energy_per_frame_mj"],
                row["incremental_energy_per_frame_mj"],
            )
        )

    return write_outputs(cfg, rows, raw_rows, "cpu")


def run_cuda_suite(cfg: SuiteConfig, stem: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")
    recon = PoissonReconstructor(cfg.resolution, cfg.resolution, cfg.batch_size, device)
    host_tensors, dev_tensors = prepare_cuda_workload(cfg, device)
    nvml = NvmlReader(0)

    rows, raw_rows = [], []
    try:
        for target in cfg.target_fps_list:
            print("[%s] target %s FPS ..." % (cfg.platform, target))
            row, samples = benchmark_point_cuda(cfg, recon, host_tensors, dev_tensors, nvml, target)
            rows.append(row)
            for s in samples:
                raw_rows.append({
                    "platform": cfg.platform,
                    "target_fps": target,
                    "phase": s["phase"],
                    "sample_time_s": s["sample_time_s"],
                    "power_w": s["power_w"],
                })
            print(
                "  processed_fps=%.2f, active=%.3f W, idle=%.3f W, incr=%.3f W, full-chip E/frame=%.4f mJ, incr E/frame=%.4f mJ"
                % (
                    row["processed_fps"],
                    row["active_power_w"],
                    row["idle_power_w"],
                    row["incremental_power_w"],
                    row["full_chip_energy_per_frame_mj"],
                    row["incremental_energy_per_frame_mj"],
                )
            )
    finally:
        nvml.shutdown()

    return write_outputs(cfg, rows, raw_rows, stem)


def build_parser(default_platform: str, default_device: str, default_targets: Iterable[int]):
    p = argparse.ArgumentParser(description="Fig. 3d staged energy/throughput measurement; no latency is exported.")
    p.add_argument("--platform", default=default_platform)
    p.add_argument("--device", choices=["cpu", "cuda"], default=default_device)
    p.add_argument("--target-fps-list", default=",".join(str(x) for x in default_targets))
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)

    p.add_argument("--idle-settle-s", type=float, default=10.0)
    p.add_argument("--idle-measure-s", type=float, default=10.0)
    p.add_argument("--active-settle-s", type=float, default=10.0)
    p.add_argument("--active-measure-s", type=float, default=10.0)

    p.add_argument("--out-dir", default="panel_d_measurements")
    p.add_argument("--compute-only", action="store_true", help="CUDA only: exclude H2D/D2H transfers from workload loop.")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--rapl-energy-path", default=None, help="Optional RAPL energy_uj path.")
    p.add_argument("--power-sample-interval-s", type=float, default=0.02)
    return p


def args_to_cfg(args) -> SuiteConfig:
    return SuiteConfig(
        platform=args.platform,
        device=args.device,
        target_fps_list=[int(x) for x in str(args.target_fps_list).split(",") if str(x).strip()],
        resolution=args.resolution,
        batch_size=args.batch_size,
        warmup=args.warmup,
        idle_settle_s=args.idle_settle_s,
        idle_measure_s=args.idle_measure_s,
        active_settle_s=args.active_settle_s,
        active_measure_s=args.active_measure_s,
        out_dir=args.out_dir,
        include_transfers=not args.compute_only,
        no_pin_memory=args.no_pin_memory,
        seed=args.seed,
        num_threads=args.num_threads,
        rapl_energy_path=args.rapl_energy_path,
        power_sample_interval_s=args.power_sample_interval_s,
    )
