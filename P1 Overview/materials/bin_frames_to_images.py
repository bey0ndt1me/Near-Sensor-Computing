#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bin_frames_to_images.py
=======================

将 VDMA 采集得到的 frame_*.bin 批量读取并保存为图片。

适配本数据包中的帧格式：
    28 Byte header
    + 128 * 128 * 3 Byte VDMA0 图像数据，默认 BGR 排列
    + 128 * 128 * 3 Byte VDMA1 深度数据，默认 sfix24_en17 小端排列

默认输出：
    output/rgb/frame_xxxxxx.png             原始 RGB 图像
    output/depth_heatmap/frame_xxxxxx.png   深度热力图 PNG
    output/depth_gray16/frame_xxxxxx.png    深度归一化 16-bit 灰度 PNG

示例：
    # 读取解压后的目录
    python bin_frames_to_images.py --input ./video/data/test_1 --output ./data_ana/test_1

    # 直接读取 zip 文件
    python bin_frames_to_images.py --input ./video.zip --output ./data_ana/test_1

    # 只处理前 20 帧，并额外输出四联图分析图
    python bin_frames_to_images.py --input ./video.zip --output ./data_ana/test_1 --start 0 --end 20 --composite

依赖：
    pip install numpy pillow matplotlib
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FrameConfig:
    width: int = 128
    height: int = 128
    header_size: int = 28
    rgb_order: str = "bgr"       # bin 中 VDMA0 的通道顺序：bgr 或 rgb
    depth_endian: str = "little" # VDMA1 三字节深度的字节序：little 或 big
    frac_bits: int = 17          # sfix24_en17 中的小数位数

    @property
    def rgb_size(self) -> int:
        return self.width * self.height * 3

    @property
    def depth_size(self) -> int:
        return self.width * self.height * 3

    @property
    def min_file_size(self) -> int:
        return self.header_size + self.rgb_size


@dataclass
class FrameData:
    name: str
    frame_id: int
    rgb: np.ndarray              # uint8, shape = H x W x 3, RGB
    depth: Optional[np.ndarray]  # float64, shape = H x W; None 表示无深度数据


def parse_frame_id_from_name(name: str) -> int:
    """当 header 中 frame_id 不可靠时，从 frame_000001.bin 文件名中提取编号。"""
    m = re.search(r"frame_(\d+)", Path(name).name)
    return int(m.group(1)) if m else -1


def decode_sfix24_en17(v1_data: bytes, cfg: FrameConfig) -> np.ndarray:
    """将 3 Byte signed fixed-point depth 解码为 float depth。"""
    b = np.frombuffer(v1_data, dtype=np.uint8).reshape(-1, 3)

    if cfg.depth_endian == "little":
        raw = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
    elif cfg.depth_endian == "big":
        raw = (
            b[:, 2].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 0].astype(np.int32) << 16)
        )
    else:
        raise ValueError(f"Unsupported depth_endian: {cfg.depth_endian}")

    # 24-bit 有符号数符号扩展
    raw = np.where(raw & 0x800000, raw - 0x1000000, raw)
    depth = raw.astype(np.float64) / float(2 ** cfg.frac_bits)
    return depth.reshape(cfg.height, cfg.width)


def parse_frame_bytes(name: str, data: bytes, cfg: FrameConfig) -> Optional[FrameData]:
    """解析一个 frame_*.bin 的二进制内容。"""
    if len(data) < cfg.min_file_size:
        print(f"[skip] {name}: file too small, {len(data)} bytes")
        return None

    header = data[:cfg.header_size]
    payload = data[cfg.header_size:]

    # 当前文档代码使用大端读取 frame_id；若异常则回退到文件名编号。
    try:
        frame_id = struct.unpack(">I", header[4:8])[0]
    except Exception:
        frame_id = parse_frame_id_from_name(name)

    if frame_id < 0 or frame_id > 10_000_000:
        frame_id = parse_frame_id_from_name(name)

    v0_data = payload[:cfg.rgb_size]
    v1_data = payload[cfg.rgb_size:cfg.rgb_size + cfg.depth_size]

    img = np.frombuffer(v0_data, dtype=np.uint8).reshape(cfg.height, cfg.width, 3)
    if cfg.rgb_order == "bgr":
        rgb = img[:, :, ::-1].copy()
    elif cfg.rgb_order == "rgb":
        rgb = img.copy()
    else:
        raise ValueError(f"Unsupported rgb_order: {cfg.rgb_order}")

    depth = None
    if len(v1_data) == cfg.depth_size:
        depth = decode_sfix24_en17(v1_data, cfg)

    return FrameData(name=name, frame_id=frame_id, rgb=rgb, depth=depth)


def iter_bin_files(input_path: Path) -> Iterator[Tuple[str, bytes]]:
    """支持读取目录或 zip 文件，返回 (文件名, 二进制内容)。"""
    if input_path.is_dir():
        files = sorted(input_path.rglob("frame_*.bin"))
        for f in files:
            yield str(f), f.read_bytes()
    elif input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as zf:
            names = sorted(n for n in zf.namelist() if Path(n).name.startswith("frame_") and n.endswith(".bin"))
            for name in names:
                yield name, zf.read(name)
    elif input_path.is_file() and input_path.suffix.lower() == ".bin":
        yield str(input_path), input_path.read_bytes()
    else:
        raise FileNotFoundError(f"Unsupported input path: {input_path}")


def normalize_to_uint16(depth: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """将 depth 线性归一化为 uint16，便于无损保存为 16-bit PNG。"""
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return np.zeros(depth.shape, dtype=np.uint16)

    lo = float(np.min(valid)) if vmin is None else float(vmin)
    hi = float(np.max(valid)) if vmax is None else float(vmax)
    if abs(hi - lo) < 1e-12:
        return np.zeros(depth.shape, dtype=np.uint16)

    y = (depth - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 65535.0 + 0.5).astype(np.uint16)


def save_rgb(rgb: np.ndarray, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_file)


def save_depth_gray16(depth: np.ndarray, out_file: Path, vmin: Optional[float], vmax: Optional[float]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    gray16 = normalize_to_uint16(depth, vmin=vmin, vmax=vmax)
    Image.fromarray(gray16).save(out_file)


def save_depth_heatmap(depth: np.ndarray, out_file: Path, vmin: Optional[float], vmax: Optional[float], dpi: int = 150) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4, 4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(depth, cmap="turbo", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_composite(frame: FrameData, out_file: Path, vmin: Optional[float], vmax: Optional[float], dpi: int, stride: int) -> None:
    """保存 RGB + depth heatmap + histogram + 3D surface 四联图。"""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 9), dpi=dpi)
    fig.suptitle(f"Frame #{frame.frame_id:06d}", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(frame.rgb, interpolation="nearest")
    ax1.set_title("Raw RGB")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    if frame.depth is None:
        for idx in [2, 3, 4]:
            ax = fig.add_subplot(2, 2, idx)
            ax.text(0.5, 0.5, "No depth data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        return

    depth = frame.depth
    valid = depth[np.isfinite(depth)]
    d_mean = float(np.mean(valid)) if valid.size else 0.0
    d_std = float(np.std(valid)) if valid.size else 0.0
    d_min = float(np.min(valid)) if valid.size else 0.0
    d_max = float(np.max(valid)) if valid.size else 0.0

    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(depth, cmap="turbo", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax2.set_title(f"Depth heatmap [{d_min:.4f}, {d_max:.4f}]")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(valid.ravel(), bins=80, alpha=0.9)
    ax3.axvline(d_mean, linewidth=1.2, linestyle="--", label=f"mean={d_mean:.4f}")
    ax3.axvline(d_mean - d_std, linewidth=0.8, linestyle=":", label=f"std={d_std:.4f}")
    ax3.axvline(d_mean + d_std, linewidth=0.8, linestyle=":")
    ax3.legend(fontsize=8)
    ax3.set_title("Depth histogram")
    ax3.set_xlabel("depth value")
    ax3.set_ylabel("count")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    xs = np.arange(0, depth.shape[1], stride)
    ys = np.arange(0, depth.shape[0], stride)
    X, Y = np.meshgrid(xs, ys)
    Z = depth[::stride, ::stride]
    ax4.plot_surface(X, Y, Z, cmap="turbo", edgecolor="none", antialiased=False)
    ax4.set_title("Depth 3D surface")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("depth")
    ax4.view_init(elev=35, azim=-45)

    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def pre_scan_global_depth_range(items: list[Tuple[str, bytes]], cfg: FrameConfig) -> Tuple[Optional[float], Optional[float]]:
    """预扫描所有帧，得到全局 depth min/max，用于跨帧一致的色标。"""
    mins, maxs = [], []
    for name, data in items:
        frame = parse_frame_bytes(name, data, cfg)
        if frame is not None and frame.depth is not None:
            valid = frame.depth[np.isfinite(frame.depth)]
            if valid.size:
                mins.append(float(np.min(valid)))
                maxs.append(float(np.max(valid)))
    if not mins:
        return None, None
    return min(mins), max(maxs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VDMA frame_*.bin files to PNG images.")
    parser.add_argument("--input", "-i", required=True, help="输入目录、单个 .bin 文件或 .zip 文件")
    parser.add_argument("--output", "-o", default="./data_ana/test_1", help="输出目录")
    parser.add_argument("--start", type=int, default=0, help="起始帧序号，按排序后的文件列表切片，默认 0")
    parser.add_argument("--end", type=int, default=None, help="结束帧序号，不包含该帧；默认处理到最后")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--header-size", type=int, default=28)
    parser.add_argument("--rgb-order", choices=["bgr", "rgb"], default="bgr", help="VDMA0 数据的通道顺序，默认 bgr")
    parser.add_argument("--depth-endian", choices=["little", "big"], default="little", help="VDMA1 24-bit 深度字节序，默认 little")
    parser.add_argument("--frac-bits", type=int, default=17, help="sfix24_en17 小数位数，默认 17")
    parser.add_argument("--depth-scale", choices=["frame", "global"], default="global", help="深度图归一化范围：每帧独立或全局一致")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--stride", type=int, default=4, help="composite 中 3D 曲面下采样步长")
    parser.add_argument("--no-rgb", action="store_true", help="不保存 RGB 图片")
    parser.add_argument("--no-depth", action="store_true", help="不保存深度热力图和 16-bit 灰度图")
    parser.add_argument("--composite", action="store_true", help="额外保存四联图分析图")
    args = parser.parse_args()

    cfg = FrameConfig(
        width=args.width,
        height=args.height,
        header_size=args.header_size,
        rgb_order=args.rgb_order,
        depth_endian=args.depth_endian,
        frac_bits=args.frac_bits,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)

    items = list(iter_bin_files(input_path))
    if not items:
        raise RuntimeError(f"未找到 frame_*.bin: {input_path}")

    items = items[args.start:args.end]
    print(f"Found {len(items)} frame(s).")

    global_vmin, global_vmax = (None, None)
    if args.depth_scale == "global" and not args.no_depth:
        print("Scanning global depth range ...")
        global_vmin, global_vmax = pre_scan_global_depth_range(items, cfg)
        print(f"Global depth range: vmin={global_vmin}, vmax={global_vmax}")

    ok = 0
    for idx, (name, data) in enumerate(items, start=1):
        frame = parse_frame_bytes(name, data, cfg)
        if frame is None:
            continue

        stem = f"frame_{frame.frame_id:06d}" if frame.frame_id >= 0 else Path(name).stem
        vmin, vmax = global_vmin, global_vmax
        if args.depth_scale == "frame" and frame.depth is not None:
            valid = frame.depth[np.isfinite(frame.depth)]
            if valid.size:
                vmin, vmax = float(np.min(valid)), float(np.max(valid))

        if not args.no_rgb:
            save_rgb(frame.rgb, output_dir / "rgb" / f"{stem}.png")

        if not args.no_depth and frame.depth is not None:
            save_depth_heatmap(frame.depth, output_dir / "depth_heatmap" / f"{stem}.png", vmin, vmax, dpi=args.dpi)
            save_depth_gray16(frame.depth, output_dir / "depth_gray16" / f"{stem}.png", vmin, vmax)

        if args.composite:
            save_composite(frame, output_dir / "composite" / f"{stem}.png", vmin, vmax, dpi=args.dpi, stride=args.stride)

        ok += 1
        print(f"[{idx:04d}/{len(items):04d}] saved {stem}", end="\r")

    print(f"\nDone. Saved {ok} frame(s) to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
