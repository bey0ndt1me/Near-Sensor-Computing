#!/usr/bin/env python3
"""
VDMA 逐帧分析工具
==================
对每一帧生成一张包含 4 个子图的 PNG:
  1. 原始帧 RGB
  2. 深度 2D 热力图
  3. 深度值直方图
  4. 深度 3D 曲面图

用法:
    python frame_analysis.py ./data/depth_test/delay_test_002
    python frame_analysis.py ./data/depth_test/delay_test_002 --start 10 --end 20
    python frame_analysis.py ./data/depth_test/delay_test_002 --dpi 120
"""

import os
import sys
import glob
import struct
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头渲染，不弹窗
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============== 参数 ==============
FRAME_WIDTH  = 128
FRAME_HEIGHT = 128
VDMA0_PIXEL_BYTES = 3
VDMA1_PIXEL_BYTES = 3
VDMA0_SIZE = FRAME_WIDTH * FRAME_HEIGHT * VDMA0_PIXEL_BYTES
VDMA1_SIZE = FRAME_WIDTH * FRAME_HEIGHT * VDMA1_PIXEL_BYTES
HEADER_SIZE = 28


def parse_frame_file(filepath):
    """读取单个 frame bin 文件"""
    with open(filepath, 'rb') as f:
        header = f.read(HEADER_SIZE)
        frame_data = f.read()

    if len(header) < HEADER_SIZE:
        return None

    magic = struct.unpack('>I', header[0:4])[0]
    frame_id = struct.unpack('>I', header[4:8])[0]

    v0_data = frame_data[:VDMA0_SIZE]
    v1_data = frame_data[VDMA0_SIZE:VDMA0_SIZE + VDMA1_SIZE]

    if len(v0_data) < VDMA0_SIZE:
        return None

    # VDMA0: BGR -> RGB
    v0_bgr = np.frombuffer(v0_data, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
    v0_rgb = v0_bgr[:, :, ::-1].copy()

    # VDMA1: sfix24_en17
    depth = None
    if len(v1_data) == VDMA1_SIZE:
        v1_bytes = np.frombuffer(v1_data, dtype=np.uint8).reshape(-1, 3)
        # raw = (v1_bytes[:, 0].astype(np.int32)
        #        | (v1_bytes[:, 1].astype(np.int32) << 8)
        #        | (v1_bytes[:, 2].astype(np.int32) << 16))
        raw = (v1_bytes[:, 0].astype(np.int32)            # byte2 → 最低位
                | (v1_bytes[:, 1].astype(np.int32) << 8)
                | (v1_bytes[:, 2].astype(np.int32) << 16))    # byte0 → 最高位
        raw = np.where(raw & 0x800000, raw - 0x1000000, raw)
        depth = raw.astype(np.float64) / (2 ** 17)
        depth = depth.reshape((FRAME_HEIGHT, FRAME_WIDTH))

    return {
        'frame_id': frame_id,
        'rgb': v0_rgb,
        'depth': depth,
        'path': filepath,
    }


def collect_frames(data_dir, start=None, end=None):
    pattern = os.path.join(data_dir, 'frame_*.bin')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"错误: 在 {data_dir} 中没有找到 frame_*.bin 文件")
        sys.exit(1)
    print(f"找到 {len(files)} 帧文件")
    if start is not None or end is not None:
        s = start or 0
        e = end or len(files)
        files = files[s:e]
        print(f"截取范围: [{s}:{e}]，共 {len(files)} 帧")
    return files


def plot_frame(info, output_path, dpi=100, stride=4):
    """
    画一帧的 4 个子图并保存为 PNG

    Parameters:
        info: parse_frame_file 返回的字典
        output_path: 输出 PNG 路径
        dpi: 分辨率
        stride: 3D 曲面下采样步长 (加速渲染)
    """
    fid = info['frame_id']
    rgb = info['rgb']
    depth = info['depth']

    fig = plt.figure(figsize=(14, 10), facecolor='#1a1a1a')
    fig.suptitle(f'Frame #{fid:06d}', color='white', fontsize=14, fontweight='bold')

    # ---- 1. 原始帧 RGB ----
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(rgb, interpolation='nearest')
    ax1.set_title('Raw RGB (VDMA0)', color='white', fontsize=11)
    ax1.set_xlabel('x', color='#aaa', fontsize=9)
    ax1.set_ylabel('y', color='#aaa', fontsize=9)
    ax1.tick_params(colors='#888', labelsize=8)

    if depth is None:
        # 没有深度数据时，其余三个子图留空
        for pos in [2, 3, 4]:
            ax = fig.add_subplot(2, 2, pos)
            ax.text(0.5, 0.5, 'No Depth Data', ha='center', va='center',
                    color='#666', fontsize=12, transform=ax.transAxes)
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='#888')
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        return

    d_min, d_max = np.nanmin(depth), np.nanmax(depth)
    d_mean, d_std = np.nanmean(depth), np.nanstd(depth)

    # ---- 2. 深度 2D 热力图 ----
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(depth, cmap='turbo', interpolation='nearest')
    ax2.set_title(f'Depth 2D  [{d_min:.4f}, {d_max:.4f}]', color='white', fontsize=11)
    ax2.set_xlabel('x', color='#aaa', fontsize=9)
    ax2.set_ylabel('y', color='#aaa', fontsize=9)
    ax2.tick_params(colors='#888', labelsize=8)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#888', labelsize=8)

    # ---- 3. 深度值直方图 ----
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#222')
    valid = depth[np.isfinite(depth)].flatten()
    if len(valid) > 0:
        ax3.hist(valid, bins=80, color='#00aaff', edgecolor='none', alpha=0.85)
        # 均值线
        ax3.axvline(d_mean, color='#ff4444', linewidth=1.2, linestyle='--',
                    label=f'mean={d_mean:.4f}')
        ax3.axvline(d_mean - d_std, color='#ffaa00', linewidth=0.8, linestyle=':',
                    label=f'std={d_std:.4f}')
        ax3.axvline(d_mean + d_std, color='#ffaa00', linewidth=0.8, linestyle=':')
        ax3.legend(fontsize=8, facecolor='#333', edgecolor='#555', labelcolor='white')
    ax3.set_title('Depth Histogram', color='white', fontsize=11)
    ax3.set_xlabel('depth value', color='#aaa', fontsize=9)
    ax3.set_ylabel('count', color='#aaa', fontsize=9)
    ax3.tick_params(colors='#888', labelsize=8)

    # ---- 4. 深度 3D 曲面图 ----
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.set_facecolor('#1a1a1a')

    # 下采样加速
    X = np.arange(0, FRAME_WIDTH, stride)
    Y = np.arange(0, FRAME_HEIGHT, stride)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = depth[::stride, ::stride]

    ax4.plot_surface(X_grid, Y_grid, Z, cmap='turbo',
                     edgecolor='none', alpha=0.9,
                     rstride=1, cstride=1, antialiased=False)
    ax4.set_title('Depth 3D Surface', color='white', fontsize=11, pad=0)
    ax4.set_xlabel('x', color='#aaa', fontsize=8, labelpad=2)
    ax4.set_ylabel('y', color='#aaa', fontsize=8, labelpad=2)
    ax4.set_zlabel('depth', color='#aaa', fontsize=8, labelpad=2)
    ax4.tick_params(colors='#888', labelsize=7, pad=0)
    ax4.view_init(elev=35, azim=-45)
    # 背景颜色
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.xaxis.pane.set_edgecolor('#333')
    ax4.yaxis.pane.set_edgecolor('#333')
    ax4.zaxis.pane.set_edgecolor('#333')

    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def main():
    data_dir = './data/test_1'
    start = 0
    end = None
    dpi = 100
    stride = 4
    out_dir = f'./data_ana/test_1'
    os.makedirs(out_dir, exist_ok=True)
    files = collect_frames(data_dir, start, end)

    for i, fpath in enumerate(files):
        info = parse_frame_file(fpath)
        if info is None:
            print(f"  跳过损坏文件: {fpath}")
            continue

        fid = info['frame_id']
        out_path = os.path.join(out_dir, f'frame_{fid:06d}.png')
        plot_frame(info, out_path, dpi=dpi, stride=stride)

        print(f"  [{i+1}/{len(files)}] frame_{fid:06d}.png", end='\r')

    print(f"\n\n完成! {len(files)} 张 PNG 已保存到: {out_dir}")


if __name__ == '__main__':
    main()
