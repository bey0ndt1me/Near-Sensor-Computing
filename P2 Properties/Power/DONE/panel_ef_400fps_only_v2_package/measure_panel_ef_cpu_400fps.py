#!/usr/bin/env python3
"""
CPU panel e/f measurement, 400-FPS only.

Run:
    python measure_panel_ef_cpu_400fps.py --out-dir panel_ef_measurements

This script measures latency back-to-back, then measures idle/active full-chip
power with the active workload paced at exactly 400 FPS.
"""
import torch

from panel_ef_measure_common_400fps import (
    build_parser,
    run_suite,
    RAPLPackageReader,
)

def main():
    parser = build_parser("CPU i7-13700")
    args = parser.parse_args()
    args.candidate_fps = [400.0]
    args.platform_fps_cap = 400.0
    # Keep default resolutions: 64, 128, 256. Override with --resolutions if needed.

    power_reader = RAPLPackageReader(args.rapl_path)
    run_suite(
        platform="CPU",
        device_name="i7-13700",
        torch_device=torch.device("cpu"),
        power_reader=power_reader,
        args=args,
        default_fps_cap=400.0,
        stem="cpu_i7_13700",
    )

if __name__ == "__main__":
    main()
