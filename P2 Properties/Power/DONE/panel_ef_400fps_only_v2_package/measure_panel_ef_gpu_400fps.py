#!/usr/bin/env python3
"""
GPU panel e/f measurement, 400-FPS only.

Run:
    python measure_panel_ef_gpu_400fps.py --out-dir panel_ef_measurements

This script measures latency back-to-back, then measures idle/active GPU power
with the active workload paced at exactly 400 FPS.
"""
import torch

from panel_ef_measure_common_400fps import (
    build_parser,
    run_suite,
    NVMLReader,
)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on the RTX 4090 machine.")

    parser = build_parser("RTX 4090")
    args = parser.parse_args()
    args.candidate_fps = [400.0]
    args.platform_fps_cap = 400.0

    power_reader = NVMLReader(args.gpu_index)
    run_suite(
        platform="GPU",
        device_name="RTX 4090",
        torch_device=torch.device(f"cuda:{args.gpu_index}"),
        power_reader=power_reader,
        args=args,
        default_fps_cap=400.0,
        stem="gpu_rtx4090",
    )

if __name__ == "__main__":
    main()
