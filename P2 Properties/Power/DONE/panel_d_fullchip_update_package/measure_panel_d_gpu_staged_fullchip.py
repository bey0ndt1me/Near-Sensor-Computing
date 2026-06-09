#!/usr/bin/env python3
from panel_d_measure_common_staged_fullchip import build_parser, args_to_cfg, run_cuda_suite, CPU_GPU_TARGET_FPS


def main():
    parser = build_parser(
        default_platform="RTX 4090",
        default_device="cuda",
        default_targets=CPU_GPU_TARGET_FPS,
    )
    args = parser.parse_args()
    cfg = args_to_cfg(args)
    run_cuda_suite(cfg, stem="gpu")


if __name__ == "__main__":
    main()
