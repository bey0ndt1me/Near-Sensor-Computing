#!/usr/bin/env python3
from panel_d_measure_common_staged_fullchip import build_parser, args_to_cfg, run_cpu_suite, CPU_GPU_TARGET_FPS


def main():
    parser = build_parser(
        default_platform="CPU i7-13700",
        default_device="cpu",
        default_targets=CPU_GPU_TARGET_FPS,
    )
    args = parser.parse_args()
    cfg = args_to_cfg(args)
    run_cpu_suite(cfg)


if __name__ == "__main__":
    main()
