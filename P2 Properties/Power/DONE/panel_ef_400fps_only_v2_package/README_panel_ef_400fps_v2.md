# Panel e/f 400-FPS-only package v2

This version fixes the previous import error:

```text
ImportError: cannot import name 'args_to_cfg'
```

The previous wrapper expected `args_to_cfg` / `run_cpu_suite`, but the common module actually exposes `build_parser`, `run_suite`, `RAPLPackageReader`, and `NVMLReader`.

## Run

CPU:

```bash
python measure_panel_ef_cpu_400fps.py --out-dir panel_ef_measurements
```

GPU:

```bash
python measure_panel_ef_gpu_400fps.py --out-dir panel_ef_measurements
```

## Protocol

- Resolutions: 64, 128, 256.
- Latency is measured back-to-back and saved.
- Active-power measurement is forced to exactly 400 FPS.
- Output CSVs:
  - `cpu_i7_13700_panel_ef_400fps.csv`
  - `gpu_rtx4090_panel_ef_400fps.csv`
- The script also updates:
  - `template/fig3e_pareto_points_template.csv`
  - `template/fig3f_resolution_scaling_template.csv`
