# Fig.3a Jetson Orin latency collection

This package provides a Jetson-specific CDF data collector adapted from `keep_running.py`.

## Why this script is different from RTX GPU scripts

Jetson Orin is an integrated SoC. CPU and GPU share system DRAM, so the transfer path is not equivalent to a discrete RTX GPU over PCIe. Therefore:

- `compute_only`: measures only the CUDA-resident Poisson reconstruction path.
- `app_e2e`: measures CPU-side tensor preparation/copy to CUDA memory, GPU reconstruction and output materialization. This is usually the better Fig.3a Jetson number if the application pipeline starts from CPU/camera-side buffers.

## Recommended Fig.3a Jetson command

```bash
python collect_panel_a_latency_jetson_orin.py \
  --platform "Jetson Orin NX" \
  --width 128 --height 128 \
  --batch-size 1 \
  --num-iters 1000 \
  --warmup 100 \
  --measurement-scope app_e2e \
  --log-power \
  --out-dir fig3a_jetson_latency
```

## Pure compute-only command

```bash
python collect_panel_a_latency_jetson_orin.py \
  --platform "Jetson Orin NX" \
  --width 128 --height 128 \
  --batch-size 1 \
  --num-iters 1000 \
  --warmup 100 \
  --measurement-scope compute_only \
  --out-dir fig3a_jetson_latency_compute_only
```

## Outputs

- `*_fig3a_latency_samples.csv`: one row per frame, directly used for real CDF.
- `*_latency_summary.csv`: contains `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`.

Use the samples CSV for Fig.3a CDF, not only the summary percentiles.

compute_only = 输入已经在 CUDA tensor 中，只测 GPU Poisson reconstruction
app_e2e      = CPU tensor → CUDA tensor → GPU compute → 输出同步到 CPU