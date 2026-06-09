# Fig. 3a real CDF data workflow

## CSV template
`fig3a_latency_cdf_template.csv` is a raw-sample template. Each row is one measured frame/batch. The plotting script constructs the empirical CDF from `latency_ms`.

Required columns for Fig. 3a:

- `platform`: platform name used for legend grouping.
- `frame_id`: measured frame index after warm-up.
- `latency_ms`: measured latency in milliseconds.

Recommended metadata columns:

- `resolution_width`, `resolution_height`, `fps_target`, `batch_size`
- `backend`, `device_name`, `measurement_scope`, `include_transfers`
- `timestamp_s`, `warmup`, `num_iters`, `notes`

Do not use only p50/p95/p99 for panel a. Percentiles are useful annotations, but they cannot reconstruct the CDF.

## Collection examples

Jetson including H2D + compute + D2H:

```bash
python collect_panel_a_latency.py --platform "Jetson Orin NX" --backend cuda \
  --measurement-scope h2d_compute_d2h --width 128 --height 128 \
  --num-iters 1000 --warmup 100 --out-dir fig3a_jetson
```

RTX 4090 compute-only:

```bash
python collect_panel_a_latency.py --platform "RTX 4090" --backend cuda \
  --measurement-scope compute_only --width 128 --height 128 \
  --num-iters 1000 --warmup 100 --out-dir fig3a_rtx4090
```

CPU baseline:

```bash
python collect_panel_a_latency.py --platform "CPU i7-12700" --backend cpu \
  --num-threads 20 --width 128 --height 128 \
  --num-iters 1000 --warmup 100 --out-dir fig3a_cpu
```

## Merge per-platform samples

```bash
python merge_fig3a_latency_samples.py \
  fig3a_jetson/*_fig3a_latency_samples.csv \
  fig3a_rtx4090/*_fig3a_latency_samples.csv \
  fig3a_cpu/*_fig3a_latency_samples.csv \
  --out fig3a_latency_cdf_filled.csv
```

## Plot panel a directly from raw samples

```bash
python plot_panel_a_cdf_real.py --input fig3a_latency_cdf_filled.csv --out-prefix Fig3a_real_cdf
```
