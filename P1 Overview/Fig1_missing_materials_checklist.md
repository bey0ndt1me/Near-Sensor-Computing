# Fig. 1 缺少的内容 / 数据 / 图片清单

## 全局需要确认
- 最终图中文字是否统一使用 **Nature Sensors / Nature Portfolio** 风格：短标签、少句子、无长段落说明。
- 所有数值是否统一为最终版本：**128 × 128**, **35,107 cycles**, **0.211 ms**, **166 MHz**, **400 FPS input**。
- Depth map 的色标范围是否固定为 **-2 to 2 mm**；如果真实数据范围不同，需要给出统一的 `vmin/vmax`。
- 若图中出现 power/latency 对比，需确认是 **full-chip/device power** 还是 **incremental/core compute power**。

## Panel a：Conventional vs near-sensor reconstruction
已有：
- `materials/raw_rgb.png`
- `materials/my_depth_2d.png`
- `materials/Sensor_Explosion.jpg`

建议补充：
- 一张更像“传感器实物/相机模组”的正视或半透明渲染图，用于替代 exploded view 在 panel a 中重复出现。
- 传统 host-dependent 路线的定量标注：
  - CPU / GPU / Jetson 的典型 latency 范围或 p50/p95/p99；
  - full-chip/device power 或系统功耗；
  - 是否明确写 “host-dependent / non-deterministic latency”。
- near-sensor 路线的最终定量标注：
  - FPGA latency = 0.211 ms；
  - ASIC / FPGA power 选哪个数值放在 Fig.1，还是仅把 power 放到 Fig.3。

## Panel b：Sensor exploded view / cross-section
已有：
- `materials/Sensor_Explosion.jpg`

建议补充：
- 如果投稿主图需要更科学而非产品渲染风格，建议提供：
  - 真实 cross-section schematic，或
  - exploded-view 透明渲染，单独标出 elastomer, diffuser, RGB LEDs, CMOS image sensor, PCB / holder。
- 标注每层厚度或关键尺寸：
  - elastomer thickness；
  - diffuser thickness；
  - LED-to-surface distance；
  - CMOS sensor model: IMX219；
  - field of view / working distance，如有。
- 需要一个真实比例尺或尺寸标注，否则 scale bar 不应写具体长度。

## Panel c：FPGA / ASIC pipeline
当前脚本已使用：
- Photometric stereo
- Divergence
- Forward 2D DST
- Transpose buffer 1
- Spectral division
- Transpose buffer 2
- Inverse 2D DST
- Scaling output
- 35,107 cycles @ 166 MHz = 0.211 ms

建议补充/确认：
- 每个 pipeline stage 的精确 cycle 数和 latency：
  - RGB normalization / photometric stereo
  - divergence
  - column / row DST
  - transpose buffer 1 / 2
  - spectral division
  - inverse DST
  - output scaling
- 若 cycles 的 stage 总和与 total cycles 不完全一致，需要说明是否包含 input/output latency、line buffer、transpose latency 或 control overhead。
- 是否将 “FPGA pipeline” 改写为 “ASIC-compatible streaming pipeline”，避免 Fig.1 过度绑定 FPGA。

## Panel d：Continuous depth reconstruction
当前缺少：
- 真实连续 depth sequence。

建议提供：
- 放入 `materials/depth_sequence/` 的 6-8 张连续 depth map：
  - `depth_000.png`
  - `depth_001.png`
  - ...
  - `depth_007.png`
- 每帧对应时间戳，若使用固定 pipeline latency，可按：
  - 0.000 ms, 0.211 ms, 0.422 ms, ...
- 如果是 sensor 400 FPS 输入，实际相邻输入帧间隔是 2.5 ms；如果 panel d 想表达 pipeline latency 而非 input frame interval，需要在图注中明确：
  - “pipeline output latency is 0.211 ms”
  - 而不是暗示系统每 0.211 ms 产生一帧。
- 需要真实物理尺度：
  - mm/pixel 或接触区域直径；
  - scale bar 长度，例如 5 mm 或 10 mm。
- 若展示 indenting process，需要对应的 raw RGB frames 一起补充，或者仅展示 depth sequence。

## 当前脚本读取路径
- `materials/Sensor_Explosion.jpg`
- `materials/raw_rgb.png`
- `materials/my_depth_2d.png`
- optional: `materials/depth_sequence/depth_000.png` ... `depth_007.png`

## 当前输出文件
- `Fig1_nature_sensors_style.png`
- `Fig1_nature_sensors_style.pdf`
