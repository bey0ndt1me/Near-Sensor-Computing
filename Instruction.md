下面是按 **Nature Machine Intelligence** 风格重构后的完整作图规划。当前稿件用 **4 张主图 + Extended Data 若干项** 是合理的：主文聚焦“为什么需要近传感计算 → 架构如何实现 → 性能/精度是否可信 → 机器人任务带来什么能力”。NMI 要求所有图在正文中按 Fig. 1、Fig. 2 顺序引用，figure panel 最低 300 dpi、最大宽度 180 mm；Extended Data 最多 10 个 display items，且 Source Data 尤其是统计图建议单独提供。([Nature][1])

---

# 总体主图逻辑

建议主文保留 **4 张复合图**：

| 主图         | 核心问题                   | 作用                                          |
| ---------- | ---------------------- | ------------------------------------------- |
| **Fig. 1** | 系统是什么，为什么是 near-sensor | 建立概念、硬件、数学 pipeline                         |
| **Fig. 2** | 重建结果是否准确               | 证明 fixed-point FPGA 不牺牲深度质量                 |
| **Fig. 3** | 是否真的低延迟、低功耗、确定性        | 与 CPU/GPU/Jetson 比较，形成性能主张                  |
| **Fig. 4** | 低延迟带来了什么机器人能力          | reflex、slip、real-time depth streaming 的应用证明 |

我建议把你现在正文中 “Generalisation beyond tactile sensing” 移到 **Extended Data**，不要放进 Fig. 4 主图。主文 Fig. 4 已经承载 reflex + slip + manipulation，若再加入 Shack–Hartmann / structured-light 会分散主线。这个泛化性可以作为 Discussion 的支撑，而不是主文核心证据。

---

# Fig. 1 — Near-sensor visuotactile depth reconstruction

**目的：** 第一眼让编辑理解这不是普通 tactile sensor，而是“sensor + deterministic PDE solver”的物理近传感计算系统。
**建议版式：** 一张 180 mm 宽横向图，2 行布局。上排讲系统范式，下排讲硬件与数据流。

## Fig. 1a — Conventional pipeline vs proposed near-sensor pipeline

**内容：** 左右对照图。

左侧：
`Visuotactile sensor → RGB frames → CPU/GPU host → photometric stereo → Poisson solver → depth map → robot control`

右侧：
`Visuotactile sensor → FPGA near-sensor Poisson accelerator → depth map / GPIO reflex → robot control`

**视觉重点：**

* 左侧用长箭头、host computer、variable latency jitter 标注。
* 右侧用短路径、on-chip reconstruction、fixed latency 标注。
* 可以在箭头上标出：host-dependent / near-sensor / deterministic / low power。

**需要的数据：**

| 数据                               | 用途          |
| -------------------------------- | ----------- |
| conventional CPU/GPU latency 典型值 | 标在左侧路径      |
| FPGA latency，例如 0.2 ms 或 0.3 ms  | 标在右侧路径      |
| FPGA power，例如 4 W 或 5 W          | 标在右侧模块      |
| camera frame rate，例如 400 FPS     | 标在 sensor 旁 |

**注意：** 这里不适合放太多数字，保留 2–3 个 headline numbers 即可。

---

## Fig. 1b — Sensor and hardware photograph / cross-section

**内容：** 建议做成 “photo + schematic overlay”。

左半：实物照片，包括：

* visuotactile fingertip
* CMOS camera
* RGB illumination
* elastomer membrane
* FPGA board / Zynq SoC
* cable connection

右半：剖面结构示意：

`object → reflective elastomer / PDMS → RGB LEDs → camera lens → CMOS sensor`

**需要的数据/素材：**

| 数据/素材                              | 用途           |
| ---------------------------------- | ------------ |
| 高清实物照片，最好白底或黑底                     | Nature 风格系统图 |
| 传感器 CAD 或剖面尺寸                      | 绘制结构层级       |
| PDMS 厚度、照明位置、camera 型号             | 标注关键参数       |
| MIPI CSI-2 / FFC / power line 连接关系 | 标注信号流        |

**当前稿件需要统一：** Methods 中写的是 **OV5645**，Results 中写的是 **IMX219**。作图前必须确定最终 camera 型号，否则 Fig. 1b 与 Methods 会冲突。

---

## Fig. 1c — Streaming spectral Poisson solver pipeline

**内容：** 这是 Fig. 1 最重要的技术 panel。建议画成横向 pipeline：

`RGB pixel stream → calibration/LUT gradient estimation → divergence → row DST → transpose → column DST → spectral division → column IDST → transpose → row IDST → depth map`

在 pipeline 上方标注数学形式：

`∇²z = ∇·g`
`DST → element-wise division → IDST`

在 pipeline 下方标注硬件性质：

* no iterative loop
* no convergence criterion
* fixed cycle count
* one-pixel / streaming output after pipeline fill
* 24-bit fixed point

**需要的数据：**

| 数据                            | 用途         |
| ----------------------------- | ---------- |
| 每个 stage 的 latency cycles     | 标在模块下方     |
| 每个 stage 的 bit width          | 可标在小字中     |
| 主要资源占用：LUT、DSP、BRAM、URAM      | 可用小图标表示    |
| total cycles，比如 33,280 cycles | 总括标注       |
| clock frequency，比如 166 MHz    | 换算 latency |

**建议：** 主图中只放 simplified pipeline；详细 bit width 和资源表放 Supplementary Fig. S1 / Extended Data。

---

## Fig. 1d — Continuous real-time depth reconstruction sequence

**内容：** 连续帧深度图序列，展示传感器输出是 dense depth map，而不是 contact/no-contact。

建议放 5–7 帧：

`t0, t0+Δt, t0+2Δt, ...`

对象可以用 hemispherical indenter 或实际抓取物体。每帧下面标出时间戳。

**需要的数据：**

| 数据                                | 用途                     |
| --------------------------------- | ---------------------- |
| 同一次实验的 raw RGB frames             | 可作为小 inset             |
| FPGA output depth maps            | 主体图                    |
| 每帧 timestamp                      | 标注 temporal resolution |
| depth colorbar，单位 mm              | 保证量纲明确                 |
| frame rate / inter-frame interval | 说明时序密度                 |

**建议色图：** depth 用单调 perceptually uniform colormap；不要用 jet。红蓝 colormap 留给正负差分图。

---

# Fig. 2 — Reconstruction accuracy and fixed-point fidelity

**目的：** 证明 FPGA fixed-point pipeline 的输出与 floating-point software reference 几乎一致，并且误差主要来自 sensor calibration / boundary condition，而不是硬件计算。

**建议版式：** 2 × 2 或 2 × 3。Fig. 2 是证据图，必须有 source data 和统计量。

---

## Fig. 2a — Depth-map comparison: FPGA vs Python vs ground truth

**内容：** 每个 indenter 一行，每行三列：

`FPGA depth | Python float64 depth | analytical ground truth`

建议至少展示 2–3 种几何：

* hemisphere
* cylinder
* prism / wedge / flat punch

**需要的数据：**

| 数据                                      | 用途                         |
| --------------------------------------- | -------------------------- |
| raw RGB frame                           | 可作为小 inset 或 Extended Data |
| FPGA depth map `D_FPGA(x,y)`            | 主图                         |
| Python float64 depth map `D_SW(x,y)`    | 对照                         |
| analytical ground truth `D_GT(x,y)`     | 精度基准                       |
| indenter shape、radius、indentation depth | 生成 GT                      |
| calibration LUT / photometric model     | 保证 FPGA 与 Python 输入一致      |

**统计要求：**

* 每个 shape 至少 n ≥ 10 trials，最好 n ≥ 30。
* 图中不要只展示一帧，caption 中说明 “representative of n trials”。

---

## Fig. 2b — Pixel-wise error map and cross-section profile

**内容：** 对一个代表性样本画：

左：`FPGA - GT` error heatmap
右：中心线 profile：`FPGA / Python / GT`

**为什么需要：** RMSE bar 不能告诉读者误差在哪里。error heatmap 能支持你正文中 “residuals concentrate at contact boundary / Dirichlet boundary assumption” 的论证。

**需要的数据：**

| 数据                                  | 用途               |
| ----------------------------------- | ---------------- |
| `D_FPGA - D_GT` pixel-wise residual | heatmap          |
| `D_SW - D_GT` residual              | 可放 Extended Data |
| 中心线坐标和深度值                           | line profile     |
| contact mask / boundary mask        | 可标注误差集中区域        |

**视觉编码：**

* residual heatmap 用 diverging colormap，中心为 0。
* profile 图必须有单位：depth/mm、position/mm 或 pixels。

---

## Fig. 2c — RMSE / MAE / PSNR by geometry

**内容：** grouped bar + individual trial dots。

每个 shape 下有两组：

* FPGA vs GT
* Python vs GT

也可以再加一个：

* FPGA vs Python

**需要的数据：**

| 数据                              | 用途                     |
| ------------------------------- | ---------------------- |
| 每次 trial 的 RMSE                 | individual dots        |
| 每种 shape 的 mean ± s.d. 或 95% CI | bar / error bar        |
| PSNR                            | 可以做右轴或放 caption        |
| p-value / equivalence test      | 证明 FPGA 与 Python 无显著差异 |

**建议统计：**

* 不建议只做普通 t-test 说 “no significant difference”。更强的是做 equivalence test，证明 FPGA–Python 差异低于预设容忍阈值。
* 如果样本量较小，用 bootstrap 95% CI。

---

## Fig. 2d — Fixed-point bit-width sweep

**内容：** x-axis = word length，y-axis = RMSE 或 PSNR。

点位：

* 16-bit
* 20-bit，如果你能补测更好
* 24-bit
* 32-bit
* float64 reference

图中用竖线或高亮点标出 deployed 24-bit operating point。

**需要的数据：**

| 数据                                    | 用途                       |
| ------------------------------------- | ------------------------ |
| bit-accurate simulation output        | RMSE / PSNR              |
| word length sweep results             | 曲线                       |
| per-stage quantization config         | caption 或 Extended Data  |
| representative failure case at 16-bit | 可放 inset 或 Extended Data |

**建议：** 你现在已有 `RMSE = 8.8 × 10^-4` 和 `PSNR = 30.3 dB`，但 caption 里必须说明这个 RMSE 是相对于 **float64 software reference**，不是相对于物理 ground truth。

---

## 可选 Fig. 2e — Boundary-condition limitation

如果主文空间允许，建议加入一个小 panel 展示 edge contact 失真。
如果主图太拥挤，则放 Extended Data。

**内容：**

* center contact vs edge contact
* GT vs reconstruction
* edge underestimation area

**作用：** 主动呈现 limitation，增强可信度。

---

# Fig. 3 — Deterministic latency and power efficiency

**目的：** 这是投稿 NMI 最关键的性能图之一。要证明的不只是 “fast”，而是：

1. latency low
2. latency deterministic
3. energy efficient
4. Pareto-optimal relative to alternatives

---

## Fig. 3a — Per-frame latency distribution / CDF

**内容：** cumulative distribution 或 violin/box plot。

平台建议包括：

* FPGA pipeline
* Python SciPy DST on CPU
* C++ FFTW on CPU
* CUDA cuFFT on desktop GPU
* CUDA/cuFFTDx on Jetson Orin Nano
* maybe CPU host loop including Ethernet, if relevant

**推荐 CDF 横轴用 log scale：**

`0.1 ms → 100 ms`

FPGA 应该是一条几乎竖直线。

**需要的数据：**

| 数据                                        | 用途                  |
| ----------------------------------------- | ------------------- |
| FPGA ILA measured latency，n = 1000 frames | FPGA distribution   |
| Python timing logs，n = 1000               | baseline            |
| C++ timing logs，n = 1000                  | baseline            |
| CUDA event timing，n = 1000                | GPU baseline        |
| Jetson timing，n = 1000                    | embedded comparison |
| host transfer included/excluded 标记        | 保证公平性               |

**必须明确：** GPU timing 是否包含 host-device transfer。正文中已经区分 cuFFT 和 cuFFTDx，这一点图例要清楚。

---

## Fig. 3b — Pipeline latency breakdown

**内容：** stacked bar 或 waterfall chart。

分块：

* pixel streaming / LUT gradient
* divergence
* row DST
* transpose 1
* column DST + spectral division
* column IDST
* transpose 2
* row IDST

**需要的数据：**

| 数据                                     | 用途          |
| -------------------------------------- | ----------- |
| 每个 stage 的 clock cycles                | stacked bar |
| clock frequency                        | cycles → µs |
| total cycle count                      | 标注总 latency |
| overlapping / non-overlapping stage 说明 | 避免读者误解简单求和  |

**建议：** 如果 pipeline stage 有重叠，图中要清楚区分 “pipeline fill latency” 和 “per-frame throughput”。否则 reviewer 会质疑 cycle breakdown 与总 latency 不一致。

---

## Fig. 3c — Power and energy per frame

**内容：** 两个 y-axis 不推荐。建议分成上下两个小图或 grouped bars：

上：system power, W
下：energy per reconstructed frame, mJ/frame

平台：

* FPGA board
* Jetson Orin Nano
* CPU workstation
* desktop GPU

**需要的数据：**

| 数据                         | 用途                         |
| -------------------------- | -------------------------- |
| DC power meter measurement | 最可信 power 数据               |
| Vivado Power Estimator     | 可作为 FPGA internal estimate |
| platform idle power        | 可做 active-minus-idle       |
| throughput / frame rate    | 计算 energy/frame            |
| TDP                        | 不建议作为主数据，只可作为 reference    |

**重要建议：** 当前 Methods 中写 “CPU/GPU power values are rated TDP”，这在高水平审稿中会被认为不够严谨。主图最好使用实测 wall power 或 DC input power；TDP 可以放 Extended Data 或作为保守估计说明。

---

## Fig. 3d — Latency–power Pareto landscape

**内容：** scatter plot。

x-axis：latency，log scale
y-axis：power，log scale
点大小：depth map resolution 或 output type
点颜色：platform type

类别建议：

* this work: FPGA near-sensor dense depth
* CPU/GPU visuotactile reconstruction
* embedded GPU
* event-based tactile sensor
* capacitive/resistive near-sensor tactile systems
* prior FPGA PDE solvers，如果你主张 “PDE-level near-sensor computing”

**需要的数据：**

| 数据                                                         | 用途        |
| ---------------------------------------------------------- | --------- |
| 你自己的 latency / power                                       | 核心点       |
| baselines 的 latency / power                                | 对照点       |
| literature values with citations                           | landscape |
| output type：dense depth / contact / event / classification | 防止不公平比较   |
| resolution / frame rate                                    | 点大小或注释    |

**建议：** 不要把 event-based tactile sensor 直接当成同类 dense depth baseline。图例要标明它们输出的是 event/contact rather than dense geometry，否则 reviewer 会抓住比较不公平。

---

# Fig. 4 — Robotic functions enabled by sub-millisecond near-sensor depth

**目的：** 证明 Fig. 3 的低延迟不是工程炫技，而是带来 robot manipulation 的实际能力。
**建议版式：** 2 × 3 panel。主线：实时深度流 → reflex → slip。

---

## Fig. 4a — Robotic fingertip setup and real-time depth stream

**内容：** 左侧放实验 setup photo：sensor mounted on robot finger / gripper。右侧放一条 temporal strip：抓取过程中连续 depth maps。

**需要的数据：**

| 数据                                           | 用途                   |
| -------------------------------------------- | -------------------- |
| 机器人实验照片                                      | setup                |
| grasp sequence raw video                     | background           |
| synchronized FPGA depth maps                 | temporal strip       |
| object list：cylinder、foam ball、credit card 等 | caption              |
| dropped-frame count / frame timestamps       | 证明 no dropped frames |

**建议：** 你正文 249–252 行写了 real-time depth streaming，但当前 Fig. 4 caption 没有对应 panel。应把它加入 Fig. 4a，否则正文引用 `Fig. applications` 会悬空。

---

## Fig. 4b — High-speed protective reflex frames

**内容：** high-speed video frames，展示 contact onset 到 finger retraction。

时间标注示例：

`t = 0 ms: contact onset`
`t = 0.3 ms: GPIO asserted / detection`
`t = 3–5 ms: actuator response begins`
`t = 7–10 ms: visible retraction`

**需要的数据：**

| 数据                                  | 用途                                        |
| ----------------------------------- | ----------------------------------------- |
| high-speed camera video，最好 ≥300 fps | frame strip                               |
| oscilloscope trigger time           | 精确 contact onset                          |
| GPIO edge time                      | detection marker                          |
| actuator command / current signal   | 区分 detection latency 与 mechanical latency |
| frame timestamps                    | 标在每帧上                                     |

**关键区分：** 图中必须把 **on-chip detection latency** 和 **mechanical retraction latency** 分开。否则 “0.3 ms reflex” 容易被 reviewer 认为夸大，因为机械运动不可能 0.3 ms 完成。

---

## Fig. 4c — Reflex latency comparison

**内容：** violin plot / box plot / swarm plot。

组别：

* FPGA GPIO pathway
* CPU loop
* optionally Ethernet + host + actuator
* optionally biological reflex temporal reference as shaded band，不作为统计组

**需要的数据：**

| 数据                                           | 用途                       |
| -------------------------------------------- | ------------------------ |
| n = 50 trials FPGA contact-to-GPIO latency   | swarm dots               |
| n = 50 trials CPU contact-to-command latency | swarm dots               |
| median、IQR、99th percentile                   | 标注                       |
| Mann–Whitney U test 或 bootstrap CI           | 显示统计显著性                  |
| oscilloscope traces                          | 可放 inset 或 Extended Data |

**建议：** biological spinal reflex 不建议画成与 FPGA/CPU 同等的 “bar”。它不是同一实验系统，建议用灰色参考区间或 timeline label。

---

## Fig. 4d — Slip detection principle

**内容：** 简洁示意图：

object lateral displacement → contact patch shifts → depth difference shows bipolar pattern。

**需要的数据：**

| 数据                           | 用途                    |
| ---------------------------- | --------------------- |
| 不一定需要实验数据                    | 可用 schematic          |
| 定义 `ΔD(t) = D(t) - D(t-1)`   | panel 内公式             |
| contact mask `C`             | 解释 metric             |
| centroid displacement arrows | 解释 direction estimate |

---

## Fig. 4e — Depth-difference maps during slip

**内容：** 多帧 `ΔD` maps：

`before slip | slip onset | during slip | after slip`

用 red–blue diverging colormap，中心为 0。

**需要的数据：**

| 数据                                  | 用途            |
| ----------------------------------- | ------------- |
| depth maps `D_t` over time          | 计算差分          |
| frame timestamps                    | 标注 slip onset |
| object lateral velocity，例如 5 mm/s   | caption       |
| ground-truth motion stage / encoder | 验证 slip onset |
| contact mask                        | 只在接触区域积分      |

---

## Fig. 4f — Slip metric time series and detection threshold

**内容：** time-series plot：

* `M(t) = Σ |ΔD|`
* threshold
* ground-truth slip onset
* detected slip onset
* detection delay

可以再加一条 line 显示 direction angle 或 centroid displacement。

**需要的数据：**

| 数据                                 | 用途                  |
| ---------------------------------- | ------------------- |
| `M(t)` for each trial              | 主曲线                 |
| threshold value                    | detection criterion |
| ground-truth slip onset time       | vertical line       |
| detection time                     | vertical line       |
| noise floor during static contact  | threshold design    |
| multiple trials delay distribution | 可放 inset            |

**建议：** 主图给 representative trace + inset detection delay distribution；完整多速度、多物体 slip 结果放 Extended Data。

---

# Extended Data 建议规划

Nature Machine Intelligence 允许最多 10 个 Extended Data display items；这些图不应只是“补充好看”，而应支撑主图关键主张。([Nature][1])

---

## Extended Data Fig. 1 — Full FPGA resource utilization

**对应主文：** Fig. 1c / Fig. 3b
**内容：**

* LUT / FF / BRAM / URAM / DSP utilization bar chart
* XCZU7EV available vs used
* maybe per-module resource breakdown

**需要数据：**

* Vivado post-implementation utilization report
* per-module hierarchical utilization
* timing summary, WNS/TNS

---

## Extended Data Fig. 2 — Detailed timing and pipeline scheduling

**对应主文：** Fig. 3b
**内容：**

* Gantt-style pipeline schedule
* stage latency table visualized
* throughput vs latency distinction

**需要数据：**

* cycle-accurate simulation
* ILA markers
* valid/ready signal timing
* frame boundary timing

---

## Extended Data Fig. 3 — Resolution scaling

**对应主文：** Fig. 3 / Discussion
**内容：**

* 64×64, 128×128, 256×256 latency vs resolution
* URAM vs resolution
* resource utilization vs resolution
* extrapolation to 512×512 as dashed line

**需要数据：**

* synthesis results for each resolution
* measured or simulated cycle counts
* power estimates or measurements

---

## Extended Data Fig. 4 — Fixed-point bit allocation and sensitivity

**对应主文：** Fig. 2d
**内容：**

* bit allocation by stage
* dynamic range histogram
* per-stage 16-bit sensitivity
* quantization error propagation

**需要数据：**

* bit-accurate simulation logs
* max/min dynamic range per stage
* RMSE/PSNR per stage reduction

---

## Extended Data Fig. 5 — Calibration and lookup-table construction

**对应主文：** Fig. 1 / Fig. 2
**内容：**

* RGB intensity to gradient calibration curve
* LUT quantization
* photometric stereo calibration setup
* calibration residuals

**需要数据：**

* calibration images
* known surface normals / calibration ball data
* LUT entries
* calibration residual statistics

---

## Extended Data Fig. 6 — Baseline implementation fairness

**对应主文：** Fig. 3a
**内容：**

* CPU/GPU/Jetson hardware specs
* software stack versions
* timing protocol
* included/excluded transfer costs
* warm-up and idle-load details

**需要数据：**

* CPU model, GPU model, Jetson mode
* Python/SciPy/FFTW/CUDA/cuFFT/cuFFTDx versions
* compiler flags
* timing logs
* raw latency CSV

---

## Extended Data Fig. 7 — Object diversity in real-time depth streaming

**对应主文：** Fig. 4a
**内容：**

* depth sequences for rigid cylinder, soft foam, credit card, textured object, fragile object
* no dropped frames statistic

**需要数据：**

* synchronized RGB/depth recordings
* timestamps
* object labels and dimensions
* frame drop logs

---

## Extended Data Fig. 8 — Slip detection robustness

**对应主文：** Fig. 4e–f
**内容：**

* slip speed sweep：1, 2, 5, 10 mm/s
* object material sweep
* detection delay distribution
* false positive under static contact

**需要数据：**

* motion stage encoder or robot commanded displacement
* depth maps
* slip labels
* detection threshold
* confusion matrix / delay statistics

---

## Extended Data Fig. 9 — Beyond tactile: gradient-to-scalar generalization

**对应正文：** Discussion “generalises to Shack–Hartmann / structured-light / acoustic holography”
**内容：**

* synthetic mesh normals → reconstructed depth
* Shack–Hartmann wavefront slopes → reconstructed wavefront
* RMSE vs reference

**需要数据：**

* synthetic gradient fields
* known ground-truth scalar field
* Shack–Hartmann dataset
* FPGA or bit-accurate pipeline output
* error metrics

**建议：** 这正好承接你正文 274–282 行，但不建议占主图。

---

## Extended Data Fig. 10 — Oscilloscope traces and reflex measurement protocol

**对应主文：** Fig. 4b–c
**内容：**

* contact trigger channel
* GPIO output channel
* actuator command channel
* representative FPGA and CPU traces
* latency definition schematic

**需要数据：**

* oscilloscope waveform files
* channel calibration
* sampling rate
* trigger condition
* n = 50 trial latency table

---

# 每张主图的 Source Data 文件建议

NMI 鼓励统计图提供 Source Data，统计图建议每张图一个 Excel 或 CSV 文件，imaging source data 可放 repository。([Nature][1]) 建议你从一开始按下面方式整理：

| 文件名                    | 内容                                                                     |
| ---------------------- | ---------------------------------------------------------------------- |
| `SourceData_Fig1.xlsx` | Fig. 1 headline latency/power/frame-rate values、pipeline stage numbers |
| `SourceData_Fig2.xlsx` | 每个 trial 的 RMSE、MAE、PSNR、geometry、indentation depth、bit width          |
| `SourceData_Fig3.xlsx` | 每个平台每帧 latency、power、energy/frame、throughput                           |
| `SourceData_Fig4.xlsx` | reflex latency trials、slip metric time series、detection delays         |
| `ImagingData_Fig2/`    | representative RGB、FPGA depth、Python depth、GT、residual maps            |
| `ImagingData_Fig4/`    | high-speed frames、depth sequences、difference maps                      |

---

# 当前 LaTeX 中作图前必须修正的几个问题

1. **分辨率冲突：** Introduction 写 `256 × 256` 和 `0.3 ms`，Methods 多处写 `128 × 128` 和 `0.200 ms`。主图中只能保留一个 headline configuration；另一个作为 scaling result 放 Extended Data。

2. **camera 型号冲突：** Results 写 IMX219，Methods 写 OV5645。Fig. 1b 与 Methods 必须一致。

3. **功耗冲突：** Abstract 写约 4 W，宏定义 `\pwrFPGA` 是 5 W，正文又有 4 W / 5 W 混用。Fig. 3c 前必须确定最终实测值。

4. **Fig. 4 引用标签混乱：** 正文引用了 `fig:applications`、`fig:reflex`、`fig:slip`，但当前只有一个 Fig. 4，label 却是 `fig:general`。建议统一为 `\label{fig:applications}`，并把 reflex/slip 都作为 Fig. 4 的 panel。

5. **Fig. 4 caption 与正文不匹配：** 正文有 real-time depth streaming 和 generalisation beyond tactile，但 caption 只有 reflex/slip。建议：real-time depth streaming 加入 Fig. 4a；generalisation 移至 Extended Data Fig. 9。

6. **power baseline 的可信度：** 如果主图用 CPU/GPU TDP 而不是实测功耗，审稿风险较高。建议至少补充 wall-power 或 DC input power，并在 Methods 说明 active-minus-idle 的计算方式。

---

# 推荐最终主图标题

可以把 4 张主图标题改成下面这种更像 Nature 系列的叙事：

1. **Fig. 1 | Near-sensor spectral Poisson reconstruction for visuotactile depth.**
2. **Fig. 2 | Fixed-point hardware reconstruction preserves depth accuracy.**
3. **Fig. 3 | Deterministic sub-millisecond latency and energy-efficient operation.**
4. **Fig. 4 | Sub-millisecond tactile feedback enables reflexive and slip-aware manipulation.**

这套规划能让全文图像逻辑形成闭环：**Fig. 1 说明架构创新，Fig. 2 证明没有牺牲精度，Fig. 3 证明性能优势，Fig. 4 证明机器人意义。**

[1]: https://www.nature.com/natmachintell/submission-guidelines/aip-and-formatting "AIP and formatting | Nature Machine Intelligence"
