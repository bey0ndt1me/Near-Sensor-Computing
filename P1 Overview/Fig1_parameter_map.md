# Fig1 手动调整 → 代码参数映射表

## 使用方式
1. 在 PowerPoint 中打开 `Fig1_Nature_Sensors_Interactive.pptx`，拖拽调整位置/大小
2. 或打开 `panel_svgs/panel_X.svg`，在 Illustrator 中调整元素级位置
3. 记录下你改动的数值（mm），查下表找到对应的代码参数
4. 修改 `Fig1_nature_sensors_artguide_v3.py` 中对应的数字
5. 运行 `python Fig1_nature_sensors_artguide_v3.py` 重新生成完整图
6. 运行 `python Fig1_export_pptx.py` 重新生成 PPTX

---

## 1. 整图尺寸

| 调整内容 | PPTX/Illustrator 度量 | 代码参数 | 代码位置 |
|---|---|---|---|
| 整图宽度 | 在 PPTX 中测量总宽度 (mm) | `FIG_W = 宽度mm/25.4` | 第80行 |
| 整图高度 | 在 PPTX 中测量总高度 (mm) | `FIG_H = 高度mm/25.4` | 第80行 |

---

## 2. 每个 Panel 在画布上的位置和大小

| Panel | 代码变量 | 当前位置 (画布比例) | 含义 |
|---|---|---|---|
| a | `ax_a` | `[0.040, 0.800, 0.650, 0.160]` | [left, bottom, width, height] |
| b | `ax_b` | `[0.040, 0.575, 0.205, 0.190]` |   四个数都是 0-1 |
| c | `ax_c` | `[0.265, 0.575, 0.425, 0.190]` |   left=0 是最左 |
| d | `ax_app` | `[0.730, 0.615, 0.230, 0.345]` |   bottom=0 是最下 |
| e | `ax_d` | `[0.060, 0.298, 0.220, 0.257]` |   width/height=1 是整张画布 |
| f | `ax_e` | `[0.335, 0.301, 0.625, 0.239]` | |

**如何换算 PPTX 位置 → 代码坐标：**
```
代码 left   = PPTX中panel左边缘距画布左边缘(mm) / 整图宽度(mm)
代码 bottom = PPTX中panel下边缘距画布下边缘(mm) / 整图高度(mm)
代码 width  = PPTX中panel宽度(mm) / 整图宽度(mm)
代码 height = PPTX中panel高度(mm) / 整图高度(mm)
```

> 代码位置：`make_figure()` 函数，第772-783行

---

## 3. Panel a 内部元素

### 3.1 五列节点的横向位置
| 节点 | 代码参数 | 当前值 | 调整说明 |
|---|---|---|---|
| 传感器 | `xs[0]` | `0.100` | 增大→右移 |
| raw RGB | `xs[1]` | `0.270` | 同上 |
| solver | `xs[2]` | `0.505` | 同上 |
| depth | `xs[3]` | `0.705` | 同上 |
| 指标 | `xs[4]` | `0.890` | 同上 |

> 代码位置：`draw_panel_a()` 第473行

### 3.2 两行纵向位置
| 行 | 代码参数 | 当前值 | 含义 |
|---|---|---|---|
| Host 行 | `yt` | `0.750` | 数据坐标 y 位置（越大越靠上） |
| Near-sensor 行 | `yb` | `0.200` | 同上 |

> 代码位置：`draw_panel_a()` 第475行

### 3.3 小图片物理尺寸
| 图片 | 代码参数 | 当前值 | 含义 |
|---|---|---|---|
| sensor photo | `img_size = axes_size_mm(ax, 10.0, 10.0)` | 10×10 mm | 改两个 10.0 统一缩放任一列图片 |

> 代码位置：`draw_panel_a()` 第478行

### 3.4 Solver 卡片文字
| 内容 | Host 行（上方） | Near-sensor 行（下方） |
|---|---|---|
| 标题 | `"Iterative multigrid"` | `"Spectral DST"` |
| 三阶段 | `("copy", "iter.", "depth")` | `("stream", "$1/\\lambda$", "depth")` |
| 脚注 | `"variable cycles"` | `"fixed 35,107 cycles"` |

> 代码位置：第488-490行（Host）、第507-509行（Near-sensor）

### 3.5 浅蓝底色底部延伸
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 蓝色底色向下延伸的距离 | `extend_down_5mm = axes_size_mm(ax, 1.0, 5.0)[1]` | 5 mm |

> 代码位置：第521行。改 `5.0` 为更大/更小值

### 3.6 指标卡片文字
| 行 | Heading | 三个指标行 |
|---|---|---|
| Host | `"variable delay"` | `("Latency", "unstable", "#8F2E2A")`, `("Power", "high startup", "#B43D35")`, `("Energy", "low efficiency", "#D15B4E")` |
| Near-sensor | `"spectral Poisson"` | `("Latency", "0.211 ms, zero jitter", "#137A43")`, `("Power", "0.305 W on chip", "#208F53")`, `("Energy", "0.031 mJ/frame", "#35A869")` |

> 代码位置：第494-497行（Host）、第514-518行（Near-sensor）

---

## 4. Panel b：传感器爆炸图

### 4.1 爆炸图图片位置和大小
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 图片在 panel 中的位置和大小 | `draw_image_original_ratio(ax, img, (0.020, 0.020, 0.504, 0.948))` | box=(left, bottom, width, height) |

> 代码位置：第540行

### 4.2 七个标签的纵向分布
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 标签纵向范围 | `label_ys = np.linspace(0.860, 0.180, 7)` | 第一个数=最上标签位置，第二个数=最下标签位置 |

> 代码位置：第549行

### 4.3 每个标签的文字和高度
| 标签 | 引出高度比例 | 标签文字 |
|---|---|---|
| 1 | `0.91` | `"PDMS elastomer"` |
| 2 | `0.80` | `"optical window"` |
| 3 | `0.68` | `"lens holder"` |
| 4 | `0.53` | `"RGB illumination"` |
| 5 | `0.39` | `"support frame"` |
| 6 | `0.25` | `"IMX219 CMOS"` |
| 7 | `0.10` | `"enclosure"` |

> 代码位置：第550-557行。`part_marks` 中每项第一个数控制引出虚线从爆炸图哪个高度发出

---

## 5. Panel c：Pipeline 展开图

### 5.1 CMOS 像素图尺寸
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 像素图物理尺寸 | `axes_size_mm(ax, 8.0, 8.0)` | 8×8 mm |

> 代码位置：第568行

### 5.2 六个流水线模块
| 模块 | 宽度 | 标题 | 细节 | 底色 | 边框色 |
|---|---|---|---|---|---|
| 1 | `0.080` | `"Photo\nstereo"` | `"RGB LUT\n$g_x,g_y$"` | `PALE_YELLOW` | `"#C9AE62"` |
| 2 | `0.072` | `"Div."` | `"$\\nabla\\cdot g$"` | `PALE_GREEN` | `"#8EB27B"` |
| 3 | `0.090` | `"DST"` | `"row/col"` | `PALE_PURPLE` | `"#A394BF"` |
| 4 | `0.090` | `"Spectral\nsolve"` | `"$\\hat f/\\lambda$"` | `PALE_RED` | `"#D1A19A"` |
| 5 | `0.090` | `"IDST"` | `"inverse"` | `PALE_PURPLE` | `"#A394BF"` |
| 6 | `0.076` | `"Output"` | `"sfix24"` | `PALE_YELLOW` | `"#C9AE62"` |

> 代码位置：第579-586行。`stage_specs` 列表中每项为 (宽度, 标题, 细节, 底色, 边框色)

### 5.3 模块高度和底部位置
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 模块底部高度 | `stage_y` | `0.350` |
| 模块高度 | `stage_h` | `0.430` |

> 代码位置：第596行

### 5.4 Pipeline 箭头长度
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 所有 stage 间箭头长度 | `pipe_arrow_len` | `0.028` |

> 代码位置：第599行

### 5.5 底部范围标注文字
| 标注 | 文字内容 |
|---|---|
| 标注1 | `"streaming, line-buffered"` |
| 标注2 | `"double-buffered transpose"` |
| 标注3 | `"streaming"` |
| 标注4 | `"fixed latency: 0.211 ms"` |

> 代码位置：第617-621行

---

## 6. Panel d：机械手应用图

### 6.1 两张图片的位置和大小
| 图片 | box 参数 | 当前值 |
|---|---|---|
| 上方（setup） | `(0.075, 0.570, 0.850, 0.340)` | (left, bottom, width, height) |
| 下方（grasp） | `(0.075, 0.085, 0.850, 0.340)` | 同上 |

> 代码位置：第634-639行

### 6.2 期望的图片素材文件名
| 用途 | 优先文件名 | 备用文件名 |
|---|---|---|
| setup | `robot_hand_setup.png` | `sensor_on_finger.png`, `dexterous_hand_setup.png` |
| grasp | `robot_hand_grasp.png` | `grasp_moment.png`, `feather_touch.png` |

> 代码位置：第633-638行。放入 `materials/` 目录后自动读取

---

## 7. Panel e：雷达图

### 7.1 五个维度名
| 维度 | 当前标签 | 代码位置 |
|---|---|---|
| 维度1 | `"Low\nlatency"` | |
| 维度2 | `"Power\nefficiency"` | |
| 维度3 | `"Throughput"` | |
| 维度4 | `"Board\nfootprint"` | |
| 维度5 | `"Timing\ndeterminism"` | |

> 代码位置：`labels = [...]` 第664-665行

### 7.2 四条线的数据和颜色
| 平台 | 五维分数 | 颜色 | 线宽 |
|---|---|---|---|
| This work | `[0.97, 0.92, 0.94, 0.96, 0.98]` | `RADAR_BLUE "#0F4D92"` | `1.15` |
| Jetson Orin NX | `[0.64, 0.52, 0.78, 0.42, 0.50]` | `RADAR_CYAN "#008A8A"` | `0.78` |
| GPU | `[0.74, 0.25, 0.95, 0.18, 0.43]` | `RADAR_ORANGE "#D97C2B"` | `0.78` |
| CPU | `[0.30, 0.34, 0.36, 0.62, 0.29]` | `RADAR_GRAY "#767676"` | `0.78` |

> 代码位置：第673-678行。`series` 列表中每项为 (图例名, [五个分数], 颜色, 线宽)

---

## 8. Panel f：按压时间序列

### 8.1 每帧深度图/按压图尺寸
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 每张深度图/按压图物理尺寸 | `axes_size_mm(ax, 12.0, 12.0)` | 12×12 mm |

> 代码位置：第735行

### 8.2 上方深度图行纵向位置
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 上排 y 位置 | `positions = [(0.040 + i*(w+gap), 0.520) for i in range(6)]` | `0.520` = y 坐标 |

> 代码位置：第738行。改 `0.520` 调整上下位置

### 8.3 六帧文字说明
| 帧 | 时间戳 | 阶段名 | 说明 |
|---|---|---|---|
| 0 | `"0.000"` | `"baseline"` | `"no load"` |
| 1 | `"0.133"` | `"first contact"` | `"touch onset"` |
| 2 | `"0.267"` | `"indent"` | `"local dent"` |
| 3 | `"0.400"` | `"spreading"` | `"contact grows"` |
| 4 | `"0.533"` | `"peak load"` | `"maximum indent"` |
| 5 | `"0.667"` | `"hold"` | `"steady hold"` |

> 代码位置：第728-731行

### 8.4 深度 colorbar
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| colorbar 位置 | `cbar_ax = ax.inset_axes([0.035, -0.085, 0.930, 0.022])` | [left, bottom, width, height] |
| 色阶范围 | `depth_norm = Normalize(vmin=0, vmax=2)` | 0-2 mm |
| 色阶标签 | `ticks=[0, 0.5, 1.0, 1.5, 2.0]` |  |

> 代码位置：第773-780行

---

## 9. 全局样式

### 9.1 字体和字号
| 调整内容 | 代码参数 | 当前值 |
|---|---|---|
| 全局基准字号 | `"font.size"` | `6.0` (pt) |
| Panel 标签字号 | `panel_label(fontsize=8.2)` | 8.2 pt |
| 一般文字字号 | 各处 `fontsize=5.0` | 5.0 pt |
| 图例字号 | `ax.legend(fontsize=5.0)` | 5.0 pt |
| 小字说明字号 | `fontsize=4.5 ~ 4.8` |  |

### 9.2 颜色常量
| 名称 | 色值 | 用途 |
|---|---|---|
| `BLUE` | `#0F4D92` | Hero method (this work) |
| `GREEN` | `#2E9E44` | Positive metrics |
| `RED` | `#E53935` | Negative metrics |
| `TEXT` | `#222222` | 正文文字 |
| `MUTED` | `#6F6F6F` | 次要文字 |
| `PALE_YELLOW` | `#FFF7D6` | Pipeline 模块底色 |
| `PALE_GREEN` | `#EDF7E8` | Pipeline 模块底色 |
| `PALE_PURPLE` | `#F1ECFA` | Pipeline 模块底色 |
| `PALE_RED` | `#FDEBEB` | Pipeline 模块底色 |

> 代码位置：第63-76行

---

## 快速参考：最常见的手动调整

| 我想... | 改哪里 |
|---|---|
| 把某个 panel 向左移动 2mm | `make_figure()` 中对应 `fig.add_axes([left, ...])` 的第一个数减小 `2/180` |
| 把某个 panel 放大 | `make_figure()` 中对应 `fig.add_axes` 的后两个数增大 |
| 让 panel a 中所有小图片变大 | `draw_panel_a()` 中 `axes_size_mm(ax, 10.0, 10.0)` → 改成 `12.0` |
| 调整箭头长度 | 对应 panel 的 `centered_gap_arrow(..., length=...)` |
| 改某段文字 | 搜索对应文字字符串，直接替换 |
| 改某个颜色 | 在第63-76行改对应常量，全局生效 |
| 移动 panel a 中某一列 | `draw_panel_a()` 中改 `xs` 数组对应值 |
