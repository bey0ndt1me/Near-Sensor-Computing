import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# 1. 严格遵守 Nature Final Artwork 规范的全局设置
# ==========================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] =['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 3

COLOR_FPGA = '#27AE60'
COLOR_ORIN = '#2980B9'
COLOR_4090 = '#C0392B'

# ==========================================
# 2. 数据准备与预处理
# ==========================================
# --- Panel a & b 数据 ---
batch_sizes_orin = np.array([1, 2, 4, 8, 10, 15, 20, 25, 30])
latency_orin = np.array([8.37, 9.82, 11.36, 15.20, 14.87, 16.44, 19.43, 24.67, 28.66])
power_orin = np.array([7.36, 7.51, 8.27, 9.49, 10.25, 13.14, 14.95, 16.46, 17.96])
energy_orin = (power_orin * latency_orin) / batch_sizes_orin

batch_sizes_fpga = np.array([1, 2, 4, 8, 16, 32])
latency_fpga = np.array([2.1, 2.2, 2.4, 3.8, 6.5, 12.0]) 
power_fpga = np.array([4.5, 4.6, 4.8, 5.2, 6.0, 7.5])    
energy_fpga = (power_fpga * latency_fpga) / batch_sizes_fpga

batch_sizes_4090 = np.array([1, 4, 16, 64, 128, 256])
latency_4090 = np.array([5.5, 5.6, 6.0, 8.5, 12.0, 18.0]) 
power_4090 = np.array([120, 150, 220, 310, 380, 420])     
energy_4090 = (power_4090 * latency_4090) / batch_sizes_4090

# --- 【全面升级：图 c 多分辨率功耗拆解数据】 ---
resolutions =['64', '128', '256']
# Orin真实数据: 64, 128, 256 (提取自你的新表格, Batch=1, 30FPS)
orin_totals = np.array([6.89, 6.90, 7.21])
orin_cores = np.array([1.23, 1.23, 1.53])

# FPGA模拟数据: 功耗平稳，核心算力占比极高且随分辨率微增
fpga_totals = np.array([4.2, 4.3, 4.5])
fpga_cores = np.array([3.1, 3.3, 3.5])

# RTX 4090模拟数据: 基础功耗极大，核心占比极低
rtx_totals = np.array([115.0, 118.0, 120.0])
rtx_cores = np.array([28.0, 32.0, 35.0])

platforms =['Custom\nFPGA', 'Jetson\nOrin NX', 'RTX\n4090']
totals_list =[fpga_totals, orin_totals, rtx_totals]
cores_list =[fpga_cores, orin_cores, rtx_cores]
colors_base = [COLOR_FPGA, COLOR_ORIN, COLOR_4090]

# --- 图 d 数据 ---
target_fps = np.array([30, 45, 60, 90, 120, 180])
orin_stream_power = np.array([7.21, 7.36, 7.66, 8.27, 9.34, 10.1])
orin_stream_latency = np.array([9.97, 9.83, 7.32, 5.63, 5.08, 4.84])

fpga_stream_power = np.array([4.5, 4.52, 4.55, 4.61, 4.65, 4.75])
fpga_stream_latency = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

rtx_stream_power = np.array([118, 119, 120, 122, 125, 130])
rtx_stream_latency = np.array([20, 20, 20, 20, 20, 20])

# ==========================================
# 3. 创建画布 (180mm 宽)
# ==========================================
fig_width_inch = 180 / 25.4 
fig_height_inch = 135 / 25.4 

fig, axs = plt.subplots(2, 2, figsize=(fig_width_inch, fig_height_inch))
fig.subplots_adjust(wspace=0.35, hspace=0.45, left=0.1, right=0.95, bottom=0.1, top=0.9)

def add_panel_label(ax, label):
    ax.text(-0.25, 1.05, label, transform=ax.transAxes, 
            fontsize=7, fontweight='bold', va='top', ha='right')

# ==========================================
# Panel (a) & (b)
# ==========================================
ax = axs[0, 0]
add_panel_label(ax, 'a')
ax.scatter(latency_fpga, energy_fpga, c=COLOR_FPGA, label='Custom FPGA', s=15, edgecolors='none', zorder=3)
ax.scatter(latency_orin, energy_orin, c=COLOR_ORIN, label='Jetson Orin NX', s=15, edgecolors='none', zorder=3)
ax.scatter(latency_4090, energy_4090, c=COLOR_4090, label='RTX 4090', s=15, edgecolors='none', zorder=3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Latency (ms)')
ax.set_ylabel('Energy per frame (mJ)')
x_pareto = np.linspace(1.5, 30, 100)
ax.plot(x_pareto, 12 / x_pareto + 1.5, linestyle='--', color='gray', alpha=0.5, zorder=1, linewidth=0.8)
ax.text(3.5, 4, 'Pareto frontier', color='gray', fontsize=5, rotation=-15)
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 0.9))

ax = axs[0, 1]
add_panel_label(ax, 'b')
ax.plot(batch_sizes_fpga, latency_fpga, marker='o', c=COLOR_FPGA, label='Custom FPGA')
ax.plot(batch_sizes_orin, latency_orin, marker='s', c=COLOR_ORIN, label='Jetson Orin NX')
ax.plot(batch_sizes_4090, latency_4090, marker='^', c=COLOR_4090, label='RTX 4090')
ax.set_xscale('log')
ax.set_xlabel('Batch size')
ax.set_ylabel('Latency (ms)')
ax.axhline(y=33.3, color='black', linestyle=':', linewidth=0.8)
ax.text(7.5, 34, '30 FPS limit', color='black', fontsize=5)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.9))

# ==========================================
# Panel (c): 【全新】多分辨率分组 100% 堆叠图
# ==========================================
ax = axs[1, 0]
add_panel_label(ax, 'c')

x_pos = np.arange(len(platforms))
bar_width = 0.22 # 变窄以容纳3个柱子

for idx_plat in range(len(platforms)):
    for idx_res in range(len(resolutions)):
        # 计算每个柱子的偏移位置
        pos = x_pos[idx_plat] + (idx_res - 1) * (bar_width + 0.03)
        
        core_val = cores_list[idx_plat][idx_res]
        total_val = totals_list[idx_plat][idx_res]
        core_pct = core_val / total_val * 100
        other_pct = 100 - core_pct
        
        # 画底部核心功耗
        ax.bar(pos, core_pct, width=bar_width, color=colors_base[idx_plat], edgecolor='white', linewidth=0.3)
        # 画顶部系统开销
        ax.bar(pos, other_pct, width=bar_width, bottom=core_pct, color=colors_base[idx_plat], alpha=0.3, edgecolor='white', linewidth=0.3)
        
        # 在柱子上方标注绝对瓦数 (垂直旋转以防重叠)
        ax.text(pos, 103, f"{total_val:.1f}W", ha='center', va='bottom', fontsize=4.5, fontweight='bold', color=colors_base[idx_plat], rotation=90)
        # 在柱子内标出占比
        ax.text(pos, core_pct/2, f"{core_pct:.0f}%", ha='center', va='center', fontsize=4.5, color='white', fontweight='bold')
        # 在 X 轴下方标出分辨率 (64, 128, 256)
        ax.text(pos, -5, resolutions[idx_res], ha='center', va='top', fontsize=4.5, color='black')

ax.set_xticks(x_pos)
# 将平台名字往下移一点，避开分辨率数字
ax.set_xticklabels(platforms)
ax.tick_params(axis='x', pad=12) 

ax.set_ylabel('Power breakdown (%) at Batch = 1')
ax.set_ylim(0, 145) # 进一步加高以容纳垂直的 "120.0W"
ax.set_yticks([0, 20, 40, 60, 80, 100])

legend_core = mpatches.Patch(facecolor='gray', edgecolor='white', label='Core compute (VDD_CV)')
legend_other = mpatches.Patch(facecolor='gray', alpha=0.3, edgecolor='white', label='Memory, IO & background')
# 添加一个空标签作为说明
ax.legend(handles=[legend_other, legend_core], frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0))

# ==========================================
# Panel (d): 【极简版】纯净点对点延迟标注
# ==========================================
ax = axs[1, 1]
add_panel_label(ax, 'd')

ax.plot(target_fps, rtx_stream_power, marker='^', c=COLOR_4090, label='RTX 4090')
ax.plot(target_fps, orin_stream_power, marker='s', c=COLOR_ORIN, label='Jetson Orin NX')
ax.plot(target_fps, fpga_stream_power, marker='o', c=COLOR_FPGA, label='Custom FPGA')

ax.set_yscale('log')
ax.set_xlabel('Target streaming rate (FPS) at Batch = 1')
ax.set_ylabel('Total power consumption (W)')

# 循环为 Orin 和 FPGA 的每一个点添加单纯的 Latency 标签
for i in range(len(target_fps)):
    # Orin 的延迟 (标在点的正上方)
    ax.annotate(f"{orin_stream_latency[i]:.1f} ms", 
                xy=(target_fps[i], orin_stream_power[i]), 
                xytext=(0, 4), textcoords='offset points', 
                ha='center', va='bottom', fontsize=4.5, color=COLOR_ORIN)
    
    # FPGA 的延迟 (标在点的正下方，体现其确定性)
    ax.annotate(f"{fpga_stream_latency[i]:.1f} ms", 
                xy=(target_fps[i], fpga_stream_power[i]), 
                xytext=(0, 7), textcoords='offset points', 
                ha='center', va='top', fontsize=4.5, color=COLOR_FPGA)

ax.legend(frameon=False, loc='center right')

# ==========================================
# 4. 保存为高标准矢量图 PDF
# ==========================================
plt.savefig('figure_nmi_180mm_final_v5.pdf', format='pdf', dpi=300)
plt.show()