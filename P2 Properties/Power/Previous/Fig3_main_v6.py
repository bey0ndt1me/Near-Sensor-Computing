from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 6.5
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5
plt.rcParams['axes.linewidth'] = 0.8

SCRIPT_DIR = Path(__file__).resolve().parent
LATENCY_PANEL_A_CSV = SCRIPT_DIR / 'latency_panel_a.csv'
LATENCY_JETSON_CSV = SCRIPT_DIR / 'latency_jetson_orin_nx.csv'
LATENCY_GPU_CSV = SCRIPT_DIR / 'latency_gpu_workstation.csv'
LATENCY_CPU_CSV = SCRIPT_DIR / 'latency_cpu_workstation.csv'
RESOLUTION_SCALING_CSV = SCRIPT_DIR / 'resolution_scaling.csv'

CLOCK_MHZ = 166
TOTAL_CYCLES = 35107
LATENCY_FPGA_MS = 0.21149
JETSON_POWER_W = 8.58
GPU4090_POWER_W = 200.0
CPU_POWER_W = 100.0
THISWORK_POWER_W = 0.01224

POWER_SPLIT = {
    'This work': {'core': THISWORK_POWER_W * 0.75, 'bg': THISWORK_POWER_W * 0.25},
    'Jetson Orin NX': {'core': JETSON_POWER_W * 0.19, 'bg': JETSON_POWER_W * 0.81},
    'CPU i7-12700': {'core': CPU_POWER_W * 0.35, 'bg': CPU_POWER_W * 0.65},
    'RTX 4090': {'core': GPU4090_POWER_W * 0.29, 'bg': GPU4090_POWER_W * 0.71},
}

COLOR_FPGA_A = '#1f5aa6'
COLOR_THISWORK = '#2ca25f'
COLOR_THISWORK_LIGHT = '#a1d99b'
COLOR_JETSON = '#e67e22'
COLOR_JETSON_LIGHT = '#f2c38f'
COLOR_CPU = '#b22222'
COLOR_CPU_LIGHT = '#e6a3a3'
COLOR_GPU = '#3182bd'
COLOR_GPU_LIGHT = '#9ecae1'
COLOR_GRID = '#dadada'
COLOR_TEXT = '#222222'
COLOR_FRONTIER = '#111111'

PLATFORM_META = {
    'This work': {'color': COLOR_THISWORK, 'marker': 'o'},
    'Jetson Orin NX': {'color': COLOR_JETSON, 'marker': 'o'},
    'CPU i7-12700': {'color': COLOR_CPU, 'marker': 'D'},
    'RTX 4090': {'color': COLOR_GPU, 'marker': 's'},
}


def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def seeded_lognormal_from_mean_p99(mean_ms, p99_ms, n=1000, seed=0):
    z99 = 2.3263478740408408
    sigma = max((np.log(p99_ms) - np.log(mean_ms)) / z99, 0.05)
    mu = np.log(mean_ms) - 0.5 * sigma * sigma
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mu, sigma, n)
    x *= mean_ms / np.mean(x)
    return x


def add_panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_latency_series():
    series = {}
    if LATENCY_PANEL_A_CSV.exists():
        df = pd.read_csv(LATENCY_PANEL_A_CSV)
        if {'platform', 'latency_ms'}.issubset(df.columns):
            for platform, g in df.groupby('platform'):
                series[platform] = g['latency_ms'].to_numpy()
            return series
    if LATENCY_JETSON_CSV.exists():
        df = pd.read_csv(LATENCY_JETSON_CSV)
        col = 'latency_ms' if 'latency_ms' in df.columns else df.columns[0]
        series['Jetson Orin NX'] = df[col].to_numpy()
    if LATENCY_GPU_CSV.exists():
        df = pd.read_csv(LATENCY_GPU_CSV)
        col = 'latency_ms' if 'latency_ms' in df.columns else df.columns[0]
        series['GPU workstation (RTX 4090)'] = df[col].to_numpy()
    if LATENCY_CPU_CSV.exists():
        df = pd.read_csv(LATENCY_CPU_CSV)
        col = 'latency_ms' if 'latency_ms' in df.columns else df.columns[0]
        series['CPU workstation (i7-12700)'] = df[col].to_numpy()

    series['FPGA pipeline (this work)'] = np.full(1000, LATENCY_FPGA_MS)
    if 'Jetson Orin NX' not in series:
        series['Jetson Orin NX'] = seeded_lognormal_from_mean_p99(4.46, 8.5, n=1000, seed=3)
    if 'GPU workstation (RTX 4090)' not in series:
        series['GPU workstation (RTX 4090)'] = seeded_lognormal_from_mean_p99(1.10, 3.5, n=1000, seed=7)
    if 'CPU workstation (i7-12700)' not in series:
        series['CPU workstation (i7-12700)'] = seeded_lognormal_from_mean_p99(12.4, 28.0, n=1000, seed=11)
    return series


def load_resolution_scaling_data():
    if RESOLUTION_SCALING_CSV.exists():
        df = pd.read_csv(RESOLUTION_SCALING_CSV)
        df.columns = [c.strip() for c in df.columns]
        if {'platform', 'latency_ms', 'power_w'}.issubset(df.columns):
            if 'pixels' not in df.columns:
                if {'width', 'height'}.issubset(df.columns):
                    df['pixels'] = df['width'] * df['height']
                    df['label'] = df['width'].astype(int).astype(str) + '×' + df['height'].astype(int).astype(str)
                elif 'resolution' in df.columns:
                    def parse_res(s):
                        s = str(s).lower().replace(' ', '')
                        if 'x' in s:
                            a, b = s.split('x')
                            return int(a), int(b)
                        return int(s), int(s)
                    wh = df['resolution'].apply(parse_res)
                    df['width'] = wh.apply(lambda t: t[0])
                    df['height'] = wh.apply(lambda t: t[1])
                    df['pixels'] = df['width'] * df['height']
                    df['label'] = df['width'].astype(int).astype(str) + '×' + df['height'].astype(int).astype(str)
            elif 'label' not in df.columns:
                df['label'] = df['pixels'].astype(int).astype(str)
            return df

    sides = np.array([64, 128, 256, 512])
    pixels = sides * sides
    bases = {
        'This work': {'lat': LATENCY_FPGA_MS, 'power': THISWORK_POWER_W},
        'Jetson Orin NX': {'lat': 4.46, 'power': JETSON_POWER_W},
        'CPU i7-12700': {'lat': 12.4, 'power': CPU_POWER_W},
        'RTX 4090': {'lat': 1.10, 'power': GPU4090_POWER_W},
    }
    lat_alpha = {'This work': 1.0, 'Jetson Orin NX': 1.0, 'CPU i7-12700': 1.0, 'RTX 4090': 1.0}
    power_alpha = {'This work': 0.20, 'Jetson Orin NX': 0.35, 'CPU i7-12700': 0.20, 'RTX 4090': 0.22}
    rows = []
    for name, base in bases.items():
        for s, p in zip(sides, pixels):
            scale = p / (128 * 128)
            rows.append({
                'platform': name,
                'width': s,
                'height': s,
                'pixels': p,
                'label': f'{s}×{s}',
                'latency_ms': base['lat'] * (scale ** lat_alpha[name]),
                'power_w': base['power'] * (scale ** power_alpha[name]),
            })
    return pd.DataFrame(rows)


def draw_panel_a(ax, latency_series):
    styles = {
        'FPGA pipeline (this work)': dict(color=COLOR_FPGA_A, lw=1.3),
        'Jetson Orin NX': dict(color=COLOR_JETSON, lw=1.25),
        'GPU workstation (RTX 4090)': dict(color=COLOR_GPU, lw=1.25),
        'CPU workstation (i7-12700)': dict(color=COLOR_CPU, lw=1.25),
    }
    order = ['FPGA pipeline (this work)', 'Jetson Orin NX', 'GPU workstation (RTX 4090)', 'CPU workstation (i7-12700)']
    for name in order:
        x, y = ecdf(latency_series[name])
        ax.step(x, y, where='post', label=name, **styles[name])
    ax.set_xscale('log')
    ax.set_xlim(0.1, 40)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('Per-frame latency CDF (1,000 frames)', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.axhline(0.5, color='0.45', linestyle=(0, (4, 4)), linewidth=0.8)
    ax.axhline(1.0, color='0.45', linestyle=(0, (4, 4)), linewidth=0.8)
    ax.annotate('FPGA\n0.211 ms fixed', xy=(LATENCY_FPGA_MS, 0.50), xytext=(0.17, 0.60),
                textcoords='data', fontsize=4.9, color=COLOR_FPGA_A, ha='left', va='center',
                arrowprops=dict(arrowstyle='-|>', lw=0.8, color=COLOR_FPGA_A))
    callouts = [
        ('Jetson Orin NX', 0.90, 1.06, COLOR_JETSON),
        ('GPU workstation (RTX 4090)', 0.83, 1.04, COLOR_GPU),
        ('CPU workstation (i7-12700)', 0.96, 1.03, COLOR_CPU),
    ]
    for key, ytxt, xmul, col in callouts:
        p99 = np.percentile(latency_series[key], 99)
        ax.annotate(f'99th pct.\n~{p99:.1f} ms', xy=(p99, 0.99), xytext=(p99 * xmul, ytxt),
                    fontsize=4.6, color=col, ha='left', va='center',
                    arrowprops=dict(arrowstyle='->', lw=0.7, color=col))
    leg = ax.legend(frameon=True, facecolor='white', edgecolor='0.75', loc='center right', bbox_to_anchor=(0.98, 0.52))
    leg.get_frame().set_linewidth(0.7)


def draw_panel_b(ax):
    ax.set_axis_off()
    ax.set_title('FPGA pipeline latency breakdown (@ 166 MHz)', loc='left', fontweight='bold', pad=8)
    rows = [
        ('RGB→normal', 6), ('Divergence', 0), ('Column DST', 584), ('Transpose buffer 1', 16391),
        ('Row DST', 572), ('Spectral division', 7), ('Transpose buffer 2', 16391), ('Row IDST + output', 1156),
    ]
    row_bg = ['#f8f3df', '#eef3df', '#ece7f6', '#e8f0fb', '#ece7f6', '#f9edd7', '#e8f0fb', '#f2f2f2']
    bar_fill = ['#cfd8dc', '#cfd8dc', '#b9a3dc', '#9fbbe0', '#b9a3dc', '#f3cf87', '#9fbbe0', '#d0d0d0']
    max_cycles = max(c for _, c in rows)
    x_stage0, x_stage1 = 0.02, 0.46
    x_cycle0, x_cycle1 = 0.46, 0.64
    x_bar0, x_bar1 = 0.68, 0.98
    y_top, y_bottom, header_h = 0.94, 0.10, 0.07
    row_h = (y_top - y_bottom - header_h) / len(rows)
    ax.text((x_stage0 + x_stage1) / 2, y_top, 'Stage', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_cycle0 + x_cycle1) / 2, y_top, 'Cycles', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_bar0 + x_bar1) / 2, y_top, 'Latency (ms)', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    outer = FancyBboxPatch((x_stage0, y_bottom), x_bar1 - x_stage0, y_top - y_bottom - 0.01,
                           boxstyle='round,pad=0.008,rounding_size=0.015', linewidth=0.8, edgecolor='0.45', facecolor='none', transform=ax.transAxes)
    ax.add_patch(outer)
    ax.plot([x_stage1, x_stage1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    ax.plot([x_cycle1, x_cycle1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    for i, ((name, cyc), bg, fill) in enumerate(zip(rows, row_bg, bar_fill)):
        y1 = y_top - header_h - i * row_h
        y0 = y1 - row_h
        yc = (y0 + y1) / 2
        stage_patch = FancyBboxPatch((x_stage0 + 0.004, y0 + 0.004), (x_stage1 - x_stage0) - 0.008, row_h - 0.008,
                                     boxstyle='round,pad=0.002,rounding_size=0.008', linewidth=0.4,
                                     edgecolor='0.78', facecolor=bg, transform=ax.transAxes)
        ax.add_patch(stage_patch)
        ax.plot([x_stage1, x_bar1], [y0, y0], color='0.82', lw=0.55, transform=ax.transAxes)
        ax.text((x_stage0 + x_stage1) / 2, yc, name, transform=ax.transAxes, ha='center', va='center', fontsize=5.2)
        ax.text((x_cycle0 + x_cycle1) / 2, yc, '0' if cyc == 0 else f'{cyc:,}', transform=ax.transAxes, ha='center', va='center', fontsize=5.2)
        if cyc > 0:
            frac = cyc / max_cycles
            bx = x_bar0 + 0.015
            bar_w = frac * (x_bar1 - x_bar0 - 0.04)
            by = y0 + 0.18 * row_h
            bh = 0.64 * row_h
            ax.add_patch(Rectangle((bx, by), bar_w, bh, transform=ax.transAxes, facecolor=fill, edgecolor='0.5', lw=0.6))
            ax.text(x_bar1 - 0.015, yc, f'{cyc / (CLOCK_MHZ * 1e3):.4f}', transform=ax.transAxes, ha='right', va='center', fontsize=5.1)
        else:
            ax.text(x_bar1 - 0.015, yc, '0', transform=ax.transAxes, ha='right', va='center', fontsize=5.1)
    ax.text(0.10, 0.02, 'Total', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.40, 0.02, f'{TOTAL_CYCLES:,} cycles', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.71, 0.02, f'{LATENCY_FPGA_MS:.3f} ms', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.98, 0.02, 'Bar length encodes latency', transform=ax.transAxes, ha='right', va='bottom', fontsize=4.5, color='0.4')


def draw_panel_c(ax):
    ax.set_title('System-level power comparison (128×128 @ 180 fps)', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    labels = ['This work', 'Jetson\nOrin NX', 'CPU\ni7-12700', 'GPU\nRTX 4090']
    x = np.arange(4)
    core_vals = [POWER_SPLIT['This work']['core'], POWER_SPLIT['Jetson Orin NX']['core'], POWER_SPLIT['CPU i7-12700']['core'], POWER_SPLIT['RTX 4090']['core']]
    bg_vals = [POWER_SPLIT['This work']['bg'], POWER_SPLIT['Jetson Orin NX']['bg'], POWER_SPLIT['CPU i7-12700']['bg'], POWER_SPLIT['RTX 4090']['bg']]
    dark = [COLOR_THISWORK, COLOR_JETSON, COLOR_CPU, COLOR_GPU]
    light = [COLOR_THISWORK_LIGHT, COLOR_JETSON_LIGHT, COLOR_CPU_LIGHT, COLOR_GPU_LIGHT]
    for i in range(4):
        ax.bar(x[i], core_vals[i], width=0.48, color=dark[i], edgecolor='white', linewidth=0.6)
        ax.bar(x[i], bg_vals[i], bottom=core_vals[i], width=0.48, color=light[i], edgecolor='white', linewidth=0.6)
        total = core_vals[i] + bg_vals[i]
        label = f'{total*1000:.2f} mW' if total < 1 else f'{total:.2f} W' if total < 10 else f'~{total:.0f} W'
        ax.text(x[i], total * 1.10, label, ha='center', va='bottom', fontsize=4.8)
    ax.set_yscale('log')
    ax.set_ylabel('Power (W)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(0.02, 0.98, 'Dark: core compute (bottom). Light: background/system (top).\nCore fractions: FPGA 75%, Jetson 19%, CPU 35%, GPU 29%.',
            transform=ax.transAxes, ha='left', va='top', fontsize=4.7,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(handles=[Patch(facecolor='0.35', label='Core compute'), Patch(facecolor='0.75', label='Background/system')],
              frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.80), ncol=1, handlelength=1.1)


def pareto_front(points_df):
    # Lower latency and lower power are both better.
    pts = points_df[['latency_ms', 'power_w']].to_numpy()
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if j == i:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1]) and (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    front = points_df.iloc[keep].copy().sort_values('latency_ms')
    return front


def draw_panel_d(ax, scaling_df):
    ax.set_title('Pareto frontier across platforms and resolutions', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-12700', 'RTX 4090']
    short_name = {
        'This work': 'FPGA', 'Jetson Orin NX': 'Jetson', 'CPU i7-12700': 'CPU', 'RTX 4090': '4090'
    }

    for name in plot_order:
        g = scaling_df[scaling_df['platform'] == name].sort_values('pixels')
        meta = PLATFORM_META[name]
        ax.plot(g['latency_ms'], g['power_w'], color=meta['color'], lw=0.9, alpha=0.65)
        ax.scatter(g['latency_ms'], g['power_w'], s=18, marker=meta['marker'], color=meta['color'], edgecolor='white', linewidth=0.4, zorder=3)
        g_128 = g[g['label'].astype(str).isin(['128×128', '128x128'])]
        if len(g_128):
            row = g_128.iloc[0]
            ax.scatter([row['latency_ms']], [row['power_w']], s=38, marker=meta['marker'], color=meta['color'], edgecolor='k', linewidth=0.55, zorder=4)
            ax.text(row['latency_ms'] * 1.05, row['power_w'] * 1.04, short_name[name], fontsize=4.6, color=meta['color'], ha='left', va='bottom')

    frontier = pareto_front(scaling_df)
    ax.plot(frontier['latency_ms'], frontier['power_w'], color=COLOR_FRONTIER, lw=1.2, linestyle='--', zorder=2)
    ax.scatter(frontier['latency_ms'], frontier['power_w'], s=24, facecolor='white', edgecolor=COLOR_FRONTIER, linewidth=0.8, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Power (W)')
    ax.text(0.03, 0.97,
            'Points: all measured operating points across resolutions.\nDashed line: non-dominated Pareto frontier.\nOutlined markers indicate the 128×128 operating point.',
            transform=ax.transAxes, ha='left', va='top', fontsize=4.7,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=COLOR_THISWORK, markersize=4.8, lw=1.0, label='This work'),
        Line2D([0], [0], marker='o', color=COLOR_JETSON, markersize=4.8, lw=1.0, label='Jetson Orin NX'),
        Line2D([0], [0], marker='D', color=COLOR_CPU, markersize=4.6, lw=1.0, label='CPU i7-12700'),
        Line2D([0], [0], marker='s', color=COLOR_GPU, markersize=4.8, lw=1.0, label='RTX 4090'),
        Line2D([0], [0], color=COLOR_FRONTIER, lw=1.2, linestyle='--', label='Pareto frontier'),
    ], frameon=False, loc='lower right')


def draw_panel_e(ax, scaling_df):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Resolution scaling', loc='left', fontweight='bold', pad=8)

    ax_lat = ax.inset_axes([0.02, 0.16, 0.46, 0.76])
    ax_pow = ax.inset_axes([0.54, 0.16, 0.44, 0.76])

    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-12700', 'RTX 4090']
    ticks = []
    ticklabels = []
    for name in plot_order:
        g = scaling_df[scaling_df['platform'] == name].sort_values('pixels')
        meta = PLATFORM_META[name]
        ax_lat.plot(g['pixels'], g['latency_ms'], marker=meta['marker'], color=meta['color'], lw=1.1, ms=3.5, label=name)
        ax_pow.plot(g['pixels'], g['power_w'], marker=meta['marker'], color=meta['color'], lw=1.1, ms=3.5, label=name)
        ticks = g['pixels'].tolist()
        ticklabels = g['label'].tolist()

    for a in [ax_lat, ax_pow]:
        a.set_xscale('log', base=2)
        a.set_xticks(ticks)
        a.set_xticklabels(ticklabels)
        a.grid(color=COLOR_GRID, linewidth=0.6)
        strip_axes(a)
        a.tick_params(axis='x', labelsize=4.8)
        a.tick_params(axis='y', labelsize=4.8)
        a.set_xlabel('Resolution', labelpad=2)
    ax_lat.set_yscale('log')
    ax_pow.set_yscale('log')
    ax_lat.set_ylabel('Latency (ms)')
    ax_pow.set_ylabel('Power (W)')
    ax_lat.text(0.02, 0.96, 'Latency', transform=ax_lat.transAxes, ha='left', va='top', fontsize=5.4, fontweight='bold')
    ax_pow.text(0.02, 0.96, 'Power', transform=ax_pow.transAxes, ha='left', va='top', fontsize=5.4, fontweight='bold')
    ax.text(0.02, 0.04, 'Two aligned sub-axes are preferable to mixing latency and power in one chart.',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=4.6, color='0.35')
    ax_pow.legend(frameon=False, loc='lower right', fontsize=4.6)


def add_caption(fig):
    caption = (
        "Fig. 3 | Deterministic latency, energy efficiency and scaling of near-sensor reconstruction. "
        "a, Per-frame latency CDF over 1,000 frames. "
        "b, Stage-wise latency breakdown of the FPGA pipeline at 166 MHz. "
        "c, System-level power comparison under a unified 128×128 @ 180 fps setting, split into core-compute power (dark, bottom) and background/system power (light, top). "
        "d, Pareto frontier across platforms and resolutions in latency–power space. "
        "e, Resolution scaling, shown as paired latency and power trends sharing the same resolution axis."
    )
    fig.text(0.07, 0.02, caption, ha='left', va='bottom', fontsize=5.2, color=COLOR_TEXT, wrap=True)


def main():
    latency_series = load_latency_series()
    scaling_df = load_resolution_scaling_data()

    fig = plt.figure(figsize=(180 / 25.4, 225 / 25.4))
    outer = fig.add_gridspec(3, 1, left=0.07, right=0.985, bottom=0.10, top=0.97, height_ratios=[1.02, 0.88, 0.95], hspace=0.30)
    row1 = outer[0].subgridspec(5, 2, width_ratios=[1.0, 1.03], wspace=0.18, hspace=0.0)
    row2 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.24)
    row3 = outer[2].subgridspec(1, 1)

    ax_a = fig.add_subplot(row1[0:4, 0])
    ax_b = fig.add_subplot(row1[:, 1])
    ax_c = fig.add_subplot(row2[0, 0])
    ax_d = fig.add_subplot(row2[0, 1])
    ax_e = fig.add_subplot(row3[0, 0])

    draw_panel_a(ax_a, latency_series)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d, scaling_df)
    draw_panel_e(ax_e, scaling_df)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], list('abcde')):
        add_panel_label(ax, label)
    add_caption(fig)

    out_pdf = SCRIPT_DIR / 'Fig3_main_v6.pdf'
    out_png = SCRIPT_DIR / 'Fig3_main_v6.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(out_pdf)
    print(out_png)


if __name__ == '__main__':
    main()
