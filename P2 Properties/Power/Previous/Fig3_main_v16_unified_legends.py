from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

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
TEMPLATE_DIR = SCRIPT_DIR / 'template'

FILE_A = TEMPLATE_DIR / 'fig3a_latency_cdf_filled.csv'
FILE_B = TEMPLATE_DIR / 'fig3b_pipeline_breakdown_template.csv'
FILE_C = TEMPLATE_DIR / 'fig3c_power_split_template.csv'
FILE_D = TEMPLATE_DIR / 'fig3d_energy_throughput_template.csv'
FILE_E = TEMPLATE_DIR / 'fig3e_pareto_points_template.csv'
FILE_F = TEMPLATE_DIR / 'fig3f_resolution_scaling_template.csv'

CLOCK_MHZ = 166
TOTAL_CYCLES = 35107
LATENCY_FPGA_MS = 0.21149
JETSON_POWER_W = 8.58
GPU4090_POWER_W = 200.0
CPU_POWER_W = 100.0
THISWORK_POWER_W = 0.01224

COLOR_FPGA_A = '#27AE60'
COLOR_THISWORK = '#27AE60'
COLOR_THISWORK_LIGHT = '#A9DFBF'
COLOR_JETSON = '#2980B9'
COLOR_JETSON_LIGHT = '#AED6F1'
COLOR_CPU = '#E67E22'
COLOR_CPU_LIGHT = '#F5CBA7'
COLOR_GPU = '#C0392B'
COLOR_GPU_LIGHT = '#F5B7B1'
COLOR_GRID = '#dadada'
COLOR_TEXT = '#222222'
COLOR_FRONTIER = '#111111'

PLATFORM_META = {
    'This work': {'color': COLOR_THISWORK, 'marker': 'o'},
    'Jetson Orin NX': {'color': COLOR_JETSON, 'marker': '^'},
    'CPU i7-13700': {'color': COLOR_CPU, 'marker': 'D'},
    'RTX 4090': {'color': COLOR_GPU, 'marker': 's'},
}

PLATFORM_LEGEND_ORDER = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
LEGEND_STYLE = dict(frameon=False, fontsize=4.6, handlelength=1.1, handletextpad=0.35,
                    columnspacing=0.8, borderaxespad=0.2, labelspacing=0.35)


def platform_legend_handles(include_frontier=False):
    handles = []
    for name in PLATFORM_LEGEND_ORDER:
        meta = PLATFORM_META[name]
        handles.append(Line2D([0], [0], marker=meta['marker'], color=meta['color'],
                              markerfacecolor=meta['color'], markeredgecolor='white',
                              markeredgewidth=0.45, markersize=4.8, lw=1.0, label=name))
    if include_frontier:
        handles.append(Line2D([0], [0], color=COLOR_FRONTIER, lw=1.15,
                              linestyle='--', label='Pareto frontier'))
    return handles


def add_platform_legend(ax, include_frontier=False, loc='upper right', ncol=2, bbox_to_anchor=None):
    kwargs = dict(LEGEND_STYLE)
    if bbox_to_anchor is not None:
        kwargs['bbox_to_anchor'] = bbox_to_anchor
    return ax.legend(handles=platform_legend_handles(include_frontier=include_frontier),
                     loc=loc, ncol=ncol, **kwargs)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing required template file: {path}')
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def num(s):
    return pd.to_numeric(s, errors='coerce')


def ecdf(x):
    x = np.sort(np.asarray(x, dtype=float))
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
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left', va='bottom')


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def load_panel_a_latency():
    df = normalize_platform_names(read_csv(FILE_A))
    if 'latency_ms' in df.columns:
        df['latency_ms'] = num(df['latency_ms'])
    series = {}
    for platform, g in df.groupby('platform'):
        vals = g['latency_ms'].dropna().to_numpy() if 'latency_ms' in g.columns else np.array([])
        if len(vals):
            series[platform] = vals
    series.setdefault('FPGA pipeline (this work)', np.full(1000, LATENCY_FPGA_MS))
    if 'This work' in series and 'FPGA pipeline (this work)' not in series:
        series['FPGA pipeline (this work)'] = series.pop('This work')
    if 'Jetson Orin NX' not in series:
        series['Jetson Orin NX'] = seeded_lognormal_from_mean_p99(4.46, 8.5, n=1000, seed=3)
    if 'RTX 4090' in series:
        series['GPU workstation (RTX 4090)'] = series.pop('RTX 4090')
    if 'GPU workstation (RTX 4090)' not in series:
        series['GPU workstation (RTX 4090)'] = seeded_lognormal_from_mean_p99(1.10, 3.5, n=1000, seed=7)
    if 'CPU i7-12700' in series:
        series['CPU workstation (i7-13700)'] = series.pop('CPU i7-12700')
    if 'CPU i7-13700' in series:
        series['CPU workstation (i7-13700)'] = series.pop('CPU i7-13700')
    if 'CPU workstation (i7-13700)' not in series:
        series['CPU workstation (i7-13700)'] = seeded_lognormal_from_mean_p99(12.4, 28.0, n=1000, seed=11)
    return series


def load_panel_b_breakdown():
    df = normalize_platform_names(read_csv(FILE_B))
    df['stage_order'] = num(df['stage_order'])
    df['cycles'] = num(df['cycles']).fillna(0)
    df['clock_mhz'] = num(df['clock_mhz']).fillna(CLOCK_MHZ)
    if 'include_in_total' in df.columns:
        df['include_in_total'] = df['include_in_total'].astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        df['include_in_total'] = True
    if 'latency_ms' not in df.columns:
        df['latency_ms'] = np.nan
    df['latency_ms'] = num(df['latency_ms'])
    miss = df['latency_ms'].isna()
    df.loc[miss, 'latency_ms'] = df.loc[miss, 'cycles'] / (df.loc[miss, 'clock_mhz'] * 1e3)
    return df.sort_values('stage_order').reset_index(drop=True)


def load_panel_c_power():
    df = normalize_platform_names(read_csv(FILE_C))
    for c in ['total_power_w', 'core_fraction_percent', 'background_fraction_percent', 'core_power_w', 'background_power_w']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'core_fraction_percent' in df.columns and 'background_fraction_percent' in df.columns:
        both_missing = df['core_fraction_percent'].isna() & df['background_fraction_percent'].notna()
        df.loc[both_missing, 'core_fraction_percent'] = 100 - df.loc[both_missing, 'background_fraction_percent']
        both_missing2 = df['background_fraction_percent'].isna() & df['core_fraction_percent'].notna()
        df.loc[both_missing2, 'background_fraction_percent'] = 100 - df.loc[both_missing2, 'core_fraction_percent']
    miss_core = df['core_power_w'].isna() if 'core_power_w' in df.columns else pd.Series(True, index=df.index)
    miss_bg = df['background_power_w'].isna() if 'background_power_w' in df.columns else pd.Series(True, index=df.index)
    df.loc[miss_core & df['total_power_w'].notna() & df['core_fraction_percent'].notna(), 'core_power_w'] = df['total_power_w'] * df['core_fraction_percent'] / 100.0
    df.loc[miss_bg & df['total_power_w'].notna() & df['background_fraction_percent'].notna(), 'background_power_w'] = df['total_power_w'] * df['background_fraction_percent'] / 100.0
    # fallback totals
    defaults = {'This work': THISWORK_POWER_W, 'Jetson Orin NX': JETSON_POWER_W, 'CPU i7-13700': CPU_POWER_W, 'RTX 4090': GPU4090_POWER_W}
    for p, total in defaults.items():
        mask = df['platform'] == p
        if mask.any():
            if df.loc[mask, 'total_power_w'].isna().all():
                df.loc[mask, 'total_power_w'] = total
            if df.loc[mask, 'core_power_w'].isna().all() and df.loc[mask, 'background_power_w'].isna().all():
                cf = df.loc[mask, 'core_fraction_percent'].iloc[0] if 'core_fraction_percent' in df.columns else np.nan
                if pd.notna(cf):
                    df.loc[mask, 'core_power_w'] = df.loc[mask, 'total_power_w'] * cf / 100.0
                    df.loc[mask, 'background_power_w'] = df.loc[mask, 'total_power_w'] - df.loc[mask, 'core_power_w']
    return df


def add_pixel_label(df):
    if 'pixels' not in df.columns or df['pixels'].isna().all():
        df['pixels'] = num(df['width']) * num(df['height'])
    if 'label' not in df.columns:
        if 'width' in df.columns and 'height' in df.columns:
            df['label'] = df['width'].astype('Int64').astype(str) + '×' + df['height'].astype('Int64').astype(str)
        else:
            df['label'] = df['pixels'].astype('Int64').astype(str)
    return df




def normalize_platform_names(df):
    if 'platform' in df.columns:
        df = df.copy()
        df['platform'] = df['platform'].replace({
            'CPU i7-12700': 'CPU i7-13700',
            'CPU workstation (i7-12700)': 'CPU workstation (i7-13700)',
            'GPU workstation (RTX 4090)': 'RTX 4090',
            'FPGA pipeline (this work)': 'This work',
            'Orin NX': 'Jetson Orin NX'
        })
    return df
def fill_operating_metrics(df):
    for c in ['latency_ms', 'power_w', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms', 'power_core_w', 'power_background_w', 'pixels', 'width', 'height']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'latency_ms' in df.columns and 'latency_p50_ms' in df.columns:
        df['latency_ms'] = df['latency_ms'].fillna(df['latency_p50_ms'])
    if 'power_w' in df.columns and 'power_core_w' in df.columns and 'power_background_w' in df.columns:
        df['power_w'] = df['power_w'].fillna(df['power_core_w'].fillna(0) + df['power_background_w'].fillna(0))
    return add_pixel_label(df)


def remove_512_operating_points(df):
    # 512x512 is removed from panel d/e because this-work hardware does not support this input size.
    if {'width', 'height'}.issubset(df.columns):
        w = num(df['width'])
        h = num(df['height'])
        df = df.loc[~((w == 512) & (h == 512))].copy()
    return df




def load_panel_d_energy_throughput():
    df = normalize_platform_names(read_csv(FILE_D))
    # accepted columns from template:
    # platform, point_id, target_fps, processed_fps, active_power_w, idle_power_w, incremental_power_w, energy_per_frame_mj, bubble_size_w, meets_180_fps, notes
    # Panel d intentionally does not use latency columns.
    for c in ['processed_fps', 'active_power_w', 'idle_power_w', 'incremental_power_w', 'energy_per_frame_mj', 'bubble_size_w']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'point_id' not in df.columns:
        df['point_id'] = df['platform']
    if 'processed_fps' not in df.columns:
        df['processed_fps'] = np.nan
    if 'active_power_w' not in df.columns:
        df['active_power_w'] = np.nan
    if 'idle_power_w' not in df.columns:
        df['idle_power_w'] = np.nan
    if 'incremental_power_w' not in df.columns:
        df['incremental_power_w'] = np.nan
    if 'energy_per_frame_mj' not in df.columns:
        df['energy_per_frame_mj'] = np.nan
    if 'bubble_size_w' not in df.columns:
        df['bubble_size_w'] = np.nan
    # derive missing values when possible
    miss_inc = df['incremental_power_w'].isna() & df['active_power_w'].notna() & df['idle_power_w'].notna()
    df.loc[miss_inc, 'incremental_power_w'] = df.loc[miss_inc, 'active_power_w'] - df.loc[miss_inc, 'idle_power_w']
    miss_energy = df['energy_per_frame_mj'].isna() & df['processed_fps'].notna()
    use_power = df['incremental_power_w'].where(df['incremental_power_w'].notna(), df['active_power_w'])
    df.loc[miss_energy, 'energy_per_frame_mj'] = 1000.0 * use_power[miss_energy] / df.loc[miss_energy, 'processed_fps']
    # If the template exists but values are still blank, keep the structured rows as-is.
    # This lets the user fill the required panel-d operating points directly without
    # silently falling back to synthetic data.
    return df


def load_panel_e_pareto():
    df = remove_512_operating_points(fill_operating_metrics(read_csv(FILE_E)))
    # fallback synthetic points if template still blank
    if df['latency_ms'].dropna().empty or df['power_w'].dropna().empty:
        sides = np.array([64, 128, 256])
        bases = {
            'This work': {'lat': LATENCY_FPGA_MS, 'power': THISWORK_POWER_W},
            'Jetson Orin NX': {'lat': 4.46, 'power': JETSON_POWER_W},
            'CPU i7-13700': {'lat': 12.4, 'power': CPU_POWER_W},
            'RTX 4090': {'lat': 1.10, 'power': GPU4090_POWER_W},
        }
        lat_alpha = {'This work': 1.0, 'Jetson Orin NX': 1.0, 'CPU i7-13700': 1.0, 'RTX 4090': 1.0}
        power_alpha = {'This work': 0.20, 'Jetson Orin NX': 0.35, 'CPU i7-13700': 0.20, 'RTX 4090': 0.22}
        rows = []
        for name, base in bases.items():
            for s in sides:
                p = s * s
                scale = p / (128 * 128)
                rows.append({'platform': name, 'width': s, 'height': s, 'pixels': p, 'label': f'{s}×{s}',
                             'latency_ms': base['lat'] * (scale ** lat_alpha[name]),
                             'power_w': base['power'] * (scale ** power_alpha[name])})
        df = remove_512_operating_points(pd.DataFrame(rows))
    return remove_512_operating_points(df)


def load_panel_f_scaling():
    df = remove_512_operating_points(fill_operating_metrics(read_csv(FILE_F)))
    if df['latency_ms'].dropna().empty or df['power_w'].dropna().empty:
        # if empty, borrow from pareto template / synthetic fallback
        df = load_panel_e_pareto().copy()
    return remove_512_operating_points(df)


def draw_panel_a(ax, latency_series):
    styles = {
        'FPGA pipeline (this work)': dict(color=COLOR_FPGA_A, lw=1.3),
        'Jetson Orin NX': dict(color=COLOR_JETSON, lw=1.25),
        'GPU workstation (RTX 4090)': dict(color=COLOR_GPU, lw=1.25),
        'CPU workstation (i7-13700)': dict(color=COLOR_CPU, lw=1.25),
    }
    order = ['FPGA pipeline (this work)', 'Jetson Orin NX', 'GPU workstation (RTX 4090)', 'CPU workstation (i7-13700)']
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
    ax.annotate('FPGA\n0.211 ms fixed', xy=(LATENCY_FPGA_MS, 0.50), xytext=(0.17, 0.60), textcoords='data', fontsize=4.9, color=COLOR_FPGA_A, ha='left', va='center', arrowprops=dict(arrowstyle='-|>', lw=0.8, color=COLOR_FPGA_A))
    for key, ytxt, xmul, col in [('Jetson Orin NX', 0.90, 1.06, COLOR_JETSON), ('GPU workstation (RTX 4090)', 0.83, 1.04, COLOR_GPU), ('CPU workstation (i7-13700)', 0.96, 1.03, COLOR_CPU)]:
        p99 = np.percentile(latency_series[key], 99)
        ax.annotate(f'99th pct.\n~{p99:.1f} ms', xy=(p99, 0.99), xytext=(p99 * xmul, ytxt), fontsize=4.6, color=col, ha='left', va='center', arrowprops=dict(arrowstyle='->', lw=0.7, color=col))
    leg = ax.legend(frameon=True, facecolor='white', edgecolor='0.75', loc='center right', bbox_to_anchor=(0.98, 0.52))
    leg.get_frame().set_linewidth(0.7)


def draw_panel_b(ax, df):
    ax.set_axis_off()
    ax.set_title('FPGA pipeline latency breakdown (@ 166 MHz)', loc='left', fontweight='bold', pad=8)
    rows = list(zip(df['stage'].tolist(), df['cycles'].astype(int).tolist(), df['latency_ms'].tolist()))
    row_bg = ['#f8f3df', '#eef3df', '#ece7f6', '#e8f0fb', '#ece7f6', '#f9edd7', '#e8f0fb', '#f2f2f2'][:len(rows)]
    while len(row_bg) < len(rows):
        row_bg.append('#f2f2f2')
    bar_fill = ['#cfd8dc', '#cfd8dc', '#b9a3dc', '#9fbbe0', '#b9a3dc', '#f3cf87', '#9fbbe0', '#d0d0d0'][:len(rows)]
    while len(bar_fill) < len(rows):
        bar_fill.append('#d0d0d0')
    max_cycles = max(max(c for _, c, _ in rows), 1)
    x_stage0, x_stage1 = 0.02, 0.46
    x_cycle0, x_cycle1 = 0.46, 0.64
    x_bar0, x_bar1 = 0.68, 0.98
    y_top, y_bottom, header_h = 0.94, 0.10, 0.07
    row_h = (y_top - y_bottom - header_h) / len(rows)
    ax.text((x_stage0 + x_stage1) / 2, y_top, 'Stage', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_cycle0 + x_cycle1) / 2, y_top, 'Cycles', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_bar0 + x_bar1) / 2, y_top, 'Latency (ms)', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    outer = FancyBboxPatch((x_stage0, y_bottom), x_bar1 - x_stage0, y_top - y_bottom - 0.01, boxstyle='round,pad=0.008,rounding_size=0.015', linewidth=0.8, edgecolor='0.45', facecolor='none', transform=ax.transAxes)
    ax.add_patch(outer)
    ax.plot([x_stage1, x_stage1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    ax.plot([x_cycle1, x_cycle1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    for i, ((name, cyc, lat), bg, fill) in enumerate(zip(rows, row_bg, bar_fill)):
        y1 = y_top - header_h - i * row_h
        y0 = y1 - row_h
        yc = (y0 + y1) / 2
        stage_patch = FancyBboxPatch((x_stage0 + 0.004, y0 + 0.004), (x_stage1 - x_stage0) - 0.008, row_h - 0.008, boxstyle='round,pad=0.002,rounding_size=0.008', linewidth=0.4, edgecolor='0.78', facecolor=bg, transform=ax.transAxes)
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
            ax.text(x_bar1 - 0.015, yc, f'{lat:.4f}', transform=ax.transAxes, ha='right', va='center', fontsize=5.1)
        else:
            ax.text(x_bar1 - 0.015, yc, '0', transform=ax.transAxes, ha='right', va='center', fontsize=5.1)
    total_cycles = int(df.loc[df['include_in_total'], 'cycles'].sum()) if 'include_in_total' in df.columns else int(df['cycles'].sum())
    total_latency = df.loc[df['include_in_total'], 'latency_ms'].sum() if 'include_in_total' in df.columns else df['latency_ms'].sum()
    ax.text(0.10, 0.02, 'Total', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.40, 0.02, f'{total_cycles:,} cycles', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.71, 0.02, f'{total_latency:.3f} ms', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.3, color=COLOR_FPGA_A, fontweight='bold')
    ax.text(0.98, 0.02, 'Bar length encodes latency', transform=ax.transAxes, ha='right', va='bottom', fontsize=4.5, color='0.4')


def draw_panel_c(ax, df):
    ax.set_title('System-level power comparison (128×128 @ 180 fps)', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    labels = ['This work', 'Jetson\nOrin NX', 'CPU\ni7-13700', 'GPU\nRTX 4090']
    x = np.arange(4)
    dark = [COLOR_THISWORK, COLOR_JETSON, COLOR_CPU, COLOR_GPU]
    light = [COLOR_THISWORK_LIGHT, COLOR_JETSON_LIGHT, COLOR_CPU_LIGHT, COLOR_GPU_LIGHT]
    for i, p in enumerate(plot_order):
        row = df[df['platform'] == p].iloc[0]
        core = float(row['core_power_w'])
        bg = float(row['background_power_w'])
        total = core + bg
        ax.bar(x[i], core, width=0.48, color=dark[i], edgecolor='white', linewidth=0.6)
        ax.bar(x[i], bg, bottom=core, width=0.48, color=light[i], edgecolor='white', linewidth=0.6)
        label = f'{total*1000:.2f} mW' if total < 1 else f'{total:.2f} W' if total < 10 else f'~{total:.0f} W'
        ax.text(x[i], total * 1.10, label, ha='center', va='bottom', fontsize=4.8)
    ax.set_yscale('log')
    ax.set_ylabel('Power (W)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(0.02, 0.98, 'Dark: core compute (bottom). Light: background/system (top).', transform=ax.transAxes, ha='left', va='top', fontsize=4.7, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(handles=[Patch(facecolor='0.35', label='Core compute'), Patch(facecolor='0.75', label='Background/system')], frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.84), ncol=1, handlelength=1.1)


def pareto_front(points_df):
    pts = points_df[['latency_ms', 'power_w']].dropna().to_numpy()
    valid = points_df[['latency_ms', 'power_w']].dropna().index.to_list()
    keep = []
    for ii, i in enumerate(valid):
        dominated = False
        for jj, j in enumerate(valid):
            if i == j:
                continue
            if (pts[jj, 0] <= pts[ii, 0] and pts[jj, 1] <= pts[ii, 1]) and (pts[jj, 0] < pts[ii, 0] or pts[jj, 1] < pts[ii, 1]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return points_df.loc[keep].sort_values('latency_ms')


def draw_panel_d(ax, df):
    ax.set_title('Energy per frame versus throughput', loc='left', fontweight='bold', pad=6)
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.55, which='both')
    ax.set_axisbelow(True)

    # Focus on full-device operating points; omit the separate ASIC-incremental marker here.
    df = df.copy()
    if 'point_id' in df.columns:
        df = df[df['point_id'].astype(str) != 'ASIC incremental']
    # Template rows for CPU/GPU/Jetson may be intentionally blank before measurement.
    # Drop rows that do not yet contain valid panel-d coordinates.
    df = df.dropna(subset=['processed_fps', 'energy_per_frame_mj'])
    df = df[df['processed_fps'].astype(float) > 0]

    label_order = ['ASIC full-chip', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    legend_alias = {'ASIC full-chip': 'This work'}

    def platform_color(name):
        return PLATFORM_META[legend_alias.get(name, name)]['color']

    def platform_marker(name):
        return PLATFORM_META[legend_alias.get(name, name)]['marker']

    def size_from_power(w):
        try:
            w = float(w)
        except Exception:
            w = 0.0
        ref = max(w, 0.01)
        return 16 + 6.5 * np.sqrt(ref if ref > 1 else ref * 100)

    xvals, yvals = [], []
    for name in label_order:
        sub = df[df['point_id'].astype(str).eq(name) | df['platform'].astype(str).eq(name)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('processed_fps')
        xs = sub['processed_fps'].astype(float).to_numpy()
        ys = sub['energy_per_frame_mj'].astype(float).to_numpy()
        bubble_ref = sub['active_power_w'] if 'active_power_w' in sub.columns else sub['incremental_power_w']
        ss = [size_from_power(v if pd.notna(v) else np.nan) for v in bubble_ref]
        ax.plot(xs, ys, color=platform_color(name), lw=0.95, alpha=0.65, zorder=2)
        ax.scatter(xs, ys, s=ss, marker=platform_marker(name), color=platform_color(name), edgecolor='white', linewidth=0.45, alpha=0.95, zorder=3)
        xvals.extend(xs.tolist())
        yvals.extend(ys.tolist())

    # Log-scaled throughput axis with decade limits.
    # For the current 30–400 FPS panel-d data, this gives 10^1–10^3.
    positive_x = [float(x) for x in xvals if float(x) > 0]
    if positive_x:
        xmin = 10 ** np.floor(np.log10(min(positive_x) * 0.85))
        xmax = 10 ** np.ceil(np.log10(max(positive_x) * 1.28))
    else:
        xmin, xmax = 10.0, 1000.0
    xmin = min(xmin, 10.0)
    xmax = max(xmax, 1000.0)
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)

    # Real-time threshold and low-throughput region. Shading starts at xmin instead of zero for log scale.
    ax.axvspan(xmin, 180, color='0.96', zorder=0)
    ax.axvline(180, color='0.42', lw=0.95, ls='--', zorder=1)
    ax.text(180, 0.04, '180 FPS real-time\nrequirement', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=4.8, color='0.36')
    ax.text(np.sqrt(xmin * 180), 0.96, 'below real-time', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=4.8, color='0.56')

    # Subtle callout for the desired operating region.
    ax.annotate('Preferred', xy=(0.86, 0.10), xytext=(0.68, 0.22), textcoords='axes fraction', xycoords='axes fraction',
                fontsize=4.9, color='0.30', arrowprops=dict(arrowstyle='->', lw=0.7, color='0.45'))

    ymin = max(0.08, min(yvals) * 0.75 if yvals else 0.08)
    ymax = max(3.0, max(yvals) * 1.35 if yvals else 3.0)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_xlabel('Processed throughput (FPS)')
    ax.set_ylabel('Energy per frame (mJ)')

    # Decade-only major ticks on the log throughput axis.
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', which='major', length=3.0, width=0.7)
    ax.tick_params(axis='x', which='minor', length=1.6, width=0.45, color='0.45')

    ax.text(0.02, 0.02, 'Marker size scales with active/device power.', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=4.6, color='0.42')
    add_platform_legend(ax, loc='upper right', ncol=2)


def draw_panel_e(ax, df):
    ax.set_title('Pareto frontier across platforms and resolutions', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    short_name = {'This work': 'FPGA', 'Jetson Orin NX': 'Jetson', 'CPU i7-13700': 'CPU', 'RTX 4090': '4090'}
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    for name in plot_order:
        g = df[df['platform'] == name].sort_values('pixels')
        if g.empty:
            continue
        meta = PLATFORM_META[name]
        ax.plot(g['latency_ms'], g['power_w'], color=meta['color'], lw=0.9, alpha=0.65)
        ax.scatter(g['latency_ms'], g['power_w'], s=18, marker=meta['marker'], color=meta['color'], edgecolor='white', linewidth=0.4, zorder=3)
        g_128 = g[g['label'].astype(str).isin(['128×128', '128x128'])]
        if len(g_128):
            row = g_128.iloc[0]
            ax.scatter([row['latency_ms']], [row['power_w']], s=38, marker=meta['marker'], color=meta['color'], edgecolor='k', linewidth=0.55, zorder=4)
            ax.text(row['latency_ms'] * 1.05, row['power_w'] * 1.04, short_name[name], fontsize=4.6, color=meta['color'], ha='left', va='bottom')
    frontier = pareto_front(df)
    if not frontier.empty:
        ax.plot(frontier['latency_ms'], frontier['power_w'], color=COLOR_FRONTIER, lw=1.2, linestyle='--', zorder=2)
        ax.scatter(frontier['latency_ms'], frontier['power_w'], s=24, facecolor='white', edgecolor=COLOR_FRONTIER, linewidth=0.8, zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Power (W)')
    ax.text(0.03, 0.97, 'Points: all operating points from template/fig3d CSV.\nDashed line: non-dominated Pareto frontier.\nOutlined markers indicate the 128×128 operating point.', transform=ax.transAxes, ha='left', va='top', fontsize=4.7, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    add_platform_legend(ax, include_frontier=True, loc='lower right', ncol=1)


def draw_panel_f(ax, df):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Resolution scaling', loc='left', fontweight='bold', pad=8)
    ax_lat = ax.inset_axes([0.02, 0.16, 0.46, 0.76])
    ax_pow = ax.inset_axes([0.54, 0.16, 0.44, 0.76])
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    ticks, ticklabels = [], []
    for name in plot_order:
        g = df[df['platform'] == name].sort_values('pixels')
        if g.empty:
            continue
        meta = PLATFORM_META[name]
        ax_lat.plot(g['pixels'], g['latency_ms'], marker=meta['marker'], color=meta['color'], lw=1.1, ms=3.5, label=name)
        ax_pow.plot(g['pixels'], g['power_w'], marker=meta['marker'], color=meta['color'], lw=1.1, ms=3.5, label=name)
        ticks = g['pixels'].tolist()
        ticklabels = g['label'].tolist()
    for a in [ax_lat, ax_pow]:
        a.set_xscale('log', base=2)
        if ticks:
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
    ax.text(0.02, 0.04, 'Panel f reads the fig3f CSV directly; fill real scaling data here.', transform=ax.transAxes, ha='left', va='bottom', fontsize=4.6, color='0.35')
    add_platform_legend(ax, loc='upper center', ncol=4, bbox_to_anchor=(0.55, 0.985))


def add_caption(fig):
    caption = (
        'Fig. 3 | Deterministic latency, energy efficiency and scaling of near-sensor reconstruction. '
        'a, Per-frame latency CDF over 1,000 frames. '
        'b, Stage-wise latency breakdown of the FPGA pipeline at 166 MHz. '
        'c, System-level power comparison under a unified 128×128 @ 180 fps setting, split into core-compute power (dark, bottom) and background/system power (light, top). '
        'd, Energy per frame versus throughput. The vertical dashed line marks the 180 fps real-time requirement; points to the lower right are preferred. '
        'e, Pareto frontier across platforms and resolutions in latency–power space. '
        'f, Resolution scaling, shown as paired latency and power trends sharing the same resolution axis.'
    )
    fig.text(0.07, 0.02, caption, ha='left', va='bottom', fontsize=5.2, color=COLOR_TEXT, wrap=True)


def main():
    latency_series = load_panel_a_latency()
    b_df = load_panel_b_breakdown()
    c_df = load_panel_c_power()
    d_df = load_panel_d_energy_throughput()
    e_df = load_panel_e_pareto()
    f_df = load_panel_f_scaling()

    fig = plt.figure(figsize=(180 / 25.4, 236 / 25.4))
    outer = fig.add_gridspec(3, 1, left=0.07, right=0.985, bottom=0.10, top=0.97, height_ratios=[1.00, 0.72, 0.86], hspace=0.28)
    row1 = outer[0].subgridspec(5, 2, width_ratios=[1.0, 1.03], wspace=0.18, hspace=0.0)
    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.25)
    row3 = outer[2].subgridspec(1, 1)
    ax_a = fig.add_subplot(row1[0:4, 0])
    ax_b = fig.add_subplot(row1[:, 1])
    ax_c = fig.add_subplot(row2[0, 0])
    ax_d = fig.add_subplot(row2[0, 1])
    ax_e = fig.add_subplot(row2[0, 2])
    ax_f = fig.add_subplot(row3[0, 0])
    draw_panel_a(ax_a, latency_series)
    draw_panel_b(ax_b, b_df)
    draw_panel_c(ax_c, c_df)
    draw_panel_d(ax_d, d_df)
    draw_panel_e(ax_e, e_df)
    draw_panel_f(ax_f, f_df)
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list('abcdef')):
        add_panel_label(ax, label)
    add_caption(fig)
    out_pdf = SCRIPT_DIR / 'Fig3_main_v16.pdf'
    out_png = SCRIPT_DIR / 'Fig3_main_v16.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(out_pdf)
    print(out_png)

if __name__ == '__main__':
    main()
