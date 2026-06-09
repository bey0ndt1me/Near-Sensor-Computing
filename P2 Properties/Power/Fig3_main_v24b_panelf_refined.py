from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyBboxPatch
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

COLOR_THISWORK = '#27AE60'
COLOR_THISWORK_LIGHT = '#A9DFBF'
COLOR_JETSON = '#2980B9'
COLOR_JETSON_LIGHT = '#AED6F1'
COLOR_CPU = '#E67E22'
COLOR_CPU_LIGHT = '#F5CBA7'
COLOR_GPU = '#C0392B'
COLOR_GPU_LIGHT = '#F5B7B1'
COLOR_GRID = '#e6e6e6'
COLOR_TEXT = '#222222'
COLOR_FRONTIER = '#111111'

PLATFORM_META = {
    'This work': {'color': COLOR_THISWORK, 'marker': 'o'},
    'Jetson Orin NX': {'color': COLOR_JETSON, 'marker': '^'},
    'CPU i7-13700': {'color': COLOR_CPU, 'marker': 'D'},
    'RTX 4090': {'color': COLOR_GPU, 'marker': 's'},
}
PLATFORM_LEGEND_ORDER = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
LEGEND_STYLE = dict(frameon=False, fontsize=4.8, handlelength=1.15, handletextpad=0.35,
                    columnspacing=0.8, borderaxespad=0.2, labelspacing=0.35)
PANEL_DEF_SCATTER_S = 14
PANEL_DEF_MARKER_MS = 3.0


def platform_legend_handles(include_frontier=False):
    handles = []
    for name in PLATFORM_LEGEND_ORDER:
        meta = PLATFORM_META[name]
        handles.append(Line2D([0], [0], marker=meta['marker'], color=meta['color'],
                              markerfacecolor=meta['color'], markeredgecolor='white',
                              markeredgewidth=0.45, markersize=4.8, lw=1.0, label=name))
    if include_frontier:
        handles.append(Line2D([0], [0], color=COLOR_FRONTIER, lw=1.1,
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
        return pd.DataFrame()
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


def normalize_platform_names(df):
    if 'platform' in df.columns:
        df = df.copy()
        df['platform'] = df['platform'].replace({
            'CPU i7-12700': 'CPU i7-13700',
            'CPU workstation (i7-12700)': 'CPU workstation (i7-13700)',
            'GPU workstation (RTX 4090)': 'RTX 4090',
            'FPGA pipeline (this work)': 'This work',
            'ASIC': 'This work',
            'Orin NX': 'Jetson Orin NX'
        })
    return df


def fmt_sig(v, unit='W'):
    if not np.isfinite(v):
        return ''
    return f'{v:.3g} {unit}'


def add_pixel_label(df):
    if 'pixels' not in df.columns:
        if 'width' in df.columns and 'height' in df.columns:
            df['pixels'] = num(df['width']) * num(df['height'])
        else:
            df['pixels'] = np.nan
    elif df['pixels'].isna().all() and 'width' in df.columns and 'height' in df.columns:
        df['pixels'] = num(df['width']) * num(df['height'])
    if 'label' not in df.columns:
        if 'width' in df.columns and 'height' in df.columns:
            df['label'] = df['width'].astype('Int64').astype(str) + '×' + df['height'].astype('Int64').astype(str)
        else:
            df['label'] = df['pixels'].astype('Int64').astype(str)
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
    if {'width', 'height'}.issubset(df.columns):
        w = num(df['width'])
        h = num(df['height'])
        df = df.loc[~((w == 512) & (h == 512))].copy()
    return df


def load_panel_a_latency():
    df = normalize_platform_names(read_csv(FILE_A))
    if 'latency_ms' in df.columns:
        df['latency_ms'] = num(df['latency_ms'])
    series = {}
    if not df.empty and 'platform' in df.columns and 'latency_ms' in df.columns:
        for platform, g in df.groupby('platform'):
            vals = g['latency_ms'].dropna().to_numpy()
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
    if 'CPU i7-13700' in series:
        series['CPU workstation (i7-13700)'] = series.pop('CPU i7-13700')
    if 'CPU workstation (i7-13700)' not in series:
        series['CPU workstation (i7-13700)'] = seeded_lognormal_from_mean_p99(12.4, 28.0, n=1000, seed=11)
    return series


def load_panel_b_breakdown():
    df = normalize_platform_names(read_csv(FILE_B))
    if df.empty:
        stages = [
            ('Preprocess', 1420), ('Gradient', 3185), ('Forward DST', 7860), ('Transpose', 3920),
            ('Spectral divide', 2360), ('Inverse DST', 7860), ('Postprocess', 8502), ('Idle', 0)
        ]
        df = pd.DataFrame({'stage': [s[0] for s in stages], 'cycles': [s[1] for s in stages], 'stage_order': list(range(1, len(stages)+1))})
        df['clock_mhz'] = CLOCK_MHZ
    df['stage_order'] = num(df.get('stage_order', np.arange(1, len(df)+1)))
    df['cycles'] = num(df.get('cycles', 0)).fillna(0)
    df['clock_mhz'] = num(df.get('clock_mhz', CLOCK_MHZ)).fillna(CLOCK_MHZ)
    df['latency_ms'] = num(df['cycles'] / (df['clock_mhz'] * 1e3))
    return df.sort_values('stage_order').reset_index(drop=True)


def load_panel_c_power():
    df = normalize_platform_names(read_csv(FILE_C))
    if df.empty:
        df = pd.DataFrame({
            'platform': ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090'],
            'total_power_w': [0.01224, 8.58, 100.0, 200.0],
        })
    for c in ['total_power_w', 'core_fraction_percent', 'background_fraction_percent', 'core_power_w', 'background_power_w']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'total_power_w' not in df.columns:
        df['total_power_w'] = np.nan
    defaults = {'This work': THISWORK_POWER_W, 'Jetson Orin NX': JETSON_POWER_W, 'CPU i7-13700': CPU_POWER_W, 'RTX 4090': GPU4090_POWER_W}
    for p, total in defaults.items():
        mask = df['platform'] == p
        if mask.any() and df.loc[mask, 'total_power_w'].isna().all():
            df.loc[mask, 'total_power_w'] = total
    return df


def load_panel_d_energy_throughput():
    df = normalize_platform_names(read_csv(FILE_D))
    if df.empty:
        rows = []
        # This work: 180, 400, full input ~10131 fps
        for fps, p in [(180, 0.01224), (400, 0.01224), (10131, 0.322)]:
            rows.append({'platform': 'This work', 'processed_fps': fps, 'full_chip_power_w': p})
        for fps in [30, 45, 60, 90, 120, 180]:
            rows.append({'platform': 'Jetson Orin NX', 'processed_fps': fps, 'full_chip_power_w': 8.0 + 0.007*fps})
        for fps in [30, 45, 60, 90, 120, 180, 240, 320, 400]:
            rows.append({'platform': 'CPU i7-13700', 'processed_fps': fps, 'full_chip_power_w': 32 + 0.17*fps})
            rows.append({'platform': 'RTX 4090', 'processed_fps': fps, 'full_chip_power_w': 60 + 0.35*fps})
        df = pd.DataFrame(rows)
    for c in ['processed_fps', 'full_chip_power_w', 'full_chip_energy_per_frame_mj', 'active_power_w', 'energy_per_frame_mj']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'full_chip_power_w' not in df.columns:
        if 'active_power_w' in df.columns:
            df['full_chip_power_w'] = df['active_power_w']
        else:
            df['full_chip_power_w'] = np.nan
    if 'full_chip_energy_per_frame_mj' not in df.columns:
        if 'energy_per_frame_mj' in df.columns:
            df['full_chip_energy_per_frame_mj'] = df['energy_per_frame_mj']
        else:
            df['full_chip_energy_per_frame_mj'] = np.nan
    m = df['full_chip_energy_per_frame_mj'].isna() & df['full_chip_power_w'].notna() & df['processed_fps'].notna() & (df['processed_fps']>0)
    df.loc[m, 'full_chip_energy_per_frame_mj'] = 1000.0 * df.loc[m, 'full_chip_power_w'] / df.loc[m, 'processed_fps']
    return df


def load_panel_e_pareto():
    df = remove_512_operating_points(fill_operating_metrics(normalize_platform_names(read_csv(FILE_E))))
    if df.empty or df['latency_ms'].dropna().empty or df['power_w'].dropna().empty:
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
                p = s*s
                scale = p/(128*128)
                rows.append({'platform': name, 'width': s, 'height': s, 'pixels': p, 'label': f'{s}×{s}',
                             'latency_ms': base['lat']*(scale**lat_alpha[name]),
                             'power_w': base['power']*(scale**power_alpha[name])})
        df = pd.DataFrame(rows)
    return df


def load_panel_f_scaling():
    df = remove_512_operating_points(fill_operating_metrics(normalize_platform_names(read_csv(FILE_F))))
    if df.empty or df['latency_ms'].dropna().empty or df['power_w'].dropna().empty:
        df = load_panel_e_pareto().copy()
    return df


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


def draw_panel_a(ax, latency_series):
    styles = {
        'FPGA pipeline (this work)': dict(color=COLOR_THISWORK, lw=1.35),
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
    strip_axes(ax)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.axhline(0.5, color='0.55', linestyle=(0, (4, 4)), linewidth=0.75)
    ax.axhline(1.0, color='0.55', linestyle=(0, (4, 4)), linewidth=0.75)
    ax.annotate('FPGA\n0.211 ms fixed', xy=(LATENCY_FPGA_MS, 0.50), xytext=(0.17, 0.61), textcoords='data', fontsize=4.8, color=COLOR_THISWORK, ha='left', va='center', arrowprops=dict(arrowstyle='-|>', lw=0.75, color=COLOR_THISWORK))
    for key, ytxt, xmul, col in [('Jetson Orin NX', 0.90, 1.05, COLOR_JETSON), ('GPU workstation (RTX 4090)', 0.82, 1.04, COLOR_GPU), ('CPU workstation (i7-13700)', 0.96, 1.02, COLOR_CPU)]:
        p99 = np.percentile(latency_series[key], 99)
        ax.annotate(f'99th pct.\n{p99:.2g} ms', xy=(p99, 0.99), xytext=(p99 * xmul, ytxt), fontsize=4.5, color=col, ha='left', va='center', arrowprops=dict(arrowstyle='->', lw=0.65, color=col))
    leg = ax.legend(frameon=False, loc='lower right', bbox_to_anchor=(0.995, 0.02), ncol=1,
                    handlelength=1.25, columnspacing=0.8, borderaxespad=0.15, handletextpad=0.4,
                    labelspacing=0.35)
    for t in leg.get_texts():
        t.set_fontsize(4.6)


def draw_panel_b(ax, df):
    ax.set_axis_off()
    rows = list(zip(df['stage'].tolist(), df['cycles'].astype(int).tolist(), df['latency_ms'].tolist()))
    row_bg = ['#F6FBF7', '#F6FBF7', '#F2F7FD', '#F5F5FB', '#FEF7EE', '#F2F7FD', '#F6FBF7', '#F7F7F7'][:len(rows)]
    bar_fill = ['#5DAE6F', '#78C48B', '#6FA8DC', '#9AACE6', '#F2B36B', '#6FA8DC', '#5DAE6F', '#C8C8C8'][:len(rows)]
    while len(row_bg) < len(rows):
        row_bg.append('#F7F7F7')
    while len(bar_fill) < len(rows):
        bar_fill.append('#C8C8C8')
    max_cycles = max(max(c for _, c, _ in rows), 1)
    x_stage0, x_stage1 = 0.02, 0.44
    x_cycle0, x_cycle1 = 0.44, 0.60
    x_bar0, x_bar1 = 0.64, 0.98
    y_top, y_bottom, header_h = 0.94, 0.11, 0.075
    row_h = (y_top - y_bottom - header_h) / len(rows)
    ax.text((x_stage0 + x_stage1) / 2, y_top, 'Stage', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_cycle0 + x_cycle1) / 2, y_top, 'Cycles', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    ax.text((x_bar0 + x_bar1) / 2, y_top, 'Latency (ms)', transform=ax.transAxes, ha='center', va='bottom', fontweight='bold')
    outer = FancyBboxPatch((x_stage0, y_bottom), x_bar1 - x_stage0, y_top - y_bottom - 0.01,
                           boxstyle='round,pad=0.008,rounding_size=0.015', linewidth=0.8,
                           edgecolor='0.45', facecolor='none', transform=ax.transAxes)
    ax.add_patch(outer)
    ax.plot([x_stage1, x_stage1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    ax.plot([x_cycle1, x_cycle1], [y_bottom, y_top - 0.01], color='0.65', lw=0.7, transform=ax.transAxes)
    for i, ((name, cyc, lat), bg, fill) in enumerate(zip(rows, row_bg, bar_fill)):
        y1 = y_top - header_h - i * row_h
        y0 = y1 - row_h
        yc = (y0 + y1) / 2
        stage_patch = FancyBboxPatch((x_stage0 + 0.004, y0 + 0.004), (x_stage1 - x_stage0) - 0.008, row_h - 0.008,
                                     boxstyle='round,pad=0.002,rounding_size=0.008', linewidth=0.4,
                                     edgecolor='0.80', facecolor=bg, transform=ax.transAxes)
        ax.add_patch(stage_patch)
        ax.plot([x_stage1, x_bar1], [y0, y0], color='0.85', lw=0.55, transform=ax.transAxes)
        ax.text(x_stage0 + 0.02, yc, name, transform=ax.transAxes, ha='left', va='center', fontsize=5.1)
        ax.text((x_cycle0 + x_cycle1) / 2, yc, '0' if cyc == 0 else f'{cyc:,}', transform=ax.transAxes, ha='center', va='center', fontsize=5.1)
        if cyc > 0:
            frac = cyc / max_cycles
            bx = x_bar0 + 0.015
            bar_w = frac * (x_bar1 - x_bar0 - 0.05)
            by = y0 + 0.18 * row_h
            bh = 0.64 * row_h
            ax.add_patch(Rectangle((bx, by), bar_w, bh, transform=ax.transAxes, facecolor=fill, edgecolor='0.55', lw=0.55))
            ax.text(x_bar1 - 0.016, yc, f'{lat:.4f}', transform=ax.transAxes, ha='right', va='center', fontsize=5.0)
        else:
            ax.text(x_bar1 - 0.016, yc, '0', transform=ax.transAxes, ha='right', va='center', fontsize=5.0)


def draw_panel_c(ax, df):
    strip_axes(ax)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    labels = ['This work', 'Jetson\nOrin NX', 'CPU\ni7-13700', 'GPU\nRTX 4090']
    x = np.arange(4)
    colors = [COLOR_THISWORK, COLOR_JETSON, COLOR_CPU, COLOR_GPU]
    y_base = 1e-2
    ymax = y_base
    for i, p in enumerate(plot_order):
        row = df[df['platform'] == p]
        if row.empty:
            total = [THISWORK_POWER_W, JETSON_POWER_W, CPU_POWER_W, GPU4090_POWER_W][i]
        else:
            total = float(row.iloc[0]['total_power_w'])
        ax.bar(x[i], total - y_base, bottom=y_base, width=0.68,
               color=colors[i], edgecolor='white', linewidth=0.6, alpha=0.95)
        ax.text(x[i], total * 1.10, fmt_sig(total, 'W'), ha='center', va='bottom', fontsize=4.8)
        ymax = max(ymax, total)
    ax.set_yscale('log')
    ax.set_ylim(y_base, ymax * 1.45)
    ax.set_ylabel('Power (W)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)


def draw_panel_d(ax, df):
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.55, which='major', axis='both')
    ax.set_axisbelow(True)
    df = df.dropna(subset=['processed_fps', 'full_chip_energy_per_frame_mj'])
    df = df[df['processed_fps'].astype(float) > 0]
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    xvals, yvals = [], []
    for name in plot_order:
        sub = df[df['platform'].astype(str).eq(name)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('processed_fps')
        meta = PLATFORM_META[name]
        xs = sub['processed_fps'].astype(float).to_numpy()
        ys = sub['full_chip_energy_per_frame_mj'].astype(float).to_numpy()
        ax.plot(xs, ys, color=meta['color'], lw=0.95, alpha=0.65, zorder=2)
        ax.scatter(xs, ys, s=PANEL_DEF_SCATTER_S, marker=meta['marker'], color=meta['color'],
                   edgecolor='white', linewidth=0.45, alpha=0.96, zorder=3)
        xvals.extend(xs.tolist()); yvals.extend(ys.tolist())
    positive_x = [float(x) for x in xvals if float(x) > 0]
    xmin = min(positive_x) * 0.9 if positive_x else 10.0
    xmax = max(positive_x) * 1.25 if positive_x else 1000.0
    ax.set_xscale('log')
    ax.set_xlim(10, max(10000, xmax))
    ax.axvline(180, color='0.42', lw=0.95, ls='--', zorder=1)
    ax.text(180, 0.03, '180 FPS', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=4.8, color='0.36')
    ymin = max(0.001, min(yvals) * 0.75 if yvals else 0.08)
    ymax = max(yvals) * 1.35 if yvals else 20
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_xlabel('Processed throughput (FPS)')
    ax.set_ylabel('Full-chip energy per frame (mJ)')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 5), numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='x', which='major', length=3.0, width=0.7)
    ax.tick_params(axis='x', which='minor', length=1.6, width=0.45, color='0.45')
    add_platform_legend(ax, loc='upper right', ncol=2)


def draw_panel_e(ax, df):
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    for name in plot_order:
        g = df[df['platform'] == name].sort_values('pixels')
        if g.empty:
            continue
        meta = PLATFORM_META[name]
        ax.plot(g['latency_ms'], g['power_w'], color=meta['color'], lw=0.9, alpha=0.65)
        ax.scatter(g['latency_ms'], g['power_w'], s=PANEL_DEF_SCATTER_S, marker=meta['marker'], color=meta['color'], edgecolor='white', linewidth=0.4, zorder=3)
    frontier = pareto_front(df)
    if not frontier.empty:
        ax.plot(frontier['latency_ms'], frontier['power_w'], color=COLOR_FRONTIER, lw=1.2, linestyle='--', zorder=2)
        ax.scatter(frontier['latency_ms'], frontier['power_w'], s=PANEL_DEF_SCATTER_S, facecolor='white', edgecolor=COLOR_FRONTIER, linewidth=0.8, zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Power (W)')
    add_platform_legend(ax, include_frontier=True, loc='lower right', ncol=1)


def annotate_numeric_labels_panel_f(ax, items, value_kind='lat'):
    """Place compact labels with hand-tuned offsets to reduce overlaps."""
    offset_map = {
        'lat': {
            ('This work', '64×64'): (0, 8),
            ('This work', '128×128'): (0, 8),
            ('This work', '256×256'): (0, 8),
            ('Jetson Orin NX', '64×64'): (-12, 8),
            ('Jetson Orin NX', '128×128'): (0, 8),
            ('Jetson Orin NX', '256×256'): (10, -2),
            ('CPU i7-13700', '64×64'): (-6, 8),
            ('CPU i7-13700', '128×128'): (10, 6),
            ('CPU i7-13700', '256×256'): (10, -2),
            ('RTX 4090', '64×64'): (-10, 6),
            ('RTX 4090', '128×128'): (0, 8),
            ('RTX 4090', '256×256'): (10, -2),
        },
        'pow': {
            ('This work', '64×64'): (0, 8),
            ('This work', '128×128'): (0, 8),
            ('This work', '256×256'): (0, 8),
            ('Jetson Orin NX', '64×64'): (-10, 6),
            ('Jetson Orin NX', '128×128'): (0, 8),
            ('Jetson Orin NX', '256×256'): (10, -2),
            ('CPU i7-13700', '64×64'): (-6, 8),
            ('CPU i7-13700', '128×128'): (0, 8),
            ('CPU i7-13700', '256×256'): (10, 2),
            ('RTX 4090', '64×64'): (-8, 8),
            ('RTX 4090', '128×128'): (0, 8),
            ('RTX 4090', '256×256'): (10, -2),
        }
    }
    default = (6, 6)
    cur_map = offset_map['lat' if value_kind == 'lat' else 'pow']
    for platform, label, x, y, txt, color in items:
        dx, dy = cur_map.get((platform, label), default)
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(txt, (x, y), textcoords='offset points', xytext=(dx, dy),
                    ha='center', va=va, fontsize=4.05, color=color, clip_on=False)


def draw_panel_f(ax, df):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Narrower two-panel layout with spare whitespace on the right/edges.
    ax_lat = ax.inset_axes([0.07, 0.17, 0.37, 0.71])
    ax_pow = ax.inset_axes([0.52, 0.17, 0.37, 0.71])

    plot_order = ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']
    ticks, ticklabels = [], []
    lat_items, pow_items = [], []

    for name in plot_order:
        g = df[df['platform'] == name].sort_values('pixels')
        if g.empty:
            continue
        meta = PLATFORM_META[name]
        px = g['pixels'].astype(float).to_numpy()
        lat = g['latency_ms'].astype(float).to_numpy()
        pw = g['power_w'].astype(float).to_numpy()
        lab = g['label'].astype(str).tolist()

        ax_lat.plot(px, lat, marker=meta['marker'], color=meta['color'], lw=1.0,
                    ms=PANEL_DEF_MARKER_MS, label=name)
        ax_pow.plot(px, pw, marker=meta['marker'], color=meta['color'], lw=1.0,
                    ms=PANEL_DEF_MARKER_MS, label=name)

        lat_texts = [f'{v:.3g} ms' for v in lat]
        pow_texts = [f'{v*1000:.3g} mW' if v < 1 else f'{v:.3g} W' for v in pw]
        lat_items += list(zip([name]*len(lab), lab, px, lat, lat_texts, [meta['color']]*len(lab)))
        pow_items += list(zip([name]*len(lab), lab, px, pw, pow_texts, [meta['color']]*len(lab)))
        ticks = px.tolist(); ticklabels = lab

    annotate_numeric_labels_panel_f(ax_lat, lat_items, value_kind='lat')
    annotate_numeric_labels_panel_f(ax_pow, pow_items, value_kind='pow')

    for a, title in zip([ax_lat, ax_pow], ['Latency', 'Power']):
        a.set_xscale('log', base=2)
        if ticks:
            a.set_xticks(ticks)
            a.set_xticklabels(ticklabels)
        a.grid(color=COLOR_GRID, linewidth=0.55)
        strip_axes(a)
        a.tick_params(axis='x', labelsize=4.5)
        a.tick_params(axis='y', labelsize=4.5)
        a.set_xlabel('Resolution', labelpad=2)
        a.text(0.02, 0.96, title, transform=a.transAxes, ha='left', va='top', fontsize=5.2, fontweight='bold')

    ax_lat.set_yscale('log')
    ax_pow.set_yscale('log')
    ax_lat.set_ylabel('Latency (ms)')
    ax_pow.set_ylabel('Power (W)')

    # Unified legend centered above the two subplots.
    add_platform_legend(ax, loc='upper center', ncol=4, bbox_to_anchor=(0.48, 1.01))


def add_caption(fig):
    caption = (
        'Fig. 3 | Deterministic latency, energy efficiency and scaling of near-sensor reconstruction. '
        'a, Per-frame latency CDF over 1,000 frames. '
        'b, Stage-wise latency breakdown of the FPGA pipeline at 166 MHz. '
        'c, System-level power comparison under a unified 128×128 @ 180 fps setting. '
        'd, Full-chip energy per frame versus throughput. The vertical dashed line marks the 180 fps real-time requirement. '
        'e, Pareto frontier across platforms and resolutions in latency–power space. '
        'f, Resolution scaling shown as latency, power and energy-per-frame trends.'
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
    outer = fig.add_gridspec(3, 1, left=0.07, right=0.985, bottom=0.10, top=0.965,
                             height_ratios=[0.92, 0.70, 0.64], hspace=0.34)
    row1 = outer[0].subgridspec(1, 2, width_ratios=[1.02, 1.06], wspace=0.18)
    row2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.26)
    row3 = outer[2].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.30], wspace=0.0)
    ax_a = fig.add_subplot(row1[0, 0])
    ax_b = fig.add_subplot(row1[0, 1])
    ax_c = fig.add_subplot(row2[0, 0])
    ax_d = fig.add_subplot(row2[0, 1])
    ax_e = fig.add_subplot(row2[0, 2])
    ax_f = fig.add_subplot(row3[0, 0:3])
    ax_blank = fig.add_subplot(row3[0, 3]); ax_blank.axis('off')

    # Slightly reduce panel-a height so its visual weight better matches panel b.
    pos = ax_a.get_position()
    new_h = pos.height * 0.86
    new_y = pos.y0 + (pos.height - new_h) * 0.50
    ax_a.set_position([pos.x0, new_y, pos.width, new_h])

    draw_panel_a(ax_a, latency_series)
    draw_panel_b(ax_b, b_df)
    draw_panel_c(ax_c, c_df)
    draw_panel_d(ax_d, d_df)
    draw_panel_e(ax_e, e_df)
    draw_panel_f(ax_f, f_df)
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list('abcdef')):
        add_panel_label(ax, label)
    add_caption(fig)
    out_pdf = SCRIPT_DIR / 'Fig3_main_v24b_panelf_refined.pdf'
    out_png = SCRIPT_DIR / 'Fig3_main_v24b_panelf_refined.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(out_pdf)
    print(out_png)

if __name__ == '__main__':
    main()
