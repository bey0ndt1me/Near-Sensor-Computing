from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 6.5
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

CLOCK_MHZ = 166
TOTAL_CYCLES = 35107
LATENCY_FPGA_MS = 0.21149
LATENCY_FPGA_US = 211.49
JETSON_LAT_MS = 4.46
JETSON_POWER_W = 8.58
ASIC_POWER_W = 0.322
OPERATING_POINT_POWER_W = 0.01224
FPGA_IDLE_POWER_W = 4.0
FPGA_ACTIVE_POWER_W = 4.6
FPGA_BATCH_LATENCY_S = 0.00021149

COLOR_FPGA = '#1F4EAA'
COLOR_FPGA_GREEN = '#27AE60'
COLOR_ORIN = '#E67E22'
COLOR_GPU = '#2E8B3C'
COLOR_CPU = '#B22222'
COLOR_ORIN_LIGHT = '#85C1E9'
COLOR_GREY = '#BFC9CA'

PHASE_COLORS = {'H2D_Transfer':'#AED6F1','GPU_Compute':'#5DADE2','D2H_Transfer':'#AED6F1'}
PHASE_LABELS = {'H2D_Transfer':'H2D transfer','GPU_Compute':'GPU compute','D2H_Transfer':'D2H transfer'}
ACTIVE_PHASES = ['H2D_Transfer','GPU_Compute','D2H_Transfer']

PANEL_B_ROW_COLORS = ['#F9EAC6','#D9E7C3','#D8D0EB','#F3D2CC','#D8D0EB','#F9EAC6','#D6E4F2','#D6E4F2']
PANEL_B_BAR_COLORS = ['#D5DBDB','#D5DBDB','#B39DDB','#E9A7A1','#B39DDB','#F6D38A','#9EB7DC','#9EB7DC']
PANEL_B_BAR_EDGES = ['#95A5A6','#95A5A6','#7D5BA6','#D16C64','#7D5BA6','#D89B25','#6C8EBF','#6C8EBF']

SCRIPT_DIR = Path(__file__).resolve().parent
POWER_CSV_CANDIDATES = ['power_profile_sustained.csv','power_profile_sustained(2).csv']
LATENCY_CSV = SCRIPT_DIR / 'latency_panel_a.csv'
PARETO_CSV = SCRIPT_DIR / 'pareto_points.csv'


def maybe_load_csv(candidates):
    for name in candidates:
        p = SCRIPT_DIR / name
        if p.exists():
            return pd.read_csv(p), p
    return None, None


def integrate_energy(t, p):
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(p, t))


def add_panel_label(ax, label):
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x)+1) / len(x)
    return x, y


def seeded_lognormal_from_mean_p99(mean_ms, p99_ms, n=1000, seed=0):
    z99 = 2.3263478740408408
    sigma = max((np.log(p99_ms) - np.log(mean_ms)) / z99, 0.05)
    mu = np.log(mean_ms) - 0.5*sigma*sigma
    rng = np.random.default_rng(seed)
    x = rng.lognormal(mu, sigma, n)
    x *= mean_ms / np.mean(x)
    return x


def pretty_cycles(v):
    return '0 (combinational)' if v == 0 else f'{v:,}'


def cycle_to_ms(c):
    return c / (CLOCK_MHZ * 1e3)


def prepare_power_trace():
    df, path = maybe_load_csv(POWER_CSV_CANDIDATES)
    if df is None:
        raise FileNotFoundError(f'Could not find {POWER_CSV_CANDIDATES}')
    req = {'Time_s','Power_W','Phase'}
    if not req.issubset(df.columns):
        raise ValueError(f'Missing columns: {req - set(df.columns)}')
    df = df.sort_values('Time_s').reset_index(drop=True)
    active_mask = df['Phase'].isin(ACTIVE_PHASES)
    first_active_idx = int(active_mask[active_mask].index[0])
    last_active_idx = int(active_mask[active_mask].index[-1])
    t_active_start = float(df.loc[first_active_idx,'Time_s'])
    t_active_end = float(df.loc[last_active_idx,'Time_s'])
    active_duration_s = t_active_end - t_active_start

    idle_end_idx = first_active_idx - 1
    idle_start_idx = idle_end_idx
    while idle_start_idx >= 0 and df.loc[idle_start_idx,'Phase'] == 'Idle':
        idle_start_idx -= 1
    idle_start_idx += 1
    recent_idle_df = df.iloc[idle_start_idx:first_active_idx].copy()
    idle_baseline_w = float(recent_idle_df['Power_W'].median())
    df['Power_Dynamic_W'] = df['Power_W'] - idle_baseline_w

    padding_s = active_duration_s * 0.15
    plot_start_s = t_active_start - padding_s
    plot_end_s = t_active_end + padding_s
    plot_df = df[(df['Time_s'] >= plot_start_s) & (df['Time_s'] <= plot_end_s)].copy()
    plot_df['Time_ms'] = (plot_df['Time_s'] - plot_start_s) * 1000.0
    plot_df['Power_Dynamic_Smooth_W'] = plot_df['Power_Dynamic_W'].rolling(window=15, center=True, min_periods=1).mean()

    phase_stats = {}
    for phase in ACTIVE_PHASES:
        phase_df = df[df['Phase'] == phase].copy()
        start_s = float(phase_df['Time_s'].min())
        end_s = float(phase_df['Time_s'].max())
        duration_s = end_s - start_s
        pos_energy = integrate_energy(phase_df['Time_s'].to_numpy(), np.clip(phase_df['Power_Dynamic_W'].to_numpy(), a_min=0.0, a_max=None))
        phase_stats[phase] = {
            'start_s': start_s, 'end_s': end_s, 'duration_ms': duration_s*1000.0,
            'percent_time': duration_s/active_duration_s*100.0, 'dynamic_energy_pos_j': pos_energy
        }
    jetson_total_dynamic_energy_pos_j = sum(v['dynamic_energy_pos_j'] for v in phase_stats.values())
    fpga_total_dynamic_energy_j = max(FPGA_ACTIVE_POWER_W - FPGA_IDLE_POWER_W, 0.0) * FPGA_BATCH_LATENCY_S
    return {
        'plot_df': plot_df, 'phase_stats': phase_stats, 'idle_baseline_w': idle_baseline_w,
        'plot_start_s': plot_start_s, 'jetson_total_dynamic_energy_pos_j': jetson_total_dynamic_energy_pos_j,
        'fpga_total_dynamic_energy_j': fpga_total_dynamic_energy_j,
    }


def draw_panel_a(ax):
    if LATENCY_CSV.exists():
        lat_df = pd.read_csv(LATENCY_CSV)
        if not {'platform','latency_ms'}.issubset(lat_df.columns):
            raise ValueError('latency_panel_a.csv must contain platform, latency_ms')
        platform_series = {p: g['latency_ms'].to_numpy() for p, g in lat_df.groupby('platform')}
    else:
        platform_series = {
            'FPGA pipeline\n(this work)': np.full(1000, LATENCY_FPGA_MS),
            'Jetson Orin NX\n(8.58 W)': seeded_lognormal_from_mean_p99(JETSON_LAT_MS, 8.5, 1000, 1),
            'GPU workstation\nRTX 3060': seeded_lognormal_from_mean_p99(1.10, 4.2, 1000, 2),
            'CPU workstation\ni7-12700 (C++)': seeded_lognormal_from_mean_p99(12.4, 28.0, 1000, 3),
        }

    style_map = {
        'FPGA pipeline\n(this work)': dict(color=COLOR_FPGA, linewidth=1.2),
        'Jetson Orin NX\n(8.58 W)': dict(color=COLOR_ORIN, linewidth=1.2),
        'GPU workstation\nRTX 3060': dict(color=COLOR_GPU, linewidth=1.2),
        'CPU workstation\ni7-12700 (C++)': dict(color=COLOR_CPU, linewidth=1.2),
        'Proposed near-sensor pipeline': dict(color=COLOR_FPGA, linewidth=1.2),
        'Jetson Orin NX': dict(color=COLOR_ORIN, linewidth=1.2),
        'GPU workstation': dict(color=COLOR_GPU, linewidth=1.2),
        'CPU workstation': dict(color=COLOR_CPU, linewidth=1.2),
    }
    for platform, series in platform_series.items():
        x, y = ecdf(series)
        ax.step(x, y, where='post', label=platform, **style_map.get(platform, {'color':COLOR_GREY,'linewidth':1.0}))

    ax.set_xscale('log')
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative probability')
    ax.set_title('Per-frame latency CDF (1,000 frames)', loc='left', pad=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0.5, color='0.25', linestyle=(0,(5,4)), linewidth=0.8)
    ax.axhline(1.0, color='0.25', linestyle=(0,(5,4)), linewidth=0.8)

    ax.annotate('0.211 ms\n(fixed)', xy=(LATENCY_FPGA_MS,0.5), xytext=(0.06,0.62), textcoords='data', color=COLOR_FPGA, fontsize=5.2,
                arrowprops=dict(arrowstyle='-|>', color=COLOR_FPGA, lw=0.8), ha='center')

    name_sets = [
        ('Jetson Orin NX\n(8.58 W)', COLOR_ORIN, 1.08, 0.87),
        ('GPU workstation\nRTX 3060', COLOR_GPU, 1.06, 0.86),
        ('CPU workstation\ni7-12700 (C++)', COLOR_CPU, 1.04, 0.99),
    ]
    for name, col, xo, yo in name_sets:
        if name in platform_series:
            p99 = np.percentile(platform_series[name], 99)
            ax.annotate(f'99th pct.\n~{p99:.1f} ms', xy=(p99,0.99), xytext=(p99*xo, yo), textcoords='data', fontsize=5, color=col,
                        ha='left', va='top', arrowprops=dict(arrowstyle='->', lw=0.7, color=col))

    fpga_key = 'FPGA pipeline\n(this work)' if 'FPGA pipeline\n(this work)' in platform_series else ('Proposed near-sensor pipeline' if 'Proposed near-sensor pipeline' in platform_series else None)
    jetson_key = 'Jetson Orin NX\n(8.58 W)' if 'Jetson Orin NX\n(8.58 W)' in platform_series else ('Jetson Orin NX' if 'Jetson Orin NX' in platform_series else None)
    gpu_key = 'GPU workstation\nRTX 3060' if 'GPU workstation\nRTX 3060' in platform_series else ('GPU workstation' if 'GPU workstation' in platform_series else None)
    cpu_key = 'CPU workstation\ni7-12700 (C++)' if 'CPU workstation\ni7-12700 (C++)' in platform_series else ('CPU workstation' if 'CPU workstation' in platform_series else None)
    summary_lines = []
    if fpga_key is not None: summary_lines.append(('FPGA', COLOR_FPGA, f'{np.mean(platform_series[fpga_key]):.3f} ms (fixed)'))
    if jetson_key is not None: summary_lines.append(('Jetson', COLOR_ORIN, f'~{np.mean(platform_series[jetson_key]):.1f} ms'))
    if gpu_key is not None: summary_lines.append(('GPU', COLOR_GPU, f'~{np.mean(platform_series[gpu_key]):.1f} ms'))
    if cpu_key is not None: summary_lines.append(('CPU', COLOR_CPU, f'~{np.mean(platform_series[cpu_key]):.1f} ms'))

    ax.text(0.47, 0.18, 'Mean latency:', transform=ax.transAxes, fontsize=5.1, fontweight='bold')
    for i, (lbl, col, val) in enumerate(summary_lines):
        ax.text(0.47, 0.14 - 0.035*i, f'{lbl}: {val}', transform=ax.transAxes, fontsize=5.1, color=col)

    leg = ax.legend(frameon=True, facecolor='white', edgecolor='0.5', loc='center right', bbox_to_anchor=(1.00,0.52),
                    handlelength=1.7, handletextpad=0.6, borderpad=0.7)
    leg.get_frame().set_linewidth(0.7)

    if not LATENCY_CSV.exists():
        ax.text(0.01, -0.22,
                'Note: non-FPGA curves are stylized placeholders to match the reference style.\nReplace with real latency_panel_a.csv logs for the final manuscript.',
                transform=ax.transAxes, fontsize=4.3, color='0.35', va='top')


def draw_panel_b(ax):
    ax.set_axis_off()
    ax.set_title('FPGA pipeline latency breakdown (@ 166 MHz)', loc='left', pad=10, fontweight='bold')
    rows = [
        ('1. RGB-to-normal\n(photometric lookup)', 6),
        ('2. Divergence\n(finite difference)', 0),
        ('3. Column DST\n(extension + FFT + twiddle)', 584),
        ('4. Transpose buffer 1\n(frame reordering)', 16391),
        ('5. Row DST\n(transform)', 572),
        ('6. Spectral division\n(element-wise)', 7),
        ('7. Transpose buffer 2\n(frame reordering)', 16391),
        ('8. Row IDST / output', 1156),
    ]

    x_stage0, x_stage1 = 0.02, 0.45
    x_cycle0, x_cycle1 = 0.45, 0.66
    x_bar0, x_bar1 = 0.68, 0.98
    y_top, y_bottom, header_h = 0.93, 0.08, 0.06
    n = len(rows)
    row_h = (y_top - y_bottom - header_h) / n
    max_cycles = max(c for _, c in rows)

    ax.text((x_stage0+x_stage1)/2, y_top, 'Pipeline stage', ha='center', va='bottom', fontsize=6, fontweight='bold', transform=ax.transAxes)
    ax.text((x_cycle0+x_cycle1)/2, y_top, 'Cycles', ha='center', va='bottom', fontsize=6, fontweight='bold', transform=ax.transAxes)
    ax.text((x_bar0+x_bar1)/2, y_top, 'Latency (ms)', ha='center', va='bottom', fontsize=6, fontweight='bold', transform=ax.transAxes)
    outer = FancyBboxPatch((x_stage0,y_bottom), x_bar1-x_stage0, y_top-y_bottom-0.01, boxstyle='round,pad=0.008,rounding_size=0.015',
                           linewidth=0.8, edgecolor='0.35', facecolor='none', transform=ax.transAxes)
    ax.add_patch(outer)
    ax.plot([x_stage1,x_stage1],[y_bottom,y_top-0.01], color='0.55', lw=0.8, transform=ax.transAxes)
    ax.plot([x_cycle1,x_cycle1],[y_bottom,y_top-0.01], color='0.55', lw=0.8, transform=ax.transAxes)

    for i, ((name, cyc), bg, bcol, ecol) in enumerate(zip(rows, PANEL_B_ROW_COLORS, PANEL_B_BAR_COLORS, PANEL_B_BAR_EDGES)):
        y1 = y_top - header_h - i*row_h
        y0 = y1 - row_h
        yc = (y0+y1)/2
        stage_patch = FancyBboxPatch((x_stage0+0.002,y0+0.002), (x_stage1-x_stage0)-0.004, row_h-0.004,
                                     boxstyle='round,pad=0.002,rounding_size=0.01', linewidth=0.4, edgecolor='0.75', facecolor=bg,
                                     transform=ax.transAxes)
        ax.add_patch(stage_patch)
        ax.plot([x_stage1,x_bar1],[y0,y0], color='0.7', lw=0.6, transform=ax.transAxes)
        ax.text((x_stage0+x_stage1)/2, yc, name, ha='center', va='center', fontsize=5.1, transform=ax.transAxes)
        ax.text((x_cycle0+x_cycle1)/2, yc, pretty_cycles(cyc), ha='center', va='center', fontsize=5.2, transform=ax.transAxes)
        if cyc > 0:
            frac = cyc / max_cycles
            bar_w = frac * (x_bar1 - x_bar0 - 0.035)
            bx, by, bh = x_bar0 + 0.015, y0 + 0.18*row_h, 0.64*row_h
            rect = Rectangle((bx,by), bar_w, bh, transform=ax.transAxes, facecolor=bcol, edgecolor=ecol, lw=0.7, alpha=0.95)
            ax.add_patch(rect)
            ax.text(x_bar1 - 0.012, yc, f'{cycle_to_ms(cyc):.4f}', ha='right', va='center', fontsize=5.0, transform=ax.transAxes)
        else:
            ax.text(x_bar1 - 0.012, yc, '0', ha='right', va='center', fontsize=5.0, transform=ax.transAxes)

    ax.plot([x_stage0,x_bar1],[y_bottom,y_bottom], color='0.55', lw=0.8, transform=ax.transAxes)
    ax.text(0.12, 0.015, 'Total', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color=COLOR_FPGA, fontweight='bold')
    ax.text(0.42, 0.015, f'{TOTAL_CYCLES:,} cycles', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color=COLOR_FPGA, fontweight='bold')
    ax.text(0.70, 0.015, f'{LATENCY_FPGA_MS:.3f} ms', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, color=COLOR_FPGA, fontweight='bold')


def draw_panel_c(ax):
    labels = ['Operating point','ASIC estimate','Jetson Orin NX']
    values = [OPERATING_POINT_POWER_W, ASIC_POWER_W, JETSON_POWER_W]
    colors = [COLOR_FPGA_GREEN, '#A9DFBF', COLOR_ORIN]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor='white', linewidth=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Power (W)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Power comparison and accounting', loc='left', pad=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ann = ['12.24 mW\n@128×128, 180 FPS', '322 mW\nDesign Compiler\nfull-block estimate', '8.58 W\nmeasured']
    for b, txt in zip(bars, ann):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.18, txt, ha='center', va='bottom', fontsize=4.8)


def draw_panel_d(ax):
    points = [
        {'name':'This work','latency_ms':LATENCY_FPGA_MS,'power_w':OPERATING_POINT_POWER_W,'category':'proposed','annotation':'dense depth\n128×128 @180 FPS'},
        {'name':'Jetson Orin NX','latency_ms':JETSON_LAT_MS,'power_w':JETSON_POWER_W,'category':'embedded_gpu','annotation':'embedded GPU'},
    ]
    if PARETO_CSV.exists():
        p_df = pd.read_csv(PARETO_CSV)
        for _, row in p_df.iterrows():
            points.append({'name':row['name'],'latency_ms':row['latency_ms'],'power_w':row['power_w'],'category':row['category'],'annotation':row.get('annotation','')})
    cat_colors = {'proposed':COLOR_FPGA_GREEN,'embedded_gpu':COLOR_ORIN,'host_gpu':'#4CAF50','host_cpu':'#D84315'}
    cat_markers = {'proposed':'*','embedded_gpu':'o','host_gpu':'s','host_cpu':'D'}
    for p in points:
        cat = p['category']
        ax.scatter(p['latency_ms'], p['power_w'], s=70 if cat=='proposed' else 45, color=cat_colors.get(cat,COLOR_GREY),
                   marker=cat_markers.get(cat,'o'), edgecolor='k' if cat=='proposed' else 'white', linewidth=0.5, zorder=3)
        ax.text(p['latency_ms']*1.07, p['power_w']*1.08, f"{p['name']}\n{p['annotation']}" if p['annotation'] else p['name'], fontsize=4.7, va='bottom', ha='left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Power (W)')
    ax.set_title('Latency–power landscape', loc='left', pad=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    faster = JETSON_LAT_MS / LATENCY_FPGA_MS; lower = JETSON_POWER_W / OPERATING_POINT_POWER_W
    ax.text(0.03, 0.97, f'vs Jetson Orin NX: {faster:.1f}× faster\n{lower:.0f}× lower power', transform=ax.transAxes, fontsize=5, va='top',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85', alpha=0.9))
    ax.legend(handles=[
        Line2D([0],[0], marker='*', color='w', markerfacecolor=COLOR_FPGA_GREEN, markeredgecolor='k', markersize=8, label='Proposed dense depth'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_ORIN, markeredgecolor='white', markersize=6.5, label='Embedded GPU')
    ], frameon=False, loc='lower right')


def draw_panel_e(ax, power_data):
    plot_df = power_data['plot_df']; phase_stats = power_data['phase_stats']; plot_start_s = power_data['plot_start_s']; idle_baseline_w = power_data['idle_baseline_w']
    for phase in ACTIVE_PHASES:
        s = phase_stats[phase]
        x0 = (s['start_s'] - plot_start_s) * 1000.0; x1 = (s['end_s'] - plot_start_s) * 1000.0
        ax.axvspan(x0, x1, facecolor=PHASE_COLORS[phase], alpha=0.25, zorder=0)
    ax.plot(plot_df['Time_ms'], plot_df['Power_Dynamic_W'], color=COLOR_ORIN_LIGHT, linewidth=0.85, alpha=0.75, drawstyle='steps-post', label='Jetson Orin NX (raw samples)')
    ax.plot(plot_df['Time_ms'], plot_df['Power_Dynamic_Smooth_W'], color='#2C7FB8', linewidth=1.3, label='Jetson Orin NX (smoothed)')
    ax.axhline(0, color='0.4', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Execution time for a single batch (ms)'); ax.set_ylabel('Dynamic power above idle (W)')
    ax.set_title('Time-resolved dynamic power profile', loc='left', pad=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    y_min = min(plot_df['Power_Dynamic_W'].min(), -0.15); y_max = max(plot_df['Power_Dynamic_Smooth_W'].max(), plot_df['Power_Dynamic_W'].max()); y_span = max(y_max-y_min, 0.3)
    ax.set_ylim(y_min - 0.05*y_span, y_max + 0.27*y_span); text_y = ax.get_ylim()[1] - 0.08*(ax.get_ylim()[1]-ax.get_ylim()[0])
    for phase in ACTIVE_PHASES:
        s = phase_stats[phase]
        x0 = (s['start_s'] - plot_start_s) * 1000.0; x1 = (s['end_s'] - plot_start_s) * 1000.0; xc = 0.5*(x0+x1)
        ax.text(xc, text_y, f"{PHASE_LABELS[phase]}\n{s['duration_ms']:.1f} ms ({s['percent_time']:.1f}%)", ha='center', va='top', fontsize=4.8, color='#2C7FB8',
                bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.78))
    ax.text(0.015,0.985, f'Idle baseline = median power of the contiguous Idle segment\nimmediately preceding H2D transfer = {idle_baseline_w:.3f} W',
            transform=ax.transAxes, ha='left', va='top', fontsize=4.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.85', alpha=0.9))
    ax.text(0.985,0.06, 'Custom FPGA: single near-sensor streaming compute phase\nwith no H2D / D2H transfer overhead', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=4.7, color=COLOR_FPGA_GREEN, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLOR_FPGA_GREEN, alpha=0.9))
    ax.legend(frameon=False, loc='lower left')


def draw_panel_f(ax, power_data):
    phase_stats = power_data['phase_stats']; jetson_total = max(power_data['jetson_total_dynamic_energy_pos_j'], 1e-15); fpga_total = power_data['fpga_total_dynamic_energy_j']
    order = ['H2D_Transfer','GPU_Compute','D2H_Transfer']; fracs = [phase_stats[p]['dynamic_energy_pos_j']/jetson_total*100.0 for p in order]
    x_positions = np.array([0.0,1.08]); bottom = 0.0; bar_width = 0.48
    for phase, frac in zip(order, fracs):
        ax.bar(x_positions[0], frac, width=bar_width, bottom=bottom, color=PHASE_COLORS[phase], edgecolor='white', linewidth=0.6)
        if frac >= 7: ax.text(x_positions[0], bottom+frac/2, f"{PHASE_LABELS[phase]}\n{frac:.1f}%", ha='center', va='center', fontsize=4.7, color='white' if phase=='GPU_Compute' else 'black')
        bottom += frac
    ax.bar(x_positions[1], 100.0, width=bar_width, color=COLOR_FPGA_GREEN, edgecolor='white', linewidth=0.6)
    ax.text(x_positions[1],50.0,'FPGA compute\n100%',ha='center',va='center',fontsize=5,color='white')
    ax.set_ylim(0,100); ax.set_ylabel('Dynamic energy composition (%)'); ax.set_xticks(x_positions); ax.set_xticklabels(['Jetson Orin NX','Custom FPGA'])
    ax.set_title('Phase-wise dynamic-energy decomposition', loc='left', pad=10, fontweight='bold'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(x_positions[0], -11.5, f'Total = {jetson_total*1000.0:.2f} mJ', ha='center', va='top', fontsize=4.8)
    ax.text(x_positions[1], -11.5, f'Total = {fpga_total*1000.0:.4f} mJ', ha='center', va='top', fontsize=4.8, color=COLOR_FPGA_GREEN)
    ratio = jetson_total / fpga_total if fpga_total>0 else np.nan
    ax.text(0.5,1.04, f'Jetson / FPGA dynamic energy ≈ {ratio:.0f}×', transform=ax.transAxes, ha='center', va='bottom', fontsize=4.8,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85', alpha=0.9))
    ax.text(0.5,-0.27, 'Jetson decomposition uses positive-only dynamic energy above the recent-idle baseline\nto avoid cancellation by near-baseline fluctuations. FPGA total uses configurable placeholders.', transform=ax.transAxes, ha='center', va='top', fontsize=4.4, color='0.28')
    ax.legend(handles=[mpatches.Patch(color=PHASE_COLORS['H2D_Transfer'], label='H2D transfer'), mpatches.Patch(color=PHASE_COLORS['GPU_Compute'], label='GPU compute'), mpatches.Patch(color=PHASE_COLORS['D2H_Transfer'], label='D2H transfer'), mpatches.Patch(color=COLOR_FPGA_GREEN, label='FPGA compute')], frameon=False, loc='upper right')


def main():
    power_data = prepare_power_trace()
    fig = plt.figure(figsize=(180/25.4, 135/25.4))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, bottom=0.08, top=0.96, wspace=0.28, hspace=0.42)
    axes = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes
    draw_panel_a(ax_a); draw_panel_b(ax_b); draw_panel_c(ax_c); draw_panel_d(ax_d); draw_panel_e(ax_e, power_data); draw_panel_f(ax_f, power_data)
    for ax, label in zip(axes, list('abcdef')): add_panel_label(ax, label)
    fig.text(0.995, 0.005, 'If latency_panel_a.csv exists, panel-a uses real per-frame logs; otherwise Jetson / GPU / CPU curves are stylized placeholders.', ha='right', va='bottom', fontsize=4.2, color='0.4')
    out_pdf = SCRIPT_DIR / 'Fig3_complete_revised.pdf'; out_png = SCRIPT_DIR / 'Fig3_complete_revised.png'
    fig.savefig(out_pdf, dpi=300); fig.savefig(out_png, dpi=300)
    print(out_pdf); print(out_png)

if __name__ == '__main__':
    main()
