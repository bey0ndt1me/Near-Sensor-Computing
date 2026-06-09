from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
FILE_EXT_B = TEMPLATE_DIR / 'extended_data_fig3b_energy_decomposition_template.csv'
FILE_EXT_C = TEMPLATE_DIR / 'extended_fig3c_incremental_energy_template.csv'
POWER_CSV_CANDIDATES = [SCRIPT_DIR / 'power_profile_sustained.csv', SCRIPT_DIR / 'power_profile_sustained(2).csv']

COLOR_THISWORK = '#27AE60'
COLOR_FPGA = '#27AE60'
COLOR_ORIN = '#2980B9'
COLOR_4090 = '#C0392B'
COLOR_CPU = '#E67E22'
PLATFORM_META = {
    'This work': {'color': COLOR_FPGA, 'marker': '*'},
    'Jetson Orin NX': {'color': COLOR_ORIN, 'marker': 'o'},
    'CPU i7-13700': {'color': COLOR_CPU, 'marker': 'D'},
    'RTX 4090': {'color': COLOR_4090, 'marker': 's'},
}
COLOR_GRID = '#dadada'
COLOR_TEXT = '#222222'
COLOR_TRACE = '#2c7fb8'
COLOR_TRACE_LIGHT = '#91c3e9'
PHASE_COLORS = {'H2D_Transfer': '#dbeaf6', 'GPU_Compute': '#7fb6e6', 'D2H_Transfer': '#dbeaf6'}
PHASE_LABELS = {'H2D_Transfer': 'H2D', 'GPU_Compute': 'GPU', 'D2H_Transfer': 'D2H'}
ACTIVE_PHASES = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']
LATENCY_FPGA_MS = 0.21149


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing required template file: {path}')
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def maybe_load_power_trace():
    for p in POWER_CSV_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [str(c).strip() for c in df.columns]
            return df, p
    raise FileNotFoundError('Missing Jetson power trace CSV (power_profile_sustained.csv or power_profile_sustained(2).csv).')


def num(s):
    return pd.to_numeric(s, errors='coerce')


def integrate_energy(t, p):
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(p, t))


def add_panel_label(ax, label):
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left', va='bottom')


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def prepare_power_trace():
    df, path = maybe_load_power_trace()
    req = {'Time_s', 'Power_W', 'Phase'}
    if not req.issubset(df.columns):
        raise ValueError(f'Missing columns in {path}: {req - set(df.columns)}')
    df = df.sort_values('Time_s').reset_index(drop=True)
    active_mask = df['Phase'].isin(ACTIVE_PHASES)
    first_active_idx = int(active_mask[active_mask].index[0])
    last_active_idx = int(active_mask[active_mask].index[-1])
    t_active_start = float(df.loc[first_active_idx, 'Time_s'])
    t_active_end = float(df.loc[last_active_idx, 'Time_s'])
    active_duration_s = t_active_end - t_active_start
    idle_end_idx = first_active_idx - 1
    idle_start_idx = idle_end_idx
    while idle_start_idx >= 0 and df.loc[idle_start_idx, 'Phase'] == 'Idle':
        idle_start_idx -= 1
    idle_start_idx += 1
    recent_idle_df = df.iloc[idle_start_idx:first_active_idx].copy()
    idle_baseline_w = float(recent_idle_df['Power_W'].mean())
    df['Power_Dynamic_W'] = df['Power_W'] - idle_baseline_w
    padding_s = active_duration_s * 0.15
    plot_start_s = t_active_start - padding_s
    plot_end_s = t_active_end + padding_s
    plot_df = df[(df['Time_s'] >= plot_start_s) & (df['Time_s'] <= plot_end_s)].copy()
    plot_df['Time_ms'] = (plot_df['Time_s'] - plot_start_s) * 1000.0
    plot_df['Power_Dynamic_Smooth_W'] = plot_df['Power_Dynamic_W'].rolling(window=15, center=True, min_periods=1).mean()
    phase_stats = {}
    for phase in ACTIVE_PHASES:
        g = df[df['Phase'] == phase].copy()
        start_s = float(g['Time_s'].min())
        end_s = float(g['Time_s'].max())
        duration_s = end_s - start_s
        pos_energy = integrate_energy(g['Time_s'].to_numpy(), np.clip(g['Power_Dynamic_W'].to_numpy(), a_min=0.0, a_max=None))
        phase_stats[phase] = {
            'start_s': start_s, 'end_s': end_s, 'duration_ms': duration_s * 1000.0,
            'percent_time': duration_s / active_duration_s * 100.0,
            'avg_dynamic_power_w': pos_energy / duration_s if duration_s > 0 else np.nan,
            'dynamic_energy_j': pos_energy,
        }
    return {'plot_df': plot_df, 'phase_stats': phase_stats, 'idle_baseline_w': idle_baseline_w, 'plot_start_s': plot_start_s,
            'jetson_total_dynamic_energy_pos_j': sum(v['dynamic_energy_j'] for v in phase_stats.values()),
            'fpga_total_dynamic_energy_j': (0.6) * LATENCY_FPGA_MS / 1000.0}


def load_ext_b_data(power_data):
    df = read_csv(FILE_EXT_B)
    for c in ['duration_ms', 'time_fraction_percent', 'avg_dynamic_power_w', 'dynamic_energy_j', 'energy_fraction_percent', 'idle_baseline_w']:
        if c in df.columns:
            df[c] = num(df[c])
    # auto-fill Jetson rows from trace if blank
    for phase in ACTIVE_PHASES:
        mask = (df['platform'] == 'Jetson Orin NX') & (df['phase'] == phase)
        if mask.any():
            s = power_data['phase_stats'][phase]
            df.loc[mask, 'duration_ms'] = df.loc[mask, 'duration_ms'].fillna(s['duration_ms'])
            df.loc[mask, 'time_fraction_percent'] = df.loc[mask, 'time_fraction_percent'].fillna(s['percent_time'])
            df.loc[mask, 'avg_dynamic_power_w'] = df.loc[mask, 'avg_dynamic_power_w'].fillna(s['avg_dynamic_power_w'])
            df.loc[mask, 'dynamic_energy_j'] = df.loc[mask, 'dynamic_energy_j'].fillna(s['dynamic_energy_j'])
            df.loc[mask, 'idle_baseline_w'] = df.loc[mask, 'idle_baseline_w'].fillna(power_data['idle_baseline_w'])
    jetson_total = df.loc[df['platform'] == 'Jetson Orin NX', 'dynamic_energy_j'].sum(skipna=True)
    if jetson_total > 0:
        maskj = (df['platform'] == 'Jetson Orin NX') & (df['dynamic_energy_j'].notna())
        df.loc[maskj, 'energy_fraction_percent'] = df.loc[maskj, 'energy_fraction_percent'].fillna(df.loc[maskj, 'dynamic_energy_j'] / jetson_total * 100.0)
    # auto-fill this-work row if blank
    maskf = df['platform'] == 'This work'
    if maskf.any():
        df.loc[maskf, 'dynamic_energy_j'] = df.loc[maskf, 'dynamic_energy_j'].fillna(power_data['fpga_total_dynamic_energy_j'])
        df.loc[maskf, 'energy_fraction_percent'] = df.loc[maskf, 'energy_fraction_percent'].fillna(100.0)
    return df


def draw_ext_a(ax, power_data):
    plot_df = power_data['plot_df']
    phase_stats = power_data['phase_stats']
    plot_start_s = power_data['plot_start_s']
    idle_baseline_w = power_data['idle_baseline_w']
    for phase in ACTIVE_PHASES:
        s = phase_stats[phase]
        x0 = (s['start_s'] - plot_start_s) * 1000.0
        x1 = (s['end_s'] - plot_start_s) * 1000.0
        ax.axvspan(x0, x1, facecolor=PHASE_COLORS[phase], alpha=0.35, zorder=0)
    ax.plot(plot_df['Time_ms'], plot_df['Power_Dynamic_W'], color=COLOR_TRACE_LIGHT, linewidth=0.9, alpha=0.85, drawstyle='steps-post', label='Raw')
    ax.plot(plot_df['Time_ms'], plot_df['Power_Dynamic_Smooth_W'], color=COLOR_TRACE, linewidth=1.35, label='Smoothed')
    ax.axhline(0, color='0.45', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Execution time for a single batch (ms)')
    ax.set_ylabel('Dynamic power above idle (W)')
    ax.set_title('Jetson dynamic power profile', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    y_min = min(plot_df['Power_Dynamic_W'].min(), -0.15)
    y_max = max(plot_df['Power_Dynamic_Smooth_W'].max(), plot_df['Power_Dynamic_W'].max())
    y_span = max(y_max - y_min, 0.3)
    ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.25 * y_span)
    text_y = ax.get_ylim()[1] - 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for phase in ACTIVE_PHASES:
        s = phase_stats[phase]
        x0 = (s['start_s'] - plot_start_s) * 1000.0
        x1 = (s['end_s'] - plot_start_s) * 1000.0
        xc = 0.5 * (x0 + x1)
        ax.text(xc, text_y, f"{PHASE_LABELS[phase]}\n{s['duration_ms']:.1f} ms ({s['percent_time']:.1f}%)", ha='center', va='top', fontsize=4.8, color=COLOR_TRACE, bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.8))
    ax.text(0.015, 0.985, f'Idle baseline = mean power of the idle segment immediately before H2D = {idle_baseline_w:.3f} W', transform=ax.transAxes, ha='left', va='top', fontsize=4.8, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(frameon=False, loc='lower left')


def draw_ext_b(ax, df):
    jet = df[df['platform'] == 'Jetson Orin NX'].copy()
    fpga = df[df['platform'] == 'This work'].copy()
    order = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']
    x_positions = np.array([0.0, 1.05])
    bar_width = 0.46
    bottom = 0.0
    for phase in order:
        row = jet[jet['phase'] == phase]
        frac = float(row['energy_fraction_percent'].iloc[0]) if len(row) else 0.0
        ax.bar(x_positions[0], frac, width=bar_width, bottom=bottom, color=PHASE_COLORS[phase], edgecolor='white', linewidth=0.6)
        if frac >= 8:
            ax.text(x_positions[0], bottom + frac / 2, f"{PHASE_LABELS[phase]}\n{frac:.1f}%", ha='center', va='center', fontsize=4.7, color='white' if phase == 'GPU_Compute' else 'black')
        bottom += frac
    ax.bar(x_positions[1], 100.0, width=bar_width, color=COLOR_THISWORK, edgecolor='white', linewidth=0.6)
    ax.text(x_positions[1], 50.0, 'This work\n100%', ha='center', va='center', fontsize=5, color='white')
    jet_total = float(jet['dynamic_energy_j'].sum()) if len(jet) else np.nan
    fpga_total = float(fpga['dynamic_energy_j'].iloc[0]) if len(fpga) else np.nan
    ax.set_ylim(0, 100)
    ax.set_ylabel('Dynamic energy composition (%)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Jetson\nOrin NX', 'This work'])
    ax.set_title('Phase-wise dynamic-energy decomposition', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    if np.isfinite(jet_total):
        ax.text(x_positions[0], -11.5, f'{jet_total * 1000.0:.2f} mJ', ha='center', va='top', fontsize=4.8)
    if np.isfinite(fpga_total):
        ax.text(x_positions[1], -11.5, f'{fpga_total * 1000.0:.4f} mJ', ha='center', va='top', fontsize=4.8, color=COLOR_THISWORK)
    if np.isfinite(jet_total) and np.isfinite(fpga_total) and fpga_total > 0:
        ax.text(0.5, 1.03, f'Jetson / this-work dynamic energy ≈ {jet_total / fpga_total:.0f}×', transform=ax.transAxes, ha='center', va='bottom', fontsize=4.8, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(handles=[Patch(color=PHASE_COLORS['H2D_Transfer'], label='H2D'), Patch(color=PHASE_COLORS['GPU_Compute'], label='GPU'), Patch(color=PHASE_COLORS['D2H_Transfer'], label='D2H'), Patch(color=COLOR_THISWORK, label='This work')], frameon=False, loc='upper right')



def load_ext_c_data():
    """Load incremental-energy data for Extended Data.

    Required columns:
        platform, target_fps, processed_fps
    Preferred columns:
        incremental_power_w, incremental_energy_per_frame_mj
    If incremental_energy_per_frame_mj is blank, it is computed as
        1000 * incremental_power_w / processed_fps.
    """
    df = read_csv(FILE_EXT_C)
    if df.empty:
        return df
    for c in ['target_fps', 'processed_fps', 'idle_power_w', 'active_power_w',
              'incremental_power_w', 'incremental_energy_per_frame_mj', 'bubble_size_w']:
        if c in df.columns:
            df[c] = num(df[c])
    if 'incremental_power_w' not in df.columns:
        if {'active_power_w', 'idle_power_w'}.issubset(df.columns):
            df['incremental_power_w'] = df['active_power_w'] - df['idle_power_w']
        else:
            df['incremental_power_w'] = np.nan
    if 'incremental_energy_per_frame_mj' not in df.columns:
        df['incremental_energy_per_frame_mj'] = np.nan
    m = df['incremental_energy_per_frame_mj'].isna() & df['incremental_power_w'].notna() & df['processed_fps'].notna()
    df.loc[m, 'incremental_energy_per_frame_mj'] = 1000.0 * df.loc[m, 'incremental_power_w'] / df.loc[m, 'processed_fps']
    if 'bubble_size_w' not in df.columns:
        df['bubble_size_w'] = df['incremental_power_w']
    else:
        df['bubble_size_w'] = df['bubble_size_w'].fillna(df['incremental_power_w'])
    return df


def add_platform_legend(ax, loc='upper right', ncol=2):
    handles = []
    labels = []
    for name in ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']:
        meta = PLATFORM_META[name]
        handles.append(plt.Line2D([0], [0], marker=meta['marker'], linestyle='None',
                                  markerfacecolor=meta['color'], markeredgecolor='white',
                                  markeredgewidth=0.4, markersize=5.2, label=name))
        labels.append(name)
    ax.legend(handles, labels, frameon=False, loc=loc, ncol=ncol,
              handletextpad=0.45, columnspacing=0.8, borderaxespad=0.2)


def draw_ext_c(ax, df):
    ax.set_title('Incremental energy per frame versus throughput', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.grid(color=COLOR_GRID, linewidth=0.55, which='both')
    ax.set_axisbelow(True)
    df = df.dropna(subset=['processed_fps', 'incremental_energy_per_frame_mj']).copy()
    df = df[df['processed_fps'].astype(float) > 0]

    def size_from_power(w):
        try:
            w = float(w)
        except Exception:
            w = 0.0
        ref = max(w, 0.01)
        return 14 + 6.0 * np.sqrt(ref if ref > 1 else ref * 100)

    xvals, yvals = [], []
    for name in ['This work', 'Jetson Orin NX', 'CPU i7-13700', 'RTX 4090']:
        sub = df[df['platform'].astype(str).eq(name)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values('processed_fps')
        meta = PLATFORM_META[name]
        xs = sub['processed_fps'].astype(float).to_numpy()
        ys = sub['incremental_energy_per_frame_mj'].astype(float).to_numpy()
        ss = [size_from_power(v if pd.notna(v) else np.nan) for v in sub.get('bubble_size_w', sub['incremental_power_w'])]
        ax.plot(xs, ys, color=meta['color'], lw=0.95, alpha=0.65, zorder=2)
        ax.scatter(xs, ys, s=ss, marker=meta['marker'], color=meta['color'],
                   edgecolor='white', linewidth=0.45, alpha=0.96, zorder=3)
        xvals.extend(xs.tolist())
        yvals.extend(ys.tolist())

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
    ax.axvspan(xmin, 180, color='0.96', zorder=0)
    ax.axvline(180, color='0.42', lw=0.95, ls='--', zorder=1)
    ax.text(180, 0.04, '180 FPS real-time\nrequirement', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=4.8, color='0.36')

    ymin = max(0.03, min(yvals) * 0.75 if yvals else 0.03)
    ymax = max(1.0, max(yvals) * 1.35 if yvals else 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_yscale('log')
    ax.set_xlabel('Processed throughput (FPS)')
    ax.set_ylabel('Incremental energy per frame (mJ)')
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.text(0.02, 0.02, 'Incremental energy = (active − idle) / throughput.', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=4.6, color='0.42')
    add_platform_legend(ax, loc='upper right', ncol=2)


def add_caption(fig):
    caption = ('Extended Data Fig. 3 | Dynamic-power overheads and incremental energy. '
               'a, Baseline-subtracted Jetson Orin NX power trace for a single batch, using the mean idle power immediately before H2D as the idle baseline. '
               'b, Phase-wise dynamic-energy decomposition. '
               'c, Incremental energy per frame, defined as active minus idle power divided by processed throughput; full-chip energy is reported in Fig. 3d.')
    fig.text(0.08, 0.02, caption, ha='left', va='bottom', fontsize=5.4, color=COLOR_TEXT, wrap=True)


def main():
    power_data = prepare_power_trace()
    extb_df = load_ext_b_data(power_data)
    extc_df = load_ext_c_data()
    fig = plt.figure(figsize=(180 / 25.4, 145 / 25.4))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.16, top=0.93,
                          width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.95],
                          hspace=0.42, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    draw_ext_a(ax_a, power_data)
    draw_ext_b(ax_b, extb_df)
    draw_ext_c(ax_c, extc_df)
    for ax, label in zip([ax_a, ax_b, ax_c], list('abc')):
        add_panel_label(ax, label)
    add_caption(fig)
    out_pdf = SCRIPT_DIR / 'ExtendedData_Fig3_v5_incremental_energy.pdf'
    out_png = SCRIPT_DIR / 'ExtendedData_Fig3_v5_incremental_energy.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(out_pdf)
    print(out_png)

if __name__ == '__main__':
    main()
