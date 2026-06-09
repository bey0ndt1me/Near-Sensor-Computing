from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
POWER_CSV_CANDIDATES = ['power_profile_sustained.csv', 'power_profile_sustained(2).csv']
COLOR_THISWORK = '#2ca25f'
COLOR_GRID = '#dadada'
COLOR_TEXT = '#222222'
COLOR_TRACE = '#2c7fb8'
COLOR_TRACE_LIGHT = '#91c3e9'
PHASE_COLORS = {'H2D_Transfer': '#dbeaf6', 'GPU_Compute': '#7fb6e6', 'D2H_Transfer': '#dbeaf6'}
PHASE_LABELS = {'H2D_Transfer': 'H2D', 'GPU_Compute': 'GPU', 'D2H_Transfer': 'D2H'}
ACTIVE_PHASES = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']
LATENCY_FPGA_MS = 0.21149


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
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def prepare_power_trace():
    df, path = maybe_load_csv(POWER_CSV_CANDIDATES)
    if df is None:
        raise FileNotFoundError(f'Could not find any of: {POWER_CSV_CANDIDATES}')
    required = {'Time_s', 'Power_W', 'Phase'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns in {path}: {required - set(df.columns)}')
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
            'start_s': start_s,
            'end_s': end_s,
            'duration_ms': duration_s * 1000.0,
            'percent_time': duration_s / active_duration_s * 100.0,
            'dynamic_energy_pos_j': pos_energy,
        }
    return {
        'plot_df': plot_df,
        'phase_stats': phase_stats,
        'idle_baseline_w': idle_baseline_w,
        'plot_start_s': plot_start_s,
        'jetson_total_dynamic_energy_pos_j': sum(v['dynamic_energy_pos_j'] for v in phase_stats.values()),
        'fpga_total_dynamic_energy_j': (0.6) * LATENCY_FPGA_MS / 1000.0,
    }


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
        ax.text(xc, text_y, f"{PHASE_LABELS[phase]}\n{s['duration_ms']:.1f} ms ({s['percent_time']:.1f}%)",
                ha='center', va='top', fontsize=4.8, color=COLOR_TRACE,
                bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.8))
    ax.text(0.015, 0.985, f'Idle baseline = mean power of the idle segment immediately before H2D = {idle_baseline_w:.3f} W',
            transform=ax.transAxes, ha='left', va='top', fontsize=4.8,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(frameon=False, loc='lower left')


def draw_ext_b(ax, power_data):
    phase_stats = power_data['phase_stats']
    jetson_total = max(power_data['jetson_total_dynamic_energy_pos_j'], 1e-15)
    fpga_total = power_data['fpga_total_dynamic_energy_j']
    order = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']
    fracs = [phase_stats[p]['dynamic_energy_pos_j'] / jetson_total * 100.0 for p in order]
    x_positions = np.array([0.0, 1.05])
    bar_width = 0.46
    bottom = 0.0
    for phase, frac in zip(order, fracs):
        ax.bar(x_positions[0], frac, width=bar_width, bottom=bottom, color=PHASE_COLORS[phase], edgecolor='white', linewidth=0.6)
        if frac >= 8:
            ax.text(x_positions[0], bottom + frac / 2, f'{PHASE_LABELS[phase]}\n{frac:.1f}%', ha='center', va='center', fontsize=4.7, color='white' if phase == 'GPU_Compute' else 'black')
        bottom += frac
    ax.bar(x_positions[1], 100.0, width=bar_width, color=COLOR_THISWORK, edgecolor='white', linewidth=0.6)
    ax.text(x_positions[1], 50.0, 'This work\n100%', ha='center', va='center', fontsize=5, color='white')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Dynamic energy composition (%)')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Jetson\nOrin NX', 'This work'])
    ax.set_title('Phase-wise dynamic-energy decomposition', loc='left', fontweight='bold', pad=8)
    strip_axes(ax)
    ax.text(x_positions[0], -11.5, f'{jetson_total * 1000.0:.2f} mJ', ha='center', va='top', fontsize=4.8)
    ax.text(x_positions[1], -11.5, f'{fpga_total * 1000.0:.4f} mJ', ha='center', va='top', fontsize=4.8, color=COLOR_THISWORK)
    ratio = jetson_total / fpga_total if fpga_total > 0 else np.nan
    ax.text(0.5, 1.03, f'Jetson / this-work dynamic energy ≈ {ratio:.0f}×', transform=ax.transAxes, ha='center', va='bottom', fontsize=4.8,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85'))
    ax.legend(handles=[Patch(color=PHASE_COLORS['H2D_Transfer'], label='H2D'), Patch(color=PHASE_COLORS['GPU_Compute'], label='GPU'), Patch(color=PHASE_COLORS['D2H_Transfer'], label='D2H'), Patch(color=COLOR_THISWORK, label='This work')],
              frameon=False, loc='upper right')


def add_caption(fig):
    caption = (
        "Extended Data Fig. 3 | Microarchitectural explanation of Jetson overheads. "
        "a, Baseline-subtracted Jetson Orin NX power trace for a single batch, using the mean idle power immediately before H2D as the idle baseline. "
        "b, Phase-wise dynamic-energy decomposition."
    )
    fig.text(0.08, 0.02, caption, ha='left', va='bottom', fontsize=5.4, color=COLOR_TEXT, wrap=True)


def main():
    power_data = prepare_power_trace()
    fig = plt.figure(figsize=(180 / 25.4, 95 / 25.4))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, bottom=0.22, top=0.92, width_ratios=[1.55, 1.0], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_ext_a(ax_a, power_data)
    draw_ext_b(ax_b, power_data)
    for ax, label in zip([ax_a, ax_b], list('ab')):
        add_panel_label(ax, label)
    add_caption(fig)
    out_pdf = SCRIPT_DIR / 'ExtendedData_Fig3_v3.pdf'
    out_png = SCRIPT_DIR / 'ExtendedData_Fig3_v3.png'
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    print(out_pdf)
    print(out_png)


if __name__ == '__main__':
    main()
