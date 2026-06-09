import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==========================================
# Global plotting style (Nature-like)
# ==========================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

# Colors
COLOR_FPGA = '#27AE60'
COLOR_ORIN = '#2980B9'
COLOR_ORIN_RAW = '#85C1E9'
PHASE_COLORS = {
    'H2D_Transfer': '#AED6F1',
    'GPU_Compute': '#5DADE2',
    'D2H_Transfer': '#AED6F1',
}
PHASE_LABELS = {
    'H2D_Transfer': 'H2D transfer',
    'GPU_Compute': 'GPU compute',
    'D2H_Transfer': 'D2H transfer',
}
ACTIVE_PHASES = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']

# ==========================================
# User-editable configuration
# ==========================================
# The CSV will be searched in the script directory using the following candidate names.
CSV_CANDIDATES = [
    'power_profile_sustained.csv',
    'power_profile_sustained(2).csv',
]

# Rolling smoothing window for the Jetson trace.
SMOOTH_WINDOW = 15

# Padding shown before and after the active batch in panel-e.
PLOT_PADDING_RATIO = 0.15  # 15% of active duration on each side

# FPGA placeholders for panel-f total dynamic energy.
# Replace these with measured values if available.
FPGA_IDLE_POWER_W = 4.0
FPGA_ACTIVE_POWER_W = 4.6
FPGA_BATCH_LATENCY_S = 0.00021149  # 211.49 us

# Whether to save PNG files in addition to PDF.
SAVE_PNG = True


# ==========================================
# Utilities
# ==========================================
def find_csv(base_dir: Path) -> Path:
    for name in CSV_CANDIDATES:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any CSV file in {base_dir}. Tried: {CSV_CANDIDATES}"
    )


def integrate_energy(time_s: np.ndarray, power_w: np.ndarray) -> float:
    """Integrate energy using trapezoidal integration."""
    if len(time_s) < 2:
        return 0.0
    return float(np.trapezoid(power_w, time_s))


# ==========================================
# Load data
# ==========================================
script_dir = Path(__file__).resolve().parent
csv_path = find_csv(script_dir)
print(f'Loading power profile from: {csv_path}')

df = pd.read_csv(csv_path).copy()
required_columns = {'Time_s', 'Power_W', 'Phase'}
missing = required_columns.difference(df.columns)
if missing:
    raise ValueError(f'Missing required columns: {missing}')

df = df.sort_values('Time_s').reset_index(drop=True)

# ==========================================
# Determine the active batch and the idle baseline
# Baseline rule: median of the contiguous Idle segment immediately before active start.
# ==========================================
active_mask = df['Phase'].isin(ACTIVE_PHASES)
if not active_mask.any():
    raise ValueError(f'No active phases found. Expected one or more of: {ACTIVE_PHASES}')

first_active_idx = int(active_mask[active_mask].index[0])
last_active_idx = int(active_mask[active_mask].index[-1])

t_active_start = float(df.loc[first_active_idx, 'Time_s'])
t_active_end = float(df.loc[last_active_idx, 'Time_s'])
active_duration_s = t_active_end - t_active_start

# Find the contiguous idle segment immediately before first active phase.
idle_end_idx = first_active_idx - 1
idle_start_idx = idle_end_idx
while idle_start_idx >= 0 and df.loc[idle_start_idx, 'Phase'] == 'Idle':
    idle_start_idx -= 1
idle_start_idx += 1

recent_idle_df = df.iloc[idle_start_idx:first_active_idx].copy()
if recent_idle_df.empty or not (recent_idle_df['Phase'] == 'Idle').all():
    # Fallback: last Idle rows before first active phase.
    recent_idle_df = df[(df['Time_s'] < t_active_start) & (df['Phase'] == 'Idle')].copy()
    if recent_idle_df.empty:
        raise ValueError('Could not determine an idle segment immediately before the active batch.')

idle_baseline_w = float(recent_idle_df['Power_W'].median())
print(f'Recent-idle baseline power = {idle_baseline_w:.5f} W')

# Baseline-subtracted dynamic power
# Positive values represent additional power above the recent-idle baseline.
df['Power_Dynamic_W'] = df['Power_W'] - idle_baseline_w

# ==========================================
# Build panel-e plotting window
# ==========================================
padding_s = active_duration_s * PLOT_PADDING_RATIO
plot_start_s = t_active_start - padding_s
plot_end_s = t_active_end + padding_s
plot_df = df[(df['Time_s'] >= plot_start_s) & (df['Time_s'] <= plot_end_s)].copy()
plot_df['Time_ms'] = (plot_df['Time_s'] - plot_start_s) * 1000.0
plot_df['Power_Dynamic_Smooth_W'] = (
    plot_df['Power_Dynamic_W']
    .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
    .mean()
)

# Collect per-phase timing and energy statistics.
phase_stats = {}
for phase in ACTIVE_PHASES:
    phase_df = df[df['Phase'] == phase].copy()
    if phase_df.empty:
        continue
    start_s = float(phase_df['Time_s'].min())
    end_s = float(phase_df['Time_s'].max())
    duration_s = end_s - start_s
    dynamic_energy_j = integrate_energy(
        phase_df['Time_s'].to_numpy(),
        phase_df['Power_Dynamic_W'].to_numpy(),
    )
    dynamic_energy_pos_j = integrate_energy(
        phase_df['Time_s'].to_numpy(),
        np.clip(phase_df['Power_Dynamic_W'].to_numpy(), a_min=0.0, a_max=None),
    )
    phase_stats[phase] = {
        'start_s': start_s,
        'end_s': end_s,
        'duration_s': duration_s,
        'duration_ms': duration_s * 1000.0,
        'percent_time': (duration_s / active_duration_s * 100.0) if active_duration_s > 0 else 0.0,
        'dynamic_energy_j': dynamic_energy_j,
        'dynamic_energy_pos_j': dynamic_energy_pos_j,
    }

jetson_total_dynamic_energy_j = sum(v['dynamic_energy_j'] for v in phase_stats.values())
jetson_total_dynamic_energy_pos_j = sum(v['dynamic_energy_pos_j'] for v in phase_stats.values())

# FPGA dynamic energy estimate for panel-f.
# Replace with measured values when available.
fpga_dynamic_power_w = max(FPGA_ACTIVE_POWER_W - FPGA_IDLE_POWER_W, 0.0)
fpga_total_dynamic_energy_j = fpga_dynamic_power_w * FPGA_BATCH_LATENCY_S

print('\nPhase summary (Jetson):')
for phase in ACTIVE_PHASES:
    if phase not in phase_stats:
        continue
    s = phase_stats[phase]
    print(
        f"  {phase:>12s}: {s['duration_ms']:.2f} ms, "
        f"{s['percent_time']:.1f}% of batch, "
        f"dynamic energy = {s['dynamic_energy_j']*1000:.3f} mJ "
        f"(positive-only for decomposition = {s['dynamic_energy_pos_j']*1000:.3f} mJ)"
    )
print(f'  Total dynamic energy (Jetson, signed) = {jetson_total_dynamic_energy_j*1000:.3f} mJ')
print(f'  Total dynamic energy (Jetson, positive-only decomposition) = {jetson_total_dynamic_energy_pos_j*1000:.3f} mJ')
print(
    f'  Total dynamic energy (FPGA, configurable placeholder) = '
    f'{fpga_total_dynamic_energy_j*1000:.6f} mJ'
)

# ==========================================
# Panel-e: baseline-subtracted dynamic power trace
# ==========================================
fig_e, ax_e = plt.subplots(figsize=(88/25.4, 55/25.4))
fig_e.subplots_adjust(left=0.14, right=0.98, bottom=0.2, top=0.9)

# Background shading by active phase.
for phase in ACTIVE_PHASES:
    if phase not in phase_stats:
        continue
    s = phase_stats[phase]
    x0 = (s['start_s'] - plot_start_s) * 1000.0
    x1 = (s['end_s'] - plot_start_s) * 1000.0
    ax_e.axvspan(x0, x1, facecolor=PHASE_COLORS[phase], alpha=0.25, zorder=0)

# Raw and smoothed dynamic power traces.
ax_e.plot(
    plot_df['Time_ms'], plot_df['Power_Dynamic_W'],
    color=COLOR_ORIN_RAW, linewidth=0.9, alpha=0.65,
    drawstyle='steps-post', label='Jetson Orin NX (raw samples)'
)
ax_e.plot(
    plot_df['Time_ms'], plot_df['Power_Dynamic_Smooth_W'],
    color=COLOR_ORIN, linewidth=1.4,
    label='Jetson Orin NX (smoothed)'
)

# Zero line indicates the recent-idle baseline.
ax_e.axhline(0, color='0.4', linewidth=0.8, linestyle='--')

# Axis labels
ax_e.set_xlabel('Execution time for a single batch (ms)')
ax_e.set_ylabel('Dynamic power above idle (W)')

# Expand y-limits slightly for annotation room.
y_min = min(plot_df['Power_Dynamic_W'].min(), -0.15)
y_max = max(plot_df['Power_Dynamic_Smooth_W'].max(), plot_df['Power_Dynamic_W'].max())
y_span = max(y_max - y_min, 0.3)
ax_e.set_ylim(y_min - 0.05 * y_span, y_max + 0.25 * y_span)

# Annotate each phase with duration and time percentage.
text_y = ax_e.get_ylim()[1] - 0.08 * (ax_e.get_ylim()[1] - ax_e.get_ylim()[0])
for phase in ACTIVE_PHASES:
    if phase not in phase_stats:
        continue
    s = phase_stats[phase]
    x0 = (s['start_s'] - plot_start_s) * 1000.0
    x1 = (s['end_s'] - plot_start_s) * 1000.0
    xc = 0.5 * (x0 + x1)
    annotation = (
        f"{PHASE_LABELS[phase]}\n"
        f"{s['duration_ms']:.1f} ms ({s['percent_time']:.1f}%)"
    )
    ax_e.text(
        xc, text_y, annotation,
        ha='center', va='top', fontsize=5, color=COLOR_ORIN,
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75)
    )

# Add note explaining the baseline and the FPGA interpretation.
baseline_note = (
    f"Idle baseline = median power of the contiguous Idle segment immediately\n"
    f"preceding H2D transfer = {idle_baseline_w:.3f} W"
)
ax_e.text(
    0.015, 0.98, baseline_note,
    transform=ax_e.transAxes, ha='left', va='top', fontsize=5,
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='0.85', alpha=0.9)
)

fpga_note = (
    'Custom FPGA: single near-sensor streaming compute phase\n'
    'with no H2D / D2H transfer overhead'
)
ax_e.text(
    0.985, 0.06, fpga_note,
    transform=ax_e.transAxes, ha='right', va='bottom', fontsize=5, color=COLOR_FPGA,
    bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor=COLOR_FPGA, alpha=0.9)
)

ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.legend(frameon=False, loc='lower left')

# Save panel-e
panel_e_pdf = script_dir / 'panel_e_dynamic_power_trace.pdf'
fig_e.savefig(panel_e_pdf, format='pdf', dpi=300)
if SAVE_PNG:
    panel_e_png = script_dir / 'panel_e_dynamic_power_trace.png'
    fig_e.savefig(panel_e_png, format='png', dpi=300)
print(f'Saved panel-e to: {panel_e_pdf}')
if SAVE_PNG:
    print(f'Saved panel-e PNG to: {panel_e_png}')

# ==========================================
# Panel-f: normalized phase-wise dynamic-energy decomposition
# Purpose: show where the batch energy is spent, while avoiding a misleading
# near-invisible FPGA bar due to the very different absolute energy scales.
# ==========================================
fig_f, ax_f = plt.subplots(figsize=(60/25.4, 55/25.4))
fig_f.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.9)

# Normalized composition for Jetson.
jetson_component_order = ['H2D_Transfer', 'GPU_Compute', 'D2H_Transfer']
jetson_total = max(jetson_total_dynamic_energy_pos_j, 1e-15)
jetson_fracs = [phase_stats[p]['dynamic_energy_pos_j'] / jetson_total * 100.0 for p in jetson_component_order]

# FPGA is treated as 100% compute.
fpga_fracs = [100.0]

x_positions = np.array([0.0, 1.1])
bar_width = 0.48

# Jetson stacked bar
bottom = 0.0
for phase, frac in zip(jetson_component_order, jetson_fracs):
    ax_f.bar(
        x_positions[0], frac, width=bar_width, bottom=bottom,
        color=PHASE_COLORS[phase], edgecolor='white', linewidth=0.6,
        label=PHASE_LABELS[phase]
    )
    if frac >= 8:
        ax_f.text(
            x_positions[0], bottom + frac / 2,
            f"{PHASE_LABELS[phase]}\n{frac:.1f}%",
            ha='center', va='center', fontsize=5,
            color=('white' if phase == 'GPU_Compute' else 'black')
        )
    bottom += frac

# FPGA single compute bar
ax_f.bar(
    x_positions[1], 100.0, width=bar_width,
    color=COLOR_FPGA, edgecolor='white', linewidth=0.6,
    label='FPGA compute'
)
ax_f.text(
    x_positions[1], 50.0, 'FPGA compute\n100%',
    ha='center', va='center', fontsize=5, color='white'
)

ax_f.set_ylim(0, 100)
ax_f.set_ylabel('Dynamic energy composition (%)')
ax_f.set_xticks(x_positions)
ax_f.set_xticklabels(['Jetson Orin NX', 'Custom FPGA'])
ax_f.spines['top'].set_visible(False)
ax_f.spines['right'].set_visible(False)

# Totals below each bar.
jetson_total_mj = jetson_total_dynamic_energy_pos_j * 1000.0
fpga_total_mj = fpga_total_dynamic_energy_j * 1000.0
ax_f.text(
    x_positions[0], -13,
    f"Total = {jetson_total_mj:.2f} mJ",
    ha='center', va='top', fontsize=5
)
ax_f.text(
    x_positions[1], -13,
    f"Total = {fpga_total_mj:.4f} mJ",
    ha='center', va='top', fontsize=5, color=COLOR_FPGA
)

# Optional ratio annotation.
if fpga_total_dynamic_energy_j > 0:
    ratio = jetson_total_dynamic_energy_j / fpga_total_dynamic_energy_j
    ratio_note = f"Jetson / FPGA dynamic energy ≈ {ratio:.0f}×"
    ax_f.text(
        0.5, 1.04, ratio_note,
        transform=ax_f.transAxes, ha='center', va='bottom', fontsize=5,
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='0.85', alpha=0.9)
    )

# Figure note for FPGA total energy assumptions.
fpga_assumption_note = (
    'Jetson decomposition uses positive-only dynamic energy above the recent-idle baseline\n'
    'to avoid cancellation by near-baseline fluctuations.\n'
    f"FPGA total dynamic energy computed from configurable placeholders: P_active = {FPGA_ACTIVE_POWER_W:.2f} W, "
    f"P_idle = {FPGA_IDLE_POWER_W:.2f} W, latency = {FPGA_BATCH_LATENCY_S*1e6:.2f} μs"
)
ax_f.text(
    0.5, -0.28, fpga_assumption_note,
    transform=ax_f.transAxes, ha='center', va='top', fontsize=4.6,
    color='0.25'
)

# Custom legend (deduplicated)
handles = [
    mpatches.Patch(color=PHASE_COLORS['H2D_Transfer'], label='H2D transfer'),
    mpatches.Patch(color=PHASE_COLORS['GPU_Compute'], label='GPU compute'),
    mpatches.Patch(color=PHASE_COLORS['D2H_Transfer'], label='D2H transfer'),
    mpatches.Patch(color=COLOR_FPGA, label='FPGA compute'),
]
ax_f.legend(handles=handles, frameon=False, loc='upper right')

# Save panel-f
panel_f_pdf = script_dir / 'panel_f_energy_decomposition.pdf'
fig_f.savefig(panel_f_pdf, format='pdf', dpi=300, bbox_inches='tight')
if SAVE_PNG:
    panel_f_png = script_dir / 'panel_f_energy_decomposition.png'
    fig_f.savefig(panel_f_png, format='png', dpi=300, bbox_inches='tight')
print(f'Saved panel-f to: {panel_f_pdf}')
if SAVE_PNG:
    print(f'Saved panel-f PNG to: {panel_f_png}')

plt.show()
