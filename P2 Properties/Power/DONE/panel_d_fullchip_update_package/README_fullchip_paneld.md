# Panel d full-chip energy update

This update changes Fig. 3 panel d to plot **full-chip energy per frame**:

```text
full_chip_energy_per_frame_mJ = 1000 * full_chip_power_W / processed_fps
```

For measured CPU/GPU/Jetson rows, `full_chip_power_w` is the active total/device power measured during the active window.

Incremental energy is moved to Extended Data panel c:

```text
incremental_energy_per_frame_mJ = 1000 * (active_power_W - idle_power_W) / processed_fps
```

## Files

- `Fig3_main_v17_fullchip_paneld.py`: main Fig. 3 script; panel d reads `template/fig3d_energy_throughput_template.csv`.
- `ExtendedData_Fig3_v5_incremental_energy.py`: Extended Data script; new panel c reads `template/extended_fig3c_incremental_energy_template.csv`.
- `template/fig3d_energy_throughput_template.csv`: main panel d full-chip template.
- `template/extended_fig3c_incremental_energy_template.csv`: Extended Data incremental-energy template.
- `measure_panel_d_*_fullchip.py`: measurement scripts now output both full-chip and incremental energy columns.

## Usage

Copy these files into the same project folder as your existing templates and run the updated plotting scripts. Existing panel a/b/c/e/f templates remain unchanged.
