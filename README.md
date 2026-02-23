# dream_ana
Online and offline analysis software for the DREAM instrument at the LCLS.

## Table of Contents

- [Main Features](#main-features)
- [Data Pipeline](#data-pipeline)
- [Software Structure](#software-structure)
- [Workflow](#workflow)
- [Online Plots](#online-plots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
- [Online Configuration (Plots)](#online-configuration-plots)
- [Offline Configuration (HDF5 Output)](#offline-configuration-hdf5-output)
- [Variable Reference](#variable-reference)
- [Custom Functions Reference](#custom-functions-reference)
- [Troubleshooting](#troubleshooting)

---

## Main Features
- Capable of running both online and offline
- Analysis configurable with human-readable files
- Plots-driven analysis for online, and h5-content-driven for offline

---

## Data Pipeline
<img width="929" height="292" alt="image" src="https://github.com/user-attachments/assets/da3ac2ba-1cd2-4e4e-bf13-9d998f95c03c" />

---

## Software Structure
<img width="1014" height="387" alt="image" src="https://github.com/user-attachments/assets/23cbb410-aefb-4632-88fc-3ee7c367596f" />

---

## Workflow
<img width="952" height="425" alt="image" src="https://github.com/user-attachments/assets/85ce42e1-1d09-41cc-a26d-b68ab652f68a" />

---

## Online Plots
<img width="3781" height="2070" alt="image" src="https://github.com/user-attachments/assets/31192a35-4d8f-4260-8d5e-6b4b32d594c3" />

---

## Installation

```bash
./install.sh
# or
pip install -e .
```

---

## Quick Start

```bash
# Set config directory
export CONFIGDIR=/path/to/dream/config/

# Online mode (real-time)
dream

# Offline mode (batch processing)
dream --exp <experiment_name> --run <run_number>

# With MPI
mpirun -n 8 dream --exp tmox12345 --run 42
```

---

## Configuration Files

```
dream/config/
├── instrument.yaml      # Instrument selection
├── dream/
│   ├── det.yaml        # Detector parameters (rarely edited)
│   ├── alg.yaml        # Algorithm definitions (rarely edited)
│   ├── online.yaml     # Plot configuration (frequently edited)
│   └── offline.yaml    # HDF5 output config (frequently edited)
```

---

## Online Configuration (Plots)

The `online.yaml` file defines real-time plots. Basic structure:

```yaml
nacc: 5          # Events to accumulate before updating

plots:
  plot_name:
    type: plot_type
    # type-specific parameters...
```

<details>
<summary><strong>multiline</strong> - Waveform Display</summary>

Display multiple waveforms stacked vertically.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `multiline` |
| `var` | List of variable names |
| `y_offset` | Vertical spacing between lines |

**Example:**
```yaml
wf[l]:
  type: multiline
  var: ['wf_l:mcp', 'wf_l:u1', 'wf_l:u2', 'wf_l:v1', 'wf_l:v2', 'wf_l:w1', 'wf_l:w2']
  y_offset: 100
```

</details>

<details>
<summary><strong>hist1d</strong> - 1D Histogram</summary>

Create a histogram from array data.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `hist1d` |
| `arange` | `{variable: [start, stop, step]}` |

**Example: TOF histogram**
```yaml
t[l]:
  type: hist1d
  arange: {'hit_l:t': [0, 15000, 1]}
```

**Example: Hit count distribution**
```yaml
n[l]:
  type: hist1d
  arange: {'hit_l:n': [0.5, 35.5, 1]}
```
Use 0.5 offset to center bins on integers.

</details>

<details>
<summary><strong>hist2d</strong> - 2D Histogram / Image</summary>

Create a 2D histogram displayed as an image.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `hist2d` |
| `arange` | `{x_var: [start, stop, step], y_var: [start, stop, step]}` |

**Example: Detector position image**
```yaml
z-y[l]:
  type: hist2d
  arange: {'hit_l:y': [-65, 65, 2], 'hit_l:z': [-65, 65, 2]}
```

**Example: PIPICO plot**
```yaml
pipico[l]:
  type: hist2d
  arange: {'ppc_l:pp1': [0, 12000, 30], 'ppc_l:pp2': [0, 12000, 30]}
```

</details>

<details>
<summary><strong>rollavg</strong> - Rolling Average</summary>

Track a scalar value over time with smoothing.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `rollavg` |
| `var` | Variable to average |
| `window` | `{w1: smoothing, w2: history_length}` |

**Example:**
```yaml
n[l]_rollavg_mcp:
  type: rollavg
  var: 'len_tpks_l:mcp'
  window: {'w1': 500, 'w2': 100}
```
- `w1=500`: Average over 500 events
- `w2=100`: Display 100 points

</details>

<details>
<summary><strong>rollavg_func</strong> - Rolling Average with Function</summary>

Apply a custom function before computing rolling average.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `rollavg_func` |
| `func` | `{name: function, args1: [vars], args2: [constants]}` |
| `window` | `{w1: smoothing, w2: history_length}` |

**Example: Filter by beam destination**
```yaml
n[l]_rollavg:
  type: rollavg_func
  func: {name: filter.filter_dest, args1: ['hit_l:n', 'timing:dest'], args2: [4]}
  window: {'w1': 500, 'w2': 100}
```

</details>

<details>
<summary><strong>sigbkg1d</strong> - Signal vs Background</summary>

Compare signal and background with automatic subtraction/division.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `sigbkg1d` |
| `plot_type` | `singleline_func` or `hist1d_func` |
| `op_type` | `sub` (sig-bkg) or `div` (sig/bkg) |
| `func_sig` | Signal function |
| `func_bkg` | Background function |
| `window` | (for singleline_func) Rolling window |
| `arange_var` | (for hist1d_func) Histogram bins |
| `norm_type` | (optional) `sum` or `count` |

**Example: ATM signal vs background**
```yaml
atm_sig_bkg:
  type: sigbkg1d
  plot_type: singleline_func
  op_type: sub
  window: 1000
  func_sig: {name: filter.atm, args1: ['atm:line', 'bld:xgmd', 'timing:280'], args2: [0.005, 1]}
  func_bkg: {name: filter.atm, args1: ['atm:line', 'bld:xgmd', 'timing:281'], args2: [0, 1]}
```

**Example: TOF histogram with normalization**
```yaml
t[l]_duck_goose_gatedOn_yz:
  type: sigbkg1d
  plot_type: hist1d_func
  op_type: sub
  func_sig: {name: filter.duck_goose_arr_gatedOn_xy, args1: ['hit_l:t', 'hit_l:n', 'timing:280', 'hit_l:y', 'hit_l:z'], args2: [1, -30, 30, -30, 30]}
  func_bkg: {name: filter.duck_goose_arr_gatedOn_xy, args1: ['hit_l:t', 'hit_l:n', 'timing:282', 'hit_l:y', 'hit_l:z'], args2: [0, -30, 30, -30, 30]}
  arange_var: {'hit_l:t': [550, 6500, 5]}
  norm_type: sum
  func_norm_sig: {name: filter.duck_goose_arr1, args1: ['bld:xgmd', 'timing:280'], args2: [1]}
  func_norm_bkg: {name: filter.duck_goose_arr1, args1: ['bld:xgmd', 'timing:282'], args2: [0]}
```

</details>

<details>
<summary><strong>scan_var_func</strong> - Scan Plot</summary>

Plot mean value vs a scan variable.

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `scan_var_func` |
| `func` | Function to compute value |
| `func_scan` | Function to get/filter scan variable |
| `decimal` | Rounding precision for grouping |
| `func_norm` | (optional) Normalization function |

**Example:**
```yaml
scan_1d_func_dest4_280_norm_xgmd:
  type: scan_var_func
  func: {name: filter.n_gatedOn_abc, args1: ['hit_l:n', 'hit_l:t', 'hit_l:y', 'hit_l:z'], args2: [2986, 3015, -20, 20, -20, 20]}
  func_scan: {name: filter.dest4_280, args1: ['scan:var1', 'timing:dest', 'timing:280'], args2: [4, 1]}
  decimal: 14
  func_norm: {args1: ['bld:xgmd']}
```

</details>

<details>
<summary><strong>hist1d_func / hist2d_func</strong> - Gated Histograms</summary>

Apply gating functions before histogramming.

**hist1d_func Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `hist1d_func` |
| `func` | Gating function |
| `arange_var` | `{label: [start, stop, step]}` |

**Example: TOF gated on position**
```yaml
t[l]_gated:
  type: hist1d_func
  func: {name: filter.a_gatedOn_bc, args1: ['hit_l:t', 'hit_l:y', 'hit_l:z'], args2: [-5, 5, -5, 5]}
  arange_var: {'gated t': [0, 15000, 1]}
```

**hist2d_func Parameters:**
| Parameter | Description |
|-----------|-------------|
| `type` | `hist2d_func` |
| `func_x` | Function for x-axis |
| `func_y` | Function for y-axis |
| `arange_var` | `{x_label: [...], y_label: [...]}` |

**Example: Position gated on TOF**
```yaml
y-z[l]_gated:
  type: hist2d_func
  func_x: {name: filter.a_gatedOn_b, args1: ['hit_l:z', 'hit_l:t'], args2: [4040, 4180]}
  func_y: {name: filter.a_gatedOn_b, args1: ['hit_l:y', 'hit_l:t'], args2: [4040, 4180]}
  arange_var: {'gated z': [-65, 65, 2], 'gated y': [-65, 65, 2]}
```

</details>

<details>
<summary><strong>Other Plot Types</strong></summary>

| Type | Description |
|------|-------------|
| `scan_var` | Simple scan (no function transform) |
| `scan2_var` / `scan2_var_func` | 2D scan (two scan variables) |
| `scan_hist1d` / `scan_hist1d_func` | Histogram per scan value |
| `singleline` / `singleline_func` | Single line plot |
| `singleimage` | Single 2D image |
| `rollavg1d` / `rollavg1d_func` | 1D rolling average |

</details>

---

## Offline Configuration (HDF5 Output)

The `offline.yaml` file defines data saved to HDF5.

<details>
<summary><strong>Global Settings</strong></summary>

```yaml
live: False        # False = process and exit; True = wait for new data
max_events:        # Limit events (empty = unlimited)
batch_size: 1000   # Events per batch
xpand: True        # Expand auxiliary data into ragged arrays
```

</details>

<details>
<summary><strong>HDF5 Path Configuration</strong></summary>

```yaml
h5:
  path1: /sdf/data/lcls/ds/tmo/
  path2: /scratch/arp/h5_v1/
  name1: run
  name2: .h5
```

Final path: `{path1}{exp}{path2}{name1}{run}{name2}`

Example: `/sdf/data/lcls/ds/tmo/tmox12345/scratch/arp/h5_v1/run42.h5`

</details>

<details>
<summary><strong>Data Output Types</strong></summary>

**Ragged (variable-length per event):**
```yaml
data:
  ragged:
    hit_l:
      var: [z, y, t, m]
```

**Uniform (fixed-length, NaN-padded):**
```yaml
data:
  uniform:
    tpks_e:
      var: ['0', '1']
      len: 100
```

**Auxiliary per-event data:**
```yaml
data:
  x:
    scan: [var1, var2]
    bld: [gmd, xgmd]
    timing: ['280','281','282','dest']
    atm: [edge, prom]
    epics: [las_ip2_atm_dly]
```

</details>

<details>
<summary><strong>Complete Example</strong></summary>

```yaml
live: False
max_events:
batch_size: 1000
xpand: True

h5:
  path1: /sdf/data/lcls/ds/tmo/
  path2: /scratch/arp/h5_v1/
  name1: run
  name2: .h5

log:
  path1: /sdf/data/lcls/ds/tmo/
  path2: /scratch/arp/log/

data:
  ragged:
    hit_l:
      var: [z, y, t, m]
  x:
    scan: [var1, var2]
    bld: [gmd, xgmd]
    timing: ['280','281','282','dest']
    atm: [edge, prom]
    epics: [las_ip2_atm_dly]
```

</details>

---

## Variable Reference

<details>
<summary><strong>Long Detector Variables</strong></summary>

Format: `detector:variable`

| Variable | Description |
|----------|-------------|
| **Hit Data** | |
| `hit_l:t` | Hit time (TOF) |
| `hit_l:y` | Hit Y position |
| `hit_l:z` | Hit Z position |
| `hit_l:n` | Hit count per event |
| `hit_l:m` | Hit multiplicity |
| **Waveforms** | |
| `wf_l:mcp` | MCP waveform |
| `wf_l:u1`, `wf_l:u2` | U delay line waveforms |
| `wf_l:v1`, `wf_l:v2` | V delay line waveforms |
| `wf_l:w1`, `wf_l:w2` | W delay line waveforms |
| **Timing Peaks** | |
| `tpks_l:mcp` | MCP timing peaks |
| `len_tpks_l:mcp` | Number of MCP peaks |
| **Diagnostics** | |
| `diag_l:tsum_u` | U-axis timing sum |
| `diag_l:tsum_v` | V-axis timing sum |
| `diag_l:tsum_w` | W-axis timing sum |
| **PIPICO** | |
| `ppc_l:pp1` | First ion TOF |
| `ppc_l:pp2` | Second ion TOF |

</details>

<details>
<summary><strong>Short Detector Variables</strong></summary>

Format: `detector:variable`

| Variable | Description |
|----------|-------------|
| **Hit Data** | |
| `hit_s:t` | Hit time (TOF) |
| `hit_s:y` | Hit Y position |
| `hit_s:z` | Hit Z position |
| `hit_s:n` | Hit count per event |
| `hit_s:m` | Hit multiplicity |
| **Waveforms** | |
| `wf_s:mcp` | MCP waveform |
| `wf_s:u1`, `wf_s:u2` | U delay line waveforms |
| `wf_s:v1`, `wf_s:v2` | V delay line waveforms |
| `wf_s:w1`, `wf_s:w2` | W delay line waveforms |
| **Timing Peaks** | |
| `tpks_s:mcp` | MCP timing peaks |
| `len_tpks_s:mcp` | Number of MCP peaks |
| **Diagnostics** | |
| `diag_s:tsum_u` | U-axis timing sum |
| `diag_s:tsum_v` | V-axis timing sum |
| `diag_s:tsum_w` | W-axis timing sum |
| **PIPICO** | |
| `ppc_s:pp1` | First ion TOF |
| `ppc_s:pp2` | Second ion TOF |

</details>

<details>
<summary><strong>Common Variables</strong></summary>

| Variable | Description |
|----------|-------------|
| **Beam Line Data** | |
| `bld:xgmd` | X-ray gas monitor |
| `bld:gmd` | Gas monitor |
| **Timing Codes** | |
| `timing:280` | Timing code 280 (laser on) |
| `timing:281` | Timing code 281 (laser off) |
| `timing:282` | Timing code 282 |
| `timing:dest` | Beam destination |
| **Scan** | |
| `scan:var1` | First scan variable |
| `scan:var2` | Second scan variable |
| **ATM** | |
| `atm:line` | ATM line data |
| `atm:gline` | ATM gated line |

</details>

---

## Custom Functions Reference

<details>
<summary><strong>filter.py Functions</strong></summary>

| Function | Parameters | Description |
|----------|------------|-------------|
| `filter.filter_dest` | `arr, dest, num` | Return arr where dest == num |
| `filter.gate1D_count` | `arr, l, r` | Count where l < arr < r |
| `filter.a_gatedOn_b` | `arr1, arr2, l2, r2` | Return arr1 where l2 < arr2 < r2 |
| `filter.a_gatedOn_bc` | `arr1, arr2, arr3, l2, r2, l3, r3` | Return arr1 where both arr2 and arr3 in ranges |
| `filter.n_gatedOn_abc` | `n_arr, arr1, arr2, arr3, l1, r1, l2, r2, l3, r3` | Count where all arrays in ranges |
| `filter.dest4_280` | `arr, dest, t280, num1, num2` | NaN where dest==num1 AND t280==num2 |
| `filter.atm` | `line, xgmd, ec, xgmd_min, ec_01` | Return line if xgmd > min AND ec == ec_01 |
| `filter.duck_goose_arr` | `arr, n_arr, ec, ec_01` | Filter with repeat based on ec |
| `filter.duck_goose_arr1` | `arr, ec, ec_01` | Filter where ec == ec_01 |
| `filter.duck_goose_arr_gatedOn_xy` | `arr, n_arr, ec, arr1, arr2, ec_01, l1, r1, l2, r2` | 2D gating with ec condition |

</details>

<details>
<summary><strong>repeat.py Functions</strong></summary>

| Function | Parameters | Description |
|----------|------------|-------------|
| `repeat.repeat` | `arr, n_arr` | Repeat elements by counts in n_arr |
| `repeat.repeat_dest4_280` | `arr, n_arr, dest, t280, num1, num2` | Repeat with filtering |

</details>

<details>
<summary><strong>Function Syntax</strong></summary>

```yaml
func:
  name: module.function    # From dream/custom/
  args1: [var1, var2]      # Variable names (from data)
  args2: [const1, const2]  # Constant values
```

**Example:** Gate on TOF 5000-6000 ns
```yaml
func:
  name: filter.a_gatedOn_b
  args1: ['hit_l:y', 'hit_l:t']
  args2: [5000, 6000]
```

</details>

---

## Troubleshooting

<details>
<summary><strong>Common Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| "Unknown plot type" | Check `type:` is valid: `multiline`, `hist1d`, `hist2d`, `rollavg`, etc. |
| Plot not appearing | Verify variable names, check YAML syntax, try `nacc: 1` |
| "Variable not found" | Check format `detector:variable`, verify spelling |
| Empty histogram | Check `arange` covers data range, verify gate conditions |
| MPI errors | Ensure CONFIGDIR accessible on all nodes |

</details>

---

## Additional Resources

- Detector parameters: `dream/config/dream/det.yaml`
- Algorithm definitions: `dream/config/dream/alg.yaml`
