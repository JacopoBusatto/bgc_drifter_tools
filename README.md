# BGC Drifter Tools (`bgcd`)

**BGC Drifter Tools** is a Python package for preprocessing, organizing, and merging biogeochemical surface drifter data (BGC-SVP and similar platforms) into reproducible per-platform MASTER datasets.

The package is designed for real-world operational datasets, including:

- CSV files with repeated headers  
- Mixed schemas across platforms  
- Malformed separators and junk tokens  
- Platform-dependent variable availability  
- MATLAB `.mat` exports (v5, v7, v7.3)  

It supports physical, biogeochemical, and Eulerian diagnostics sampled along Lagrangian trajectories.

---

# Overview

Typical workflow:

1. **Split raw multi-header CSV files**
2. **Canonical normalization** (stable schema, UTC timestamps)
3. **Compute Lagrangian kinematics** (velocity, acceleration, curvature, rotation index)
4. **Ingest MATLAB `.mat` diagnostics** (u, v, vorticity, strain, Okubo–Weiss)
5. **Ingest optional sensors**
   - Oxygen (`.mat`)
   - BBP / CTD / Chlorophyll (`.csv`)
6. **Nearest-time merge into per-platform MASTER datasets**
7. **Optional diagnostic plotting**

The entire workflow is CLI-based and fully reproducible.

---

# Repository Structure

```
bgc_drifter_tools/
├── src/bgcd/
│   ├── __init__.py
│   ├── raw_split.py
│   ├── io.py
│   ├── mat_io.py
│   ├── bbp_io.py
│   ├── master.py
│   ├── cli_prepare_drifter.py
│   ├── cli_prepare_wind.py
│   ├── cli_matdb.py
│   ├── cli_oxygen.py
│   ├── cli_prepare_bbp.py
│   ├── cli_master.py
│   ├── plot_master.py
│   └── plot_med_map.py
├── README.md
├── USAGE_MANUAL.md
├── pyproject.toml
└── LICENSE (MIT)
```

The installed Python package name is:

```
bgcd
```

All commands are executed via:

```
python -m bgcd.cli_*
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/JacopoBusatto/bgc_drifter_tools.git
cd bgc_drifter_tools
```

Create a virtual environment.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional visualization dependencies:

```bash
pip install -e ".[viz]"
```

This installs:

- cartopy  
- seaborn  
- windrose  

If Cartopy installation fails, install via conda-forge:

```
mamba install -c conda-forge cartopy proj geos pyproj
```

---

# MASTER Dataset Construction

The MASTER builder performs a **nearest-time merge** using `pandas.merge_asof` with explicit tolerances.

Default tolerances:

```
--wind-tolerance 30min
--mat-tolerance 30min
```

No interpolation is performed.  
If no match exists within tolerance, values remain `NaN`.

Two timeline modes are supported:

- `--target-time drifter` (default)
- `--target-time hourly`

Optional diagnostic columns:

```
--with-dt
```

This adds:

- `time_utc_<source>`
- `dt_<source>_min`

---

# Spatial and Temporal Filtering

Optional deployment cleaning tools:

### Geographic Filter (Mediterranean bounding box)

Enabled by default to remove pre-deployment or out-of-domain positions.

Disable with:

```
--no-bbox-filter
```

---

### Largest Continuous Temporal Segment

```
--segment-filter
--segment-max-gap 7D
```

If a temporal gap larger than the threshold is detected, a new segment starts.
Only the longest continuous segment is retained.

Filtering is applied **before merging wind and Eulerian fields**.

---

# Canonical Data Model

## Lagrangian Kinematics (Drifter)

Automatically computed during `cli_prepare_drifter`:

- `u_lag_ms`, `v_lag_ms` (m/s)
- `ax_lag_ms2`, `ay_lag_ms2` (m/s²)
- `rotation_index`
- `curvature_m1`
- `curvature_signed_m1`

### Rotation Index

The rotation index is defined as

$$
RI = \frac{\vec{v} \times \vec{a}}{|\vec{v}| |\vec{a}|} \cdot \hat{k}
= \sin(\theta)
$$

where $\theta$ is the angle between velocity and acceleration.

Interpretation:

- RI ≈ 0 → approximately straight motion  
- RI > 0 → counterclockwise rotational tendency  
- RI < 0 → clockwise rotational tendency  
- |RI| ≈ 1 → strong curvature  

---

## Eulerian Diagnostics (MAT)

Extracted from `.mat` files:

- `u`
- `v`
- `vorticity`
- `strain`
- `okubo_weiss = strain² − vorticity²`

MAT v5/v7 are supported via `scipy.io.loadmat`,  
MAT v7.3 via `h5py`.

---

# Diagnostic Plotting (Optional)

### `plot_master.py`

Generates per-platform diagnostics:

- Time series plots
- Optional hourly anomaly overlays
- Optional diurnal cycles
- Cartopy trajectory maps
- Wind rose (if `windrose` installed)
- Data availability summaries

### `plot_med_map.py`

Generates Mediterranean multi-drifter trajectory maps.

See `USAGE_MANUAL.md` for detailed command examples.

---

# Design Philosophy

- Raw data are split before normalization  
- Canonical per-platform databases ensure modularity  
- Merge tolerances are explicit  
- No hidden interpolation  
- Missing values remain visible  
- Fully reproducible CLI workflow  
- Modular architecture for future sensor expansion  

---

# License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

# Contributors

## Software Design & Development

**Jacopo Busatto**  
CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  

## Data Production & Field Operations

Marco Bellacicco (CNR-ISMAR)  
Zoi Kokkinis (CNR-ISMAR)  
Milena Menna (OGS)  

---

# Contact

For scientific collaborations or technical questions:

**Jacopo Busatto**  
CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  
📧 jacopobusatto@cnr.it  

Please open a GitHub Issue for bug reports or feature requests.