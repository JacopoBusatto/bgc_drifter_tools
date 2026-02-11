# BGC-Drifter Tools – RUN GUIDE

This document describes the complete operational workflow to prepare and merge:

- Drifter data
- Wind data
- Eulerian (MAT) vorticity/strain data
- MASTER dataset

The entire pipeline is CLI-based. No example scripts are required.

---

# 0️⃣ First-Time Setup

## 1. Clone the repository

```bash
git clone https://github.com/JacopoBusatto/bgc_drifter_tools.git
cd bgc_drifter_tools
```

## 2. Create virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install package (editable mode)

```bash
pip install -e .
```

## 4. (Optional) Install visualization extras (plots)

If you want to generate diagnostic plots (time series, Cartopy maps, wind rose):

```bash
pip install -e ".[viz]"
```

If Cartopy fails to install via pip, use conda-forge:

```bash
mamba install -c conda-forge cartopy proj geos pyproj
```
---

# 📁 Expected Input Files

You should have:

```
drifter_data.csv
wind_data.csv
timeseries/
    a<platform_id>_vort.mat
```

---

# 1️⃣ Prepare Drifter Database

## Command

```bash
python -m bgcd.cli_prepare_drifter \
  --input-file "PATH/drifter_data.csv" \
  --output-dir "PATH/db_drifter" \
  --format csv
  --no-kinematics
```

## What it does

- Splits multi-header raw CSV
- Groups by platform
- Normalizes schema
- Converts timestamps
- Cleans numeric fields

Additionally, this step computes:

- Lagrangian velocity (u_lag_ms, v_lag_ms)
- Lagrangian acceleration (ax_lag_ms2, ay_lag_ms2)
- Rotation index (rotation_index)

If --no-kinematics is omitted.
These are derived directly from successive trajectory positions using finite differences.


## Output

```
db_drifter/
    drifter_<platform_id>.csv
```

---

# 2️⃣ Prepare Wind Database

## Command

```bash
python -m bgcd.cli_prepare_wind \
  --input-file "PATH/wind_data.csv" \
  --output-dir "PATH/db_wind" \
  --format csv
```

## What it does

- Splits repeated headers
- Removes corrupted rows
- Fixes known typos
- Normalizes wind statistics schema

## Output

```
db_wind/
    wind_<platform_id>.csv
```

---

# 3️⃣ Prepare MAT (Eulerian) Database

## Command

```bash
python -m bgcd.cli_matdb \
  --input-dir "PATH/timeseries" \
  --output-dir "PATH/db_mat" \
  --format csv \
  --mode per-platform
```

## What it does

- Reads MATLAB `.mat` files
- Extracts:
  - lat
  - lon
  - time
  - u
  - v
  - vorticity
  - strain
- Converts MATLAB time to UTC datetime
- Writes canonical per-platform dataset

## Output

```
db_mat/
    mat_timeseries_<platform_id>.csv
```

---

# 4️⃣ Build MASTER Dataset

# MASTER Dataset Construction

The MASTER table merges:

- Drifter canonical data
- Wind canonical data
- Eulerian MAT diagnostics

---

## Basic Command

```powershell
python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --drifter-db-dir "C:/.../db_drifter" `
  --wind-db-dir    "C:/.../db_wind" `
  --mat-db-dir     "C:/.../db_mat" `
  --output-dir     "C:/.../db_master" `
  --format csv `
  --target-time drifter `
  --mode per-platform
```

---

## Filtering Options

### Geographic Filter (default ON)

Keeps only Mediterranean bounding box.

Disable with:

```powershell
--no-bbox-filter
```

---

### Continuous Segment Filter (optional)

Keeps only the largest continuous temporal segment:

```powershell
--segment-filter
```

Control maximum allowed gap:

```powershell
--segment-max-gap 7D
```

Example full command:

```powershell
python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --drifter-db-dir "C:/.../db_drifter" `
  --wind-db-dir    "C:/.../db_wind" `
  --mat-db-dir     "C:/.../db_mat" `
  --output-dir     "C:/.../db_master" `
  --format csv `
  --target-time drifter `
  --mode per-platform `
  --segment-filter `
  --segment-max-gap 7D
```

---

## Temporal Strategy

### `--target-time drifter` (default)

- MASTER timeline = drifter timestamps
- Wind + MAT attached via nearest-time merge
- No interpolation
- Missing values remain explicit

### `--target-time hourly`

- MASTER timeline = hourly
- MAT defines reference span
- Drifter + wind attached via nearest-time merge

---

## Merge Tolerances

Defaults:

```
--wind-tolerance 30min
--mat-tolerance 30min
```

You may increase if needed:

```
--mat-tolerance 2H
```

---

## Output Modes

### Per Platform (recommended)

```
--mode per-platform
```

Output:
```
master_<platform_id>.csv
```

### Single File

```
--mode single
```

Output:
```
master_all.csv
```

---

## Scientific Notes

- No interpolation is performed by default.
- Missing values are preserved.
- Filters are applied before merging.
- MASTER remains physically consistent with Lagrangian trajectory.

---


# 🔄 Full Pipeline Summary

```
Raw drifter_data.csv
    ↓ cli_prepare_drifter
db_drifter/

Raw wind_data.csv
    ↓ cli_prepare_wind
db_wind/

MAT timeseries/
    ↓ cli_matdb
db_mat/

db_drifter + db_wind + db_mat
    ↓ cli_master
db_master/
```

---

# 🧠 Scientific Design Philosophy

- Drifter defines physical trajectory
- Eulerian quantities sampled onto trajectory
- No implicit interpolation
- Merge tolerances explicit
- Missing values preserved
- Fully modular for future sensors (e.g., oxygen)

---

# 🛠 Troubleshooting

## ImportError after adding new CLI

Run:

```bash
pip install -e .
```

## Parquet not working

Install:

```bash
pip install pyarrow
```

## Virtual environment not active

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# ✅ Recommended Workflow for New Campaign

1. Copy raw CSV and MAT files into working directory
2. Run drifter CLI
3. Run wind CLI
4. Run MAT CLI
5. Build MASTER
6. Archive db_master output

