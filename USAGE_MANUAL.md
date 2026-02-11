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
```

## What it does

- Splits multi-header raw CSV
- Groups by platform
- Normalizes schema
- Converts timestamps
- Cleans numeric fields

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

## Single Platform

```bash
python -m bgcd.cli_master \
  --platform-id 300534065378180 \
  --drifter-db-dir "PATH/db_drifter" \
  --wind-db-dir "PATH/db_wind" \
  --mat-db-dir "PATH/db_mat" \
  --output-dir "PATH/db_master" \
  --format csv
```

---

## Multiple Platforms

```bash
python -m bgcd.cli_master \
  --platform-id 300534065378180 \
  --platform-id 300534065379230 \
  --platform-id 300534065470010 \
  --drifter-db-dir "PATH/db_drifter" \
  --wind-db-dir "PATH/db_wind" \
  --mat-db-dir "PATH/db_mat" \
  --output-dir "PATH/db_master"
```

---

## From File List

Create `platforms.txt`:

```
300534065378180
300534065379230
300534065470010
```

Then:

```bash
python -m bgcd.cli_master \
  --platform-ids-file platforms.txt \
  --drifter-db-dir "PATH/db_drifter" \
  --wind-db-dir "PATH/db_wind" \
  --mat-db-dir "PATH/db_mat" \
  --output-dir "PATH/db_master"
```

---

# ⚙️ Important Options

## Timeline Mode

Default (recommended for Lagrangian analysis):

```
--target-time drifter
```

Alternative (regular hourly grid):

```
--target-time hourly
```

---

## Merge Tolerances

Default:

```
--wind-tolerance 30min
--mat-tolerance 30min
```

Increase if necessary:

```
--wind-tolerance 1H
--mat-tolerance 1H
```

---

## Output Modes

Per platform (default):

```
--mode per-platform
```

Single merged file:

```
--mode single
```

---

## Include Time-Difference Diagnostics

```bash
--with-dt
```

Adds:

- time_wind
- dt_wind_min
- time_mat
- dt_mat_min

Useful for scientific validation.

---

# 📊 Expected Output Structure

Example MASTER columns:

```
platform_id
time_utc
lat
lon
sst_c
slp_mb
[winds fields...]
u
v
vorticity
strain
```

Missing values (`NaN`) are expected when:

- Data are outside coverage period
- No match found within tolerance

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

