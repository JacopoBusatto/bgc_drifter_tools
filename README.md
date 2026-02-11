# BGC Drifter Tools

**BGC Drifter Tools** is a Python package to preprocess, organize, and merge biogeochemical surface drifter data (BGC-SVP and similar platforms).

It supports:

- Drifter observational data (CSV)
- Wind statistics (CSV)
- Eulerian diagnostics sampled along trajectories (MATLAB `.mat`: vorticity, strain rate, velocity)
- Modular merging into per-platform MASTER datasets

The package is designed for **real-world operational datasets**, including:

- CSV files with repeated headers
- Mixed schemas across platforms
- Malformed separators and junk tokens
- Platform-dependent variable availability

---

# Key Features

- Robust multi-header CSV parsing
- Automatic split into per-platform raw chunks
- Canonical normalization to a stable schema
- Native MATLAB `.mat` support (no MATLAB required)
- Fully CLI-based workflow
- Modular and extensible architecture

---

# Repository Structure

```
bgc_drifter_tools/
│
├── src/bgcd/
│   ├── io.py                  # Canonicalization and merge utilities
│   ├── raw_split.py           # Multi-header CSV splitter
│   ├── master.py              # Master dataset builder
│   ├── cli_prepare_drifter.py
│   ├── cli_prepare_wind.py
│   ├── cli_matdb.py
│   └── cli_master.py
│
├── README.md
├── pyproject.toml
└── .gitignore
```

All operational steps are performed via CLI modules under `bgcd.cli_*`.

---

# Installation

## 1. Create virtual environment

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

If you want Parquet support:

```bash
pip install pyarrow
```

---

# Full Operational Workflow

You need:

```
drifter_data.csv
wind_data.csv
timeseries/
    a<platform_id>_vort.mat
```

---

# 1️⃣ Prepare Drifter Database

```bash
python -m bgcd.cli_prepare_drifter \
  --input-file "PATH/drifter_data.csv" \
  --output-dir "PATH/db_drifter" \
  --format csv
```

Output:

```
db_drifter/
    drifter_<platform_id>.csv
```

---

# 2️⃣ Prepare Wind Database

```bash
python -m bgcd.cli_prepare_wind \
  --input-file "PATH/wind_data.csv" \
  --output-dir "PATH/db_wind" \
  --format csv
```

Output:

```
db_wind/
    wind_<platform_id>.csv
```

---

# 3️⃣ Prepare MAT (Eulerian) Database

```bash
python -m bgcd.cli_matdb \
  --input-dir "PATH/timeseries" \
  --output-dir "PATH/db_mat" \
  --format csv \
  --mode per-platform
```

Output:

```
db_mat/
    mat_timeseries_<platform_id>.csv
```

Extracted variables:

- platform_id
- time_utc
- lat, lon
- u, v
- vorticity
- strain

---

# 4️⃣ Build MASTER Dataset

## Single platform

```bash
python -m bgcd.cli_master \
  --platform-id 300534065378180 \
  --drifter-db-dir "PATH/db_drifter" \
  --wind-db-dir "PATH/db_wind" \
  --mat-db-dir "PATH/db_mat" \
  --output-dir "PATH/db_master"
```

## Multiple platforms

```bash
python -m bgcd.cli_master \
  --platform-id 300534065378180 \
  --platform-id 300534065379230 \
  --drifter-db-dir "PATH/db_drifter" \
  --wind-db-dir "PATH/db_wind" \
  --mat-db-dir "PATH/db_mat" \
  --output-dir "PATH/db_master"
```

---

# MASTER Temporal Strategy

Default:

```
--target-time drifter
```

- Drifter defines timeline
- Wind and MAT attached via nearest-time merge
- No interpolation
- Missing values preserved

Alternative:

```
--target-time hourly
```

Creates a regular hourly grid.

---

# Merge Tolerances

Default:

```
--wind-tolerance 30min
--mat-tolerance 30min
```

If no match within tolerance → values remain `NaN`.

This is expected behavior.

---

# Output Modes

Per-platform (default):

```
--mode per-platform
```

Creates:

```
master_<platform_id>.csv
```

Single merged file:

```
--mode single
```

Creates:

```
master_all.csv
```

---

# Canonical Data Model

## Drifter (canonical)

Required:
- platform_id
- time_utc
- lat
- lon

Optional:
- sst_c
- slp_mb
- salinity_psu
- sst_sbe_c
- wind_speed_ms
- wind_dir_deg
- battery_v
- drogue_counts

## Wind (canonical)

Required:
- platform_id
- time_utc

Optional:
- wspd*
- wdir*
- samples

## MAT timeseries

- platform_id
- time_utc
- lat
- lon
- u
- v
- vorticity
- strain

---

# Design Philosophy

- Raw data are split before normalization
- Canonical per-platform databases ensure modularity
- No implicit interpolation
- Merge tolerances are explicit
- Missing values remain visible
- Fully reproducible pipeline

---

# Extensibility

New biological datasets (e.g. oxygen, chlorophyll) should:

1. Be converted into canonical per-platform files:
   ```
   oxygen_<platform_id>.csv
   ```
2. Follow same schema rules:
   - platform_id
   - time_utc
3. Be merged via `cli_master`

The architecture is designed for future sensor expansion.

---

# License

To be defined.

---

# Contributors

## Software Development and Design

- **Jacopo Busatto**  
  CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  
  Concept, architecture, implementation, and maintenance of the BGC Drifter Tools package.

---

## Data Production and Field Operations

- **Marco Bellacicco**  
  CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  

- **Jacopo Busatto**  
  CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  

- **Zoi Kokkinis**  
  CNR – Institute of Marine Sciences (ISMAR), Lerici, Italy  

- **Milena Menna**  
  OGS – National Institute of Oceanography and Applied Geophysics, Trieste, Italy  

Field deployment, sensor integration, and operational data production for BGC-SVP platforms.

---

Contributions, issue reports, and scientific collaborations are welcome.

# Contact

For questions, collaborations, or technical support regarding this repository:

**Jacopo Busatto**  
CNR – Institute of Marine Sciences (ISMAR), Rome, Italy  
📧 jacopobusatto@cnr.it  

Please open a GitHub Issue for bug reports or feature requests whenever possible.
