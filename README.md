# BGC Drifter Tools

**BGC Drifter Tools** is a Python package to preprocess, organize, and analyze biogeochemical surface drifter data (BGC-SVP and similar platforms), combining:

- drifter observational data (CSV)
- wind data (CSV)
- Eulerian diagnostics along trajectories (MATLAB `.mat` files: vorticity, strain rate, etc.)

The package is designed to handle *real-world operational datasets*, including:
- CSV files with repeated headers and mixed schemas
- malformed separators and junk tokens
- platform-dependent variable availability

The goal is to provide **clean, reproducible, per-platform databases** that can be easily merged and analyzed by scientists.

---

## Key Features

- Robust preprocessing of *multi-header* CSV exports  
- Automatic split into **per-platform raw files**
- Canonical normalization to a **stable schema**
- Native support for MATLAB `.mat` trajectory diagnostics (no MATLAB required)
- CLI utilities for batch processing
- Modular design, suitable for extension and scientific workflows

---

## Repository Structure

```
bgc_drifter_tools/
│
├── src/bgcd/                 # Core library
│   ├── io.py                 # I/O, canonicalization, merging utilities
│   ├── raw_split.py          # Split multi-header CSVs into clean chunks
│   ├── cli_matdb.py          # CLI for building MAT-based databases
│   └── master.py             # (planned) master table construction
│
├── examples/                 # Reproducible pipeline scripts
│   ├── 05_clean_drifter_raw.py
│   ├── 06_normalize_drifter_db.py
│   ├── 07_clean_wind_raw.py
│   ├── 08_normalize_wind_db.py
│   └── 02_inspect_mat.py
│
├── PIPELINE.md               # Step-by-step reproducible workflow
├── README.md                 # This file
├── pyproject.toml
└── .gitignore
```

---

## Installation

### Recommended (editable install)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

> ⚠️ Build artifacts such as `*.egg-info/`, `dist/`, `build/` are ignored and should **not** be committed.

---

## Typical Workflow

A complete, reproducible pipeline is described in detail in [`PIPELINE.md`](PIPELINE.md).

In short:

1. **Split raw CSV containers** (drifter / wind) into per-platform RAW files  
2. **Canonicalize** RAW files into stable per-platform databases  
3. **Read and convert MATLAB `.mat` files** into per-platform CSV/Parquet  
4. *(Next step)* Merge everything into per-platform MASTER tables  

Each step is implemented as a small, explicit script under `examples/`.

---

## Canonical Data Model (overview)

### Drifter (canonical)
Required:
- `platform_id`
- `time_utc`
- `lat`, `lon`

Optional (if available):
- `sst_c`, `slp_mb`
- `salinity_psu`, `sst_sbe_c`
- `wind_speed_ms`, `wind_dir_deg`
- `battery_v`, `drogue_counts`

### Wind (canonical)
Required:
- `platform_id`
- `time_utc`

Optional:
- `wspd`, `wspd_mean`, `wspd_min`, `wspd_max`, `wspd_std`
- `wspd_skewness`, `wspd_kurtosis`
- `wdir`, `wdir_mean`, `wdir_min`, `wdir_max`, `wdir_std`
- `wdir_skewness`, `wdir_kurtosis`
- `samples`

### MAT timeseries
- `platform_id`
- `time_utc`
- `lat`, `lon`
- `u`, `v`
- `vorticity`
- `strain`

---

## Design Philosophy

- **Raw ≠ Canonical**  
  Raw data are preserved per platform before any normalization.

- **Explicit over implicit**  
  Each processing step is a standalone script, easy to inspect and rerun.

- **Scientist-friendly**  
  No hidden magic, no dependency on MATLAB, minimal assumptions on input format.

---

## Roadmap

Planned next steps:
- Construction of per-platform **MASTER tables**
- Choice of target timeline (drifter vs hourly)
- Basic QC flags and diagnostics
- Simple plotting utilities for exploratory analysis

---

## License

To be defined.

---

## Authors / Contributors

Developed within the BGC-SVP data analysis workflow at CNR-ISMAR.  
Contributions and extensions are welcome.

---

## Master Dataset Construction

The MASTER table is built **per platform** by merging three independent data sources:

1. **Drifter observations** (core in-situ measurements)
2. **Wind statistics** (platform-based wind data)
3. **Eulerian-derived quantities** (vorticity, strain, velocity from CMEMS fields sampled along trajectory)

The merge is performed using `merge_asof` (nearest-time match) with configurable tolerances.

---

## Default Temporal Strategy

By default:

```bash
--target-time drifter
```

This means:

- The MASTER timeline is defined by **drifter observation timestamps**
- Wind and MAT (vorticity/strain) are attached using nearest-time matching
- No interpolation is performed by default
- If no match is found within tolerance → values remain `NaN`

This approach is scientifically consistent for Lagrangian analyses because:
- The drifter defines the physical trajectory
- Eulerian quantities are sampled onto real observation times
- No artificial temporal regularization is introduced

---

## Tolerances

Default tolerances:

- Wind: `30min`
- MAT:  `30min`

Example:

```bash
--wind-tolerance 30min
--mat-tolerance 30min
```

If no data point exists within the specified tolerance, the corresponding fields remain `NaN`.

This is expected and intentional.

---

## Why Missing Values (NaN) Appear

NaN values can occur when:

- The drifter operates outside the temporal coverage of MAT or wind data
- Small temporal mismatches exceed the tolerance
- Gaps exist in the original products

Example diagnostics:

```python
df["vorticity"].isna().mean()
df["wspd"].isna().mean()
```

A small missing fraction (e.g. 3–6%) is normal.

We do **not** artificially fill or interpolate values during merge.

---

## Alternative Temporal Strategy

You can build a regular hourly dataset using:

```bash
--target-time hourly
```

In this case:

- The MASTER timeline is hourly
- MAT defines the reference time span
- Drifter and wind are attached via nearest-time matching

Use this mode when:

- You need regular time grids
- You perform spectral or frequency-domain analysis
- You require compatibility with gridded model products

---

## Output Modes

Two output strategies are available:

### Per-platform (default)

```bash
--mode per-platform
```

Creates one file per platform:

```
master_<platform_id>.csv
```

This is recommended for:

- Platform-specific analysis
- Parallel workflows
- Modular updates

---

### Single merged file

```bash
--mode single
```

Creates:

```
master_all.csv
```

All platforms concatenated into a single table.

---

## Extensibility (Future Biological Variables)

Additional datasets (e.g., oxygen concentration, chlorophyll, bio-optical sensors) should:

1. Be pre-cleaned into canonical per-platform files:
   ```
   oxygen_<platform_id>.csv
   ```
2. Be merged into MASTER using the same nearest-time logic
3. Preserve modularity — no hardcoding in the pipeline

The architecture is intentionally designed to allow new variables without rewriting the core merging logic.

---

## Scientific Philosophy

The MASTER table:

- Preserves original observation times
- Avoids implicit interpolation
- Keeps physical meaning of the trajectory intact
- Maintains transparency via tolerances
- Keeps missing values explicit

This ensures reproducibility and scientific traceability.

---

# 🚀 Quick Start (for non-programmers)

This section explains how to run the full preprocessing pipeline step-by-step,
even if you have never used Python before.

You only need:
- A Windows computer
- Python installed (version 3.10+ recommended)

---

## Step 1 — Install Python (if not already installed)

1. Go to: https://www.python.org/downloads/
2. Download Python 3.10 or newer.
3. During installation:
   ✅ Check **"Add Python to PATH"**
   ✅ Click Install

To verify installation:
Open **PowerShell** and type:

```powershell
python --version
```

You should see something like:

```
Python 3.11.x
```

---

## Step 2 — Download the Repository

If you are using GitHub:

1. Click **Code → Download ZIP**
2. Extract the folder somewhere on your computer
   (e.g. `Documents/GitHub/bgc_drifter_tools`)

Or clone with git:

```powershell
git clone https://github.com/your-repo/bgc_drifter_tools.git
```

---

## Step 3 — Create a Virtual Environment (recommended)

Open PowerShell inside the project folder:

```powershell
cd C:\Path\To\bgc_drifter_tools
```

Then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

You should now see `(.venv)` at the beginning of your terminal line.
This means everything is installed correctly.

---

## Step 4 — Prepare Your Input Files

You should have:

- `drifter_data.csv`
- `wind_data.csv`
- a folder containing `.mat` files

Example structure:

```
DATI_PLATFORMS/
    drifter_data.csv
    wind_data.csv
    timeseries/
        a300534065378180_vort.mat
        a300534065379230_vort.mat
```

---

## Step 5 — Clean Drifter Data

Edit the file:

```
examples/05_clean_drifter_raw.py
```

and make sure the input and output paths match your computer.

Then run:

```powershell
python examples/05_clean_drifter_raw.py
```

This creates:

```
db_drifter_raw/
    drifter_raw_<PLATFORM_ID>.csv
```

---

## Step 6 — Normalize Drifter Data

```powershell
python examples/06_normalize_drifter_db.py
```

This creates clean per-platform datasets:

```
db_drifter/
    drifter_<PLATFORM_ID>.csv
```

These files are ready for analysis.

---

## Step 7 — Clean Wind Data

```powershell
python examples/07_clean_wind_raw.py
```

Then normalize:

```powershell
python examples/08_normalize_wind_db.py
```

You will obtain:

```
db_wind/
    wind_<PLATFORM_ID>.csv
```

---

## Step 8 — Process MATLAB (.mat) Files

To inspect a file:

```powershell
python examples/02_inspect_mat.py
```

To convert all `.mat` files into CSV:

```powershell
python -m bgcd.cli_matdb `
  --input-dir "C:/Path/To/timeseries" `
  --output-dir "C:/Path/To/db_mat" `
  --format csv `
  --mode per-platform
```

You will obtain:

```
db_mat/
    mat_timeseries_<PLATFORM_ID>.csv
```

---

# 🎉 Done!

At this point you have:

- Clean drifter data
- Clean wind data
- Clean Eulerian diagnostics
- All separated by platform
- Ready for merging and analysis

No MATLAB required.
No manual cleaning required.
Fully reproducible.

---

If something fails:
- Make sure paths are correct
- Make sure `.venv` is activated
- Run `pip install -e .` again

---

Next step (advanced):
Building a MASTER dataset that merges everything automatically.
