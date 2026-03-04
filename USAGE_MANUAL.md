# BGC Drifter Tools – USAGE MANUAL (Complete Technical Guide)

This document describes the complete operational workflow to build per-platform MASTER datasets from:

- Drifter observational data
- Wind statistics
- Eulerian MAT diagnostics (u, v, vorticity, strain)
- Optional Oxygen data
- Optional BBP / CTD / Chlorophyll data

The entire pipeline is CLI-based, modular, and fully reproducible.

For conceptual architecture and design philosophy, see README.md.

====================================================================

0) FIRST-TIME SETUP

--------------------------------------------------------------------

Requirements

Mandatory:
- Python 3.10+ (recommended 3.11)
- Git

Optional (only for plotting):
- cartopy
- windrose
- seaborn
- pyarrow (if using parquet output)

Internet connection is required the first time to install dependencies.

--------------------------------------------------------------------

1) Clone the Repository

```bash
git clone https://github.com/JacopoBusatto/bgc_drifter_tools.git
cd bgc_drifter_tools
```

If git is not recognized, install Git first.

--------------------------------------------------------------------

2) Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

If activation fails:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

When active, you should see (.venv) in your terminal prompt.

--------------------------------------------------------------------

3) Install the Package

Installed package name:

bgcd

```bash
pip install -e .
```

Optional plotting extras:

```bash
pip install -e ".[viz]"
```

If Cartopy fails:

```bash
mamba install -c conda-forge cartopy proj geos pyproj
```

====================================================================

RECOMMENDED WORKING DIRECTORY STRUCTURE

```text
WORKDIR/
├── raw/
│   ├── drifter_data.csv
│   ├── wind_data.csv
│   ├── timeseries/
│   │     a<platform_id>_vort.mat
│   ├── oxygen_mat/        (optional)
│   └── bbp_raw/           (optional)
└── db/
    ├── db_drifter/
    ├── db_wind/
    ├── db_mat/
    ├── db_oxygen/
    ├── db_bbp/
    └── db_master/
```

Use simple paths (avoid spaces) when possible.

====================================================================

PIPELINE OVERVIEW

```text
Raw drifter_data.csv
    ↓ cli_prepare_drifter
db_drifter/

Raw wind_data.csv
    ↓ cli_prepare_wind
db_wind/

MAT timeseries/
    ↓ cli_matdb
db_mat/

Optional oxygen_mat/
    ↓ cli_oxygen
db_oxygen/

Optional bbp_raw/
    ↓ cli_prepare_bbp
db_bbp/

All canonical databases
    ↓ cli_master
db_master/
```

====================================================================

STEP 1 – BUILD DRIFTER DATABASE

Command:

```bash
python -m bgcd.cli_prepare_drifter `
  --input-file raw/drifter_data.csv `
  --output-dir db/db_drifter `
  --format csv
```

Disable kinematics:

```bash
--no-kinematics
```

--------------------------------------------------------------------

What This Step Does

- Splits multi-header raw CSV
- Groups data by platform_id
- Normalizes schema into canonical format
- Converts timestamps to UTC
- Cleans numeric fields
- Removes malformed or corrupted rows

Additionally (unless --no-kinematics):

Lagrangian kinematics are computed using finite differences on successive trajectory positions:

- u_lag_ms, v_lag_ms (Lagrangian velocity components)
- ax_lag_ms2, ay_lag_ms2 (Lagrangian acceleration components)
- rotation_index
- curvature_m1
- curvature_signed_m1

These quantities are derived directly from position time series and remain physically consistent with the observed trajectory.

Output:

```text
db_drifter/
    drifter_<platform_id>.csv
```

====================================================================

STEP 2 – BUILD WIND DATABASE

```bash
python -m bgcd.cli_prepare_wind `
  --input-file raw/wind_data.csv `
  --output-dir db/db_wind `
  --format csv
```

What This Step Does

- Splits repeated headers
- Removes corrupted rows
- Fixes known typos
- Normalizes wind statistics schema

Output:

```text
db_wind/
    wind_<platform_id>.csv
```

====================================================================

STEP 3 – BUILD MAT (EULERIAN) DATABASE

```bash
python -m bgcd.cli_matdb `
  --input-dir raw/timeseries `
  --output-dir db/db_mat `
  --format csv `
  --mode per-platform
```

Optional file pattern selection:

```bash
--glob "a300534065378180*.mat"
```

What This Step Does

- Reads MATLAB .mat files
- Supports MAT v5/v7 via scipy
- Supports MAT v7.3 via h5py
- Extracts:
  - lat
  - lon
  - time
  - u
  - v
  - vorticity
  - strain
- Converts MATLAB datenum to UTC datetime
- Writes canonical per-platform dataset

Output:

```text
db_mat/
    mat_timeseries_<platform_id>.csv
```

====================================================================

STEP 4 – OPTIONAL SENSOR DATABASES

--------------------------------------------------------------------

4a – Oxygen

```bash
python -m bgcd.cli_oxygen `
  --input-dir raw/oxygen_mat `
  --output-dir db/db_oxygen `
  --format csv
```

Options:
```bash
--keep-cols`
```
Default columns
```python
DEFAULT_KEEP_COLS = [
    "AirSat_c",
    "CalPhase_c",
    "DO2_H2O_c",
    "DO2_c",
    "Depth_c",
    "Spsu_c",
    "TdegC_c",
    "oxy_comp_mgL_c",
    "oxy_comp_mlL_c",
    "oxy_comp_molar_c",
    "oxy_comp_saturation_c",
]

```

```bash
--zero-is-nan-cols`
```
Convert 0 -> NaN for these columns (if present).
Default
```python
["DO2_c", "oxy_comp_mgL_c"]
```

Output:

```text
db_oxygen/
    oxygen_<platform_id>.csv
```

--------------------------------------------------------------------

4b – BBP / CTD / CHL

```bash
python -m bgcd.cli_prepare_bbp `
  --input-dir raw/bbp_raw `
  --output-dir db/db_bbp `
  --format csv
```

Options:

```bash
--glob`
```
Default
```bash
*.csv
```
Glob pattern for bbp raw files


Output:

```text
db_bbp/
    bbp_<platform_id>.csv
```

====================================================================

STEP 5 – BUILD MASTER DATASET

The MASTER dataset merges:

- Drifter canonical data
- Wind canonical data
- MAT diagnostics
- Optional Oxygen
- Optional BBP

--------------------------------------------------------------------

Basic Command

```bash
python -m bgcd.cli_master `
  --platform-id <platform_id> `
  --drifter-db-dir db/db_drifter `
  --wind-db-dir db/db_wind `
  --mat-db-dir db/db_mat `
  --output-dir db/db_master `
  --format csv
```

--------------------------------------------------------------------

MASTER with Oxygen + BBP + dt diagnostics

```bash
python -m bgcd.cli_master `
  --platform-id <platform_id> `
  --drifter-db-dir db/db_drifter `
  --wind-db-dir db/db_wind `
  --mat-db-dir db/db_mat `
  --oxygen-db-dir db/db_oxygen `
  --bbp-db-dir db/db_bbp `
  --output-dir db/db_master `
  --format csv `
  --with-dt
```

MASTER with Oxygen + BBP + dt diagnostics + gap selection
```bash
python -m bgcd.cli_master `
  --platform-id <platform_id> `
  --drifter-db-dir "db/db_drifter" `
  --wind-db-dir    "db/db_wind" `
  --mat-db-dir     "db/db_mat" `
  --oxygen-db-dir  "db/db_oxygen" `
  --oxygen-cols DO2_c oxy_comp_mgL_c `
  --oxygen-tolerance 30min `
  --bbp-db-dir     "db/db_bbp" `
  --bbp-cols bbp_470_m1 bbp_532_m1 temp_ctd_c chl `
  --bbp-tolerance 30min `
  --output-dir     "db/db_master" `
  --format csv `
  --target-time drifter `
  --mode per-platform `
  --with-dt `
  --segment-filter `
  --segment-max-gap 7D
```
====================================================================

TEMPORAL STRATEGY

--target-time drifter (default)
- MASTER timeline = drifter timestamps
- Wind and MAT attached via nearest-time merge
- No interpolation
- Missing values preserved

--target-time hourly
- MASTER timeline = regular hourly grid
- MAT defines reference span
- Drifter and wind attached via nearest-time merge

====================================================================

MERGE TOLERANCES

Defaults:

--wind-tolerance 30min
--mat-tolerance 30min

You may increase:

--mat-tolerance 2H

If no match within tolerance, values remain NaN.

No interpolation is performed.

====================================================================

OUTPUT MODES

Per-platform (recommended):

--mode per-platform

Produces:

master_<platform_id>.csv

Single merged file:

--mode single

Produces:

master_all.csv

====================================================================

DEPLOYMENT CLEANING FILTERS

Geographic Filter (default ON)

Keeps only Mediterranean bounding box.

Disable:

--no-bbox-filter

Largest Continuous Temporal Segment

--segment-filter
--segment-max-gap 7D

If a gap larger than threshold exists:
- A new segment starts
- Only the longest continuous segment is retained

Filters are applied BEFORE merging external fields.

====================================================================

SCIENTIFIC NOTES

- No interpolation is performed.
- Merge tolerances are explicit.
- Missing values remain visible.
- Filters are applied before merging.
- MASTER remains physically consistent with Lagrangian trajectory.
- Eulerian quantities are sampled onto the Lagrangian trajectory without altering its geometry.
- Kinematic diagnostics are computed from observed positions only.

====================================================================

OPTIONAL PLOTTING

Per-platform diagnostics:

```bash
python src/bgcd/plot_master.py `
  --input db/db_master/master_<platform_id>.csv `
  --outdir plots/300534065378180
```

Complete plot options:
```bash
python src/bgcd/plot_master.py `
   --input "db/db_master/master_<platform_id>.csv" `
   --outdir "plots/<platform_id>" `
   --no-zero-sst `
   --decimate-quiver 7 `
   --rot-smooth 24h `
   --mark-every-days 7 `
   --with-hourly-anomaly `
   --anom-min-samples-per-hour 8 `
   --with-diurnal-cycles `
   --diurnal-binning 3
```

Options:
--no-zero-sst
action="store_true"
Treat SST==0 as NaN (fill-value behavior)

--sst-fill
default="-5.0"
Comma-separated SST fill values to mask (e.g. -5, -999)

--decimate-quiver N
default=10
Keep 1 every N points for wind arrows

--smooth-all
default=""
Time-based rolling mean window applied to ALL time series (e.g. '6H', '30min', '1D'). Empty disables.

--rot-smooth N
default=""
Time-based rolling mean window applied only to rotation_index (overrides --smooth-all for rotation). Empty disables.

--mark-every-days N
default=7
Mark time series and trajectories every N days

--with-hourly-anomaly
action="store_true"
Add hour-of-day anomaly on a right Y axis (y - mean(y|hour)).

--anom-min-samples-per-hour
default=3
Minimum samples per hour-of-day to compute the hour mean (otherwise anomaly is NaN for that hour).

--with-diurnal-cycles
action="store_true"
Write diurnal cycle plots (hour-of-day climatology) for key variables.

--diurnal-binning
default=1
Binning for diurnal cycle (e.g. '1H' for hourly, '3H' for 3-hour bins). Default is '1H' (no binning).




Multi-drifter Mediterranean map:

```bash
python src/bgcd/plot_med_map.py `
  --inputs db/db_master/master_<platform_id>.csv `
  --out plots/med_map.png
```



====================================================================

TROUBLESHOOTING

Reinstall editable mode:

```bash
pip install -e .
```

Install parquet support:

```bash
pip install pyarrow
```

Activate virtual environment if needed:

```powershell
.\.venv\Scripts\Activate.ps1
```

====================================================================

RECOMMENDED WORKFLOW FOR NEW CAMPAIGN

1. Copy raw data into WORKDIR/raw/
2. Run cli_prepare_drifter
3. Run cli_prepare_wind
4. Run cli_matdb
5. Optional: run cli_oxygen and cli_prepare_bbp
6. Build MASTER
7. Optional: generate plots
8. Archive db_master output

## ANALYSIS
#### Workflow
```
MASTER CSV (1 drifter)
   |
   v
[A] QC (opzionale ma consigliato)
   - time checks (dt/gaps)
   - duplicates (median)
   - bounds -> mask NaN
   - coverage report
   |
   |  outputs:
   |   OUT/<pid>/A_qc/
   |     data/qc_clean_master.csv
   |     reports/qc_*.csv
   |     run_log.txt
   v
[B] Window selection (SEMPRE)
   - usa analysis.data_window.required
   - spezza su gap > max_gap_h
   - sceglie la window di durata massima (>= min_points)
   |
   |  outputs:
   |   OUT/<pid>/B_window/
   |     best_window.txt
   |     reports/windows_top.csv
   |     data/master_subset_best_window.csv
   v
[C] Preprocess (opzionale, parallelo)
   - NON sostituisce i raw: crea versioni alternative
   - es: zscore, lowpass, highpass, derivative
   |
   |  outputs:
   |   OUT/<pid>/C_preprocess/
   |     data/raw.csv (o link al subset)
   |     data/lp72h.csv
   |     data/hp.csv (= raw - lp)
   |     reports/preprocess_report.csv
   v
[D] Coupling (v1)
   D1) Pairwise Pearson phys×bio
       - usa qc.pairwise_overlap:
           min_fraction = 0.7
           min_points   = 50
       - se una variabile è troppo NaN -> viene ignorata + warning
       outputs:
         OUT/<pid>/D_coupling/pairwise/
           reports/pairwise_corr.csv
           figures/heatmap.png
           run_log.txt

   D2) Rolling correlation (per coppie scelte o top pairs)
       - window = 48H
       - rolling_min_points = 10
       outputs:
         OUT/<pid>/D_coupling/rolling/
           reports/rolling_corr_x__y.csv
           figures/rolling_corr_x__y.png

   D3) Lagged correlation
       - max_lag_hours = 72
       - lag_match_tolerance_hours = 1.5 (merge_asof/nearest)
       outputs:
         OUT/<pid>/D_coupling/lagged/
           reports/lagged_corr_x__y.csv
           figures/lagged_corr_x__y.png
```

  