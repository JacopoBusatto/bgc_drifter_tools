# BGC Drifter Tools — Pipeline (reproducible workflow)

This repository provides a reproducible workflow to ingest and clean BGC-SVP drifter datasets exported as CSV (sometimes containing repeated headers / mixed schemas) and MATLAB `.mat` timeseries (vorticity/strain rate along drifter tracks).

---

## 0) Setup (Windows / PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

**Notes**
- This repo should ignore build artifacts (`*.egg-info/`, `dist/`, `build/`) and virtual envs (`.venv/`).
- If `pip install -e .` creates `src/*.egg-info`, do **not** commit it.

---

## 1) Inputs (local paths)

Typical input locations (example):

- **Drifter raw CSV**  
  `C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\drifter_data.csv`

- **Wind raw CSV**  
  `C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\wind_data.csv`

- **MAT timeseries directory**  
  `C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\timeseries`

---

## 2) Output directories

Suggested outputs:

- Per-platform **RAW drifter chunks**  
  `...\db_drifter_raw\`

- Per-platform **CANONICAL drifter DB**  
  `...\db_drifter\`

- Per-platform **RAW wind chunks**  
  `...\db_wind_raw\`

- Per-platform **CANONICAL wind DB**  
  `...\db_wind\`

- Per-platform **MAT DB (vorticity/strain)**  
  `...\db_mat\`

---

## 3) Drifter CSV — split multi-header container → per-platform RAW

The raw drifter CSV may contain repeated headers and mixed schemas.  
We split it into chunks and save per-platform raw files.

Edit `examples/05_clean_drifter_raw.py` paths if needed, then run:

```powershell
python examples/05_clean_drifter_raw.py
```

Output example:
- `db_drifter_raw\drifter_raw_<PLATFORM_ID>.csv`

---

## 4) Drifter RAW → CANONICAL per-platform DB

**Canonical schema**
- Required: `platform_id, time_utc, lat, lon`
- Optional (if present): `sst_c, slp_mb, salinity_psu, sst_sbe_c, wind_speed_ms, wind_dir_deg, battery_v, drogue_counts`

Run:

```powershell
python examples/06_normalize_drifter_db.py
```

Output example:
- `db_drifter\drifter_<PLATFORM_ID>.csv`

---

## 5) Wind CSV — split multi-header container → per-platform RAW

The raw wind CSV may contain repeated headers and junk columns (e.g. `;;;;;;;`).  
We split it into chunks and save per-platform raw files.

Run:

```powershell
python examples/07_clean_wind_raw.py
```

Output example:
- `db_wind_raw\wind_raw_<PLATFORM_ID>.csv`

---

## 6) Wind RAW → CANONICAL per-platform DB

**Canonical schema**
- Required: `platform_id, time_utc`
- Optional: `wspd, wspd_mean, ..., wdir_kurtosis, samples`

Run:

```powershell
python examples/08_normalize_wind_db.py
```

Output example:
- `db_wind\wind_<PLATFORM_ID>.csv`

---

## 7) MAT timeseries (.mat) → per-platform DB

MAT files are named like:
- `a300534065378180_vort.mat`

### 7.1 Inspect a MAT file (optional but recommended)

```powershell
python examples/02_inspect_mat.py
```

### 7.2 Build per-platform MAT databases

```powershell
python -m bgcd.cli_matdb `
  --input-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/timeseries" `
  --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --format csv `
  --mode per-platform
```

Output example:
- `db_mat\mat_timeseries_<PLATFORM_ID>.csv`

Schema:
- `platform_id, time_utc, lat, lon, u, v, vorticity, strain`

---

## 8) Next step (to be implemented)

Build per-platform **MASTER** tables by merging:
- canonical drifter DB
- canonical wind DB
- MAT timeseries DB

Two target time options will be supported:
- target = drifter timestamps
- target = hourly timeline

(Planned for the next development step.)
