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
