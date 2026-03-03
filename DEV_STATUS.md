# DEV_STATUS – bgc_drifter_tools (bgcd)

This document is a lightweight developer logbook + current status snapshot for the repository.
Goal: keep it easy to update by anyone working on the code.

How to update:
- Add one entry to the Logbook (date + short bullets)
- If you completed something, tick it in the Current Status checklist
- If you discovered an issue, add it to Known Issues (with a short reproduction note)

---

## Quick links

- Repo: https://github.com/JacopoBusatto/bgc_drifter_tools
- User documentation:
  - README.md
  - USAGE_MANUAL.md

---

## Current status (snapshot)

### Core pipeline (canonical DBs + MASTER)

- [x] Raw drifter CSV multi-header splitter (per-platform chunks)
- [x] Drifter canonicalization (types, UTC, cleaning)
- [x] Optional Lagrangian kinematics during drifter preparation
  - velocities, accelerations
  - rotation index
  - curvature (signed + unsigned) with low-speed masking
- [x] Wind canonical DB builder
- [x] MAT DB builder (Eulerian diagnostics)
  - supports MAT v5/v7 (scipy) + v7.3 (h5py)
  - extracts u, v, vorticity, strain
  - optional Okubo–Weiss if present / computable
- [x] MASTER builder (nearest-time merge)
  - explicit tolerances per source
  - optional dt diagnostics columns (time_utc_<source>, dt_<source>_min)
  - optional filters: bbox / longest contiguous segment
- [x] Output formats: csv / parquet (parquet requires pyarrow)

### Extra sensors

- [x] Oxygen DB builder (MAT)
  - selectable columns
  - optional “0 → NaN” cleaning for selected variables
- [x] BBP / CTD / CHL DB builder (raw CSV → canonical)

### Plotting / diagnostics

- [x] plot_master.py
  - per-variable time series (one file per variable)
  - optional smoothing
  - optional hourly anomaly overlay
  - optional diurnal cycles
  - optional trajectory map (cartopy)
  - optional wind rose (windrose)
- [x] plot_med_map.py
  - multi-drifter Mediterranean map
  - start/end markers

### CLI surface

- [x] bgcd.cli_prepare_drifter
- [x] bgcd.cli_prepare_wind
- [x] bgcd.cli_matdb
- [x] bgcd.cli_oxygen
- [x] bgcd.cli_prepare_bbp
- [x] bgcd.cli_master

### Testing

- [ ] Unit tests for IO edge cases (multi-header, junk tokens, mixed schemas)
- [ ] Synthetic data tests for anomaly / smoothing / plotting utilities
- [ ] Regression tests for MASTER merge tolerances and dt diagnostics
- [ ] CI workflow (GitHub Actions): lint + tests

---

## Known issues / tech debt (keep short)

Add items like:
- [ ] Issue title — short reproduction + file(s) involved + suggested fix owner
Examples:
- [ ] Plot preview math rendering depends on viewer (GitHub ok; some previews show raw LaTeX)
- [ ] Cartopy install can fail via pip on some OS (use conda-forge)

---

## Design notes (stable rules)

- Canonical per-platform DBs first, then merge into MASTER (modular pipeline)
- Nearest-time merge only (no hidden interpolation)
- Explicit tolerances (wind/mat/oxygen/bbp each has its own)
- Missing data must remain visible (NaN is expected)
- Filters (bbox / segment) applied before merging external sources

---

## Roadmap

### Next major step: Statistical analysis modules

Goal: add a new analysis layer that works directly on MASTER datasets.

Proposed package structure:
- src/bgcd/analysis/
  - __init__.py
  - stats_basic.py        (descriptive stats, distributions, robust summaries)
  - correlations.py       (pairwise correlation, partial corr, lagged corr)
  - anomaly_tools.py      (hourly anomaly, diurnal cycles, seasonal cycles)
  - multivariate.py       (PCA/EOF, clustering, dimensionality reduction)
  - regimes.py            (state classification, persistence, transition matrices)
  - significance.py       (bootstrap, permutation tests, confidence intervals)
  - report.py             (auto-summary tables + figures from a MASTER file)

CLI options (optional):
- bgcd.cli_stats (input MASTER(s) → outputs tables/figures)
- bgcd.cli_pca (quick PCA workflow)

Deliverables:
- [ ] Define analysis API (functions + expected inputs/outputs)
- [ ] Minimal CLI: correlations + PCA on selected variables
- [ ] Add examples in docs (small tutorial with 1–2 MASTER files)
- [ ] Add tests with synthetic datasets

---

## Logbook (append-only)

Format:
- YYYY-MM-DD — @name — short bullets

### 2026-03-03 — (template)
- Added DEV_STATUS.md logbook and status snapshot
- Next: start analysis/ module scaffold

(append new entries above this line)