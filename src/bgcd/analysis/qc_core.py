from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class QCGapsFillReport:
    per_var: pd.DataFrame  # var, n_missing_before, n_filled, n_missing_after, limit, max_fill_h
    warnings: list[str]


@dataclass
class QCCoverageReport:
    per_var: pd.DataFrame
    warnings: list[str]


@dataclass
class QCDuplicatesReport:
    n_rows_in: int
    n_dup_rows: int
    n_groups_with_dups: int
    strategy: str
    n_rows_out: int
    warnings: list[str]


@dataclass
class QCTimeReport:
    n_rows_in: int
    n_bad_time: int
    n_rows_out: int
    dt_median_h: float | None
    dt_p10_h: float | None
    dt_p90_h: float | None
    n_gaps_gt_warn: int
    max_gap_h: float | None
    warnings: list[str]


@dataclass
class QCBoundsReport:
    n_rows_in: int
    n_rows_out: int
    per_var: pd.DataFrame  # var, n_total, n_oob, frac_oob, min, max
    warnings: list[str]


def qc_time(
    df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    expected_dt_hours: float = 3.0,
    dt_tolerance_hours: float = 0.75,
    gap_warn_hours: float = 12.0,
) -> Tuple[pd.DataFrame, QCTimeReport]:
    """
    Step 1 QC: time parsing + NaT drop + dt/gap diagnostics.
    Does NOT resample or interpolate.

    Returns:
      df_out: copy with time_col parsed as datetime UTC and NaT rows removed
      report: QCTimeReport with dt stats and warnings
    """
    df0 = df.copy()
    n_in = len(df0)

    df0[time_col] = pd.to_datetime(df0[time_col], errors="coerce", utc=True)
    n_bad = int(df0[time_col].isna().sum())

    warnings: list[str] = []
    if n_bad > 0:
        warnings.append(f"[QC][time] {n_bad} rows have invalid {time_col} and were dropped.")

    df1 = df0.loc[df0[time_col].notna()].copy()
    df1 = df1.sort_values(time_col)

    dt_median_h = None
    dt_p10_h = None
    dt_p90_h = None
    n_gaps = 0
    max_gap_h = None

    if len(df1) >= 2:
        t = df1[time_col].values.astype("datetime64[ns]")
        dt = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0

        dt_median_h = float(np.nanmedian(dt))
        dt_p10_h = float(np.nanpercentile(dt, 10))
        dt_p90_h = float(np.nanpercentile(dt, 90))
        max_gap_h = float(np.nanmax(dt)) if dt.size else None

        if np.isfinite(dt_median_h) and abs(dt_median_h - expected_dt_hours) > dt_tolerance_hours:
            warnings.append(
                f"[QC][time] median dt = {dt_median_h:.2f}h (expected {expected_dt_hours:.2f}±{dt_tolerance_hours:.2f}h)."
            )

        gaps = dt[dt > gap_warn_hours]
        n_gaps = int(gaps.size)
        if n_gaps > 0:
            warnings.append(f"[QC][time] detected {n_gaps} gaps > {gap_warn_hours:.1f}h (max gap {max_gap_h:.1f}h).")

    rep = QCTimeReport(
        n_rows_in=n_in,
        n_bad_time=n_bad,
        n_rows_out=len(df1),
        dt_median_h=dt_median_h,
        dt_p10_h=dt_p10_h,
        dt_p90_h=dt_p90_h,
        n_gaps_gt_warn=n_gaps,
        max_gap_h=max_gap_h,
        warnings=warnings,
    )
    return df1, rep

def qc_duplicates(
    df: pd.DataFrame,
    *,
    group_col: str = "platform_id",
    time_col: str = "time_utc",
    strategy: str = "median",
) -> Tuple[pd.DataFrame, QCDuplicatesReport]:
    """
    Step 2 QC: handle duplicate (group_col, time_col) timestamps.

    strategy:
      - 'median' (recommended): median for numeric (non-bool), first for others
      - 'first' / 'last': first/last for numeric, first for others

    Returns:
      df_out: dataframe without duplicate keys
      report: QCDuplicatesReport
    """
    df0 = df.copy()
    n_in = len(df0)
    warnings: list[str] = []

    strategy = str(strategy).lower()
    if strategy not in ("median", "first", "last"):
        warnings.append(f"[QC][dups] invalid strategy={strategy!r}; falling back to 'median'.")
        strategy = "median"

    # detect duplicates
    dup_mask = df0.duplicated(subset=[group_col, time_col], keep=False)
    n_dup_rows = int(dup_mask.sum())

    if n_dup_rows == 0:
        rep = QCDuplicatesReport(
            n_rows_in=n_in,
            n_dup_rows=0,
            n_groups_with_dups=0,
            strategy=strategy,
            n_rows_out=n_in,
            warnings=[],
        )
        return df0, rep

    # count how many duplicate groups (unique keys that appear >1)
    key_counts = df0.loc[dup_mask, [group_col, time_col]].value_counts()
    n_groups_with_dups = int((key_counts > 1).sum())

    warnings.append(
        f"[QC][dups] found {n_dup_rows} duplicate rows across {n_groups_with_dups} duplicated (platform_id,time) keys. "
        f"Applying strategy={strategy!r}."
    )

    # Build aggregation spec
    cols = list(df0.columns)

    # numeric but NOT boolean
    num_cols = [
        c for c in cols
        if c not in (group_col, time_col)
        and pd.api.types.is_numeric_dtype(df0[c])
        and not pd.api.types.is_bool_dtype(df0[c])
    ]
    other_cols = [c for c in cols if c not in (group_col, time_col) and c not in num_cols]

    agg: Dict[str, Any] = {}

    if strategy == "median":
        for c in num_cols:
            agg[c] = "median"
    else:
        for c in num_cols:
            agg[c] = strategy

    for c in other_cols:
        agg[c] = "first"

    df_out = (
        df0.sort_values(time_col)
        .groupby([group_col, time_col], as_index=False)
        .agg(agg)
    )

    # sanity: no duplicates remain
    dup_after = int(df_out.duplicated(subset=[group_col, time_col], keep=False).sum())
    if dup_after > 0:
        warnings.append(f"[QC][dups] WARNING: still have {dup_after} duplicate rows after aggregation (unexpected).")

    rep = QCDuplicatesReport(
        n_rows_in=n_in,
        n_dup_rows=n_dup_rows,
        n_groups_with_dups=n_groups_with_dups,
        strategy=strategy,
        n_rows_out=len(df_out),
        warnings=warnings,
    )
    return df_out, rep

def qc_bounds(
    df: pd.DataFrame,
    *,
    bounds_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, QCBoundsReport]:
    """
    Step 3 QC: hard physical bounds masking.

    For each variable with bounds:
      - count valid points
      - mask out-of-bounds to NaN

    No interpolation, no clipping.
    """
    df0 = df.copy()
    n_in = len(df0)
    warnings: list[str] = []
    rows = []

    for var, b in (bounds_cfg or {}).items():
        if var not in df0.columns:
            continue

        s = pd.to_numeric(df0[var], errors="coerce")
        ok = s.notna()
        n_total = int(ok.sum())

        vmin = b.get("min", None)
        vmax = b.get("max", None)

        if n_total == 0:
            rows.append(dict(var=var, n_total=0, n_oob=0, frac_oob=np.nan, min=vmin, max=vmax))
            continue

        mask = pd.Series(False, index=df0.index)
        if vmin is not None:
            mask |= (s < float(vmin)) & ok
        if vmax is not None:
            mask |= (s > float(vmax)) & ok

        n_oob = int(mask.sum())
        frac = (n_oob / n_total) if n_total else np.nan
        rows.append(dict(var=var, n_total=n_total, n_oob=n_oob, frac_oob=frac, min=vmin, max=vmax))

        if n_oob > 0:
            warnings.append(f"[QC][bounds] {var}: masked {n_oob}/{n_total} outside [{vmin}, {vmax}].")
            df0.loc[mask, var] = np.nan

    rep = QCBoundsReport(
        n_rows_in=n_in,
        n_rows_out=len(df0),
        per_var=pd.DataFrame(rows),
        warnings=warnings,
    )
    return df0, rep

def qc_coverage(
    df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    vars_to_check: List[str] | None = None,
    warn_frac_valid_below: float = 0.5,
) -> Tuple[QCCoverageReport, list[str]]:
    """
    Step 4 QC: coverage diagnostics (no masking).

    Computes per-variable availability and missing-run structure.
    Works with irregular sampling; missing-run durations are computed from time deltas.

    Returns:
      report: QCCoverageReport(per_var table, warnings)
      default_vars: list of variables actually used (for traceability)
    """
    warnings: list[str] = []

    if time_col not in df.columns:
        raise ValueError(f"Missing {time_col!r} column for qc_coverage().")

    t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    ok_t = t.notna()
    if not ok_t.any():
        raise ValueError("All timestamps are NaT; cannot compute coverage.")

    df0 = df.loc[ok_t].copy()
    df0[time_col] = t.loc[ok_t]
    df0 = df0.sort_values(time_col)

    if vars_to_check is None:
        # default: all numeric (excluding time_col)
        vars_to_check = [
            c for c in df0.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df0[c])
        ]

    rows = []
    for var in vars_to_check:
        if var not in df0.columns:
            continue

        s = df0[var]
        n = len(df0)
        n_valid = int(s.notna().sum())
        frac_valid = (n_valid / n) if n else np.nan

        if n_valid == 0:
            rows.append(
                dict(
                    var=var,
                    n=n,
                    n_valid=0,
                    frac_valid=frac_valid,
                    first_valid_time=pd.NaT,
                    last_valid_time=pd.NaT,
                    n_valid_runs=0,
                    longest_missing_run_h=np.nan,
                )
            )
            warnings.append(f"[QC][coverage] {var}: 0 valid points (sensor absent/broken?).")
            continue

        # first/last valid
        first_valid_time = df0.loc[s.notna(), time_col].iloc[0]
        last_valid_time = df0.loc[s.notna(), time_col].iloc[-1]

        # runs of validity (boolean)
        is_valid = s.notna().to_numpy()
        # count valid runs: transitions False->True
        n_valid_runs = int(np.sum((~is_valid[:-1]) & (is_valid[1:])) + (1 if is_valid[0] else 0))

        # longest missing run duration in hours (based on time deltas)
        # We estimate run durations by summing dt over consecutive missing intervals.
        tt = df0[time_col].to_numpy(dtype="datetime64[ns]")
        dt_h = np.diff(tt).astype("timedelta64[s]").astype(float) / 3600.0

        longest = 0.0
        cur = 0.0
        # missing at index i means point i is missing; gap to i+1 uses dt_h[i]
        for i in range(len(dt_h)):
            if not is_valid[i] and not is_valid[i + 1]:
                cur += dt_h[i]
            else:
                longest = max(longest, cur)
                cur = 0.0
        longest = max(longest, cur)
        longest_missing_run_h = float(longest) if np.isfinite(longest) else np.nan

        if np.isfinite(frac_valid) and frac_valid < warn_frac_valid_below:
            warnings.append(f"[QC][coverage] {var}: low coverage frac_valid={frac_valid:.2f} (<{warn_frac_valid_below}).")

        rows.append(
            dict(
                var=var,
                n=n,
                n_valid=n_valid,
                frac_valid=frac_valid,
                first_valid_time=first_valid_time,
                last_valid_time=last_valid_time,
                n_valid_runs=n_valid_runs,
                longest_missing_run_h=longest_missing_run_h,
            )
        )

    rep = QCCoverageReport(per_var=pd.DataFrame(rows), warnings=warnings)
    return rep, vars_to_check



def qc_gap_fill(
    df: pd.DataFrame,
    *,
    time_col: str = "time_utc",
    vars_to_fill: List[str],
    max_fill_h: float = 0.0,
    method: str = "time",
    limit_area: str = "inside",
) -> Tuple[pd.DataFrame, QCGapsFillReport]:
    """
    Optional QC step: fill short NaN gaps for selected variables (after bounds masking).

    Idea:
      - In merged multi-sensor tables, single-row NaNs can fragment long analysis windows.
      - We fill only *short* internal NaN runs using time-based interpolation.
      - Long gaps are NOT filled (interpolate limit prevents it), and will still break windows if required.

    How "short" is defined:
      - compute dt_median_h from sorted time axis
      - limit = floor(max_fill_h / dt_median_h)
      - pandas interpolate(..., limit=limit, limit_area='inside')

    Returns:
      df_out: copy with filled values
      report: QCGapsFillReport
    """
    df0 = df.copy()
    warnings: list[str] = []

    if time_col not in df0.columns:
        raise ValueError(f"Missing {time_col!r} column for gap filling.")

    if not vars_to_fill:
        rep = QCGapsFillReport(
            per_var=pd.DataFrame(columns=["var", "n_missing_before", "n_filled", "n_missing_after", "limit", "max_fill_h"]),
            warnings=[],
        )
        return df0, rep

    # Ensure datetime UTC and sorted
    df0[time_col] = pd.to_datetime(df0[time_col], errors="coerce", utc=True)
    df0 = df0.loc[df0[time_col].notna()].sort_values(time_col).copy()

    # median dt (hours)
    dt_median_h = None
    if len(df0) >= 2:
        t = df0[time_col].values.astype("datetime64[ns]")
        dt = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
        if dt.size:
            dt_median_h = float(np.nanmedian(dt))

    max_fill_h = float(max_fill_h)
    if max_fill_h <= 0:
        # disabled, but keep a small report for traceability
        rows = []
        for v in vars_to_fill:
            if v not in df0.columns:
                continue
            s = pd.to_numeric(df0[v], errors="coerce")
            rows.append(
                {"var": v, "n_missing_before": int(s.isna().sum()), "n_filled": 0, "n_missing_after": int(s.isna().sum()),
                 "limit": 0, "max_fill_h": max_fill_h}
            )
        return df0, QCGapsFillReport(per_var=pd.DataFrame(rows), warnings=[])

    if dt_median_h is None or not np.isfinite(dt_median_h) or dt_median_h <= 0:
        warnings.append("[QC][gap_fill] Cannot compute a finite median dt; skipping gap filling.")
        rep = QCGapsFillReport(
            per_var=pd.DataFrame(columns=["var", "n_missing_before", "n_filled", "n_missing_after", "limit", "max_fill_h"]),
            warnings=warnings,
        )
        return df0, rep

    limit = int(np.floor(max_fill_h / dt_median_h + 1e-9))
    if limit <= 0:
        warnings.append(
            f"[QC][gap_fill] max_fill_h={max_fill_h:.2f}h is smaller than median dt={dt_median_h:.2f}h; nothing to fill."
        )
        rep = QCGapsFillReport(
            per_var=pd.DataFrame(columns=["var", "n_missing_before", "n_filled", "n_missing_after", "limit", "max_fill_h"]),
            warnings=warnings,
        )
        return df0, rep

    rows = []
    t_index = df0[time_col]

    for v in vars_to_fill:
        if v not in df0.columns:
            warnings.append(f"[QC][gap_fill] variable {v!r} not in dataframe; skipped.")
            continue

        if not pd.api.types.is_numeric_dtype(df0[v]):
            warnings.append(f"[QC][gap_fill] variable {v!r} is not numeric; skipped.")
            continue

        s = pd.to_numeric(df0[v], errors="coerce")
        n_before = int(s.isna().sum())

        if n_before == 0:
            rows.append({"var": v, "n_missing_before": 0, "n_filled": 0, "n_missing_after": 0, "limit": limit, "max_fill_h": max_fill_h})
            continue

        s2 = pd.Series(s.values, index=t_index.values)
        try:
            s_fill = s2.interpolate(method=method, limit=limit, limit_area=limit_area)
        except Exception as e:
            warnings.append(f"[QC][gap_fill] {v!r} interpolate failed ({type(e).__name__}: {e}); skipped.")
            rows.append({"var": v, "n_missing_before": n_before, "n_filled": 0, "n_missing_after": n_before, "limit": limit, "max_fill_h": max_fill_h})
            continue

        n_after = int(pd.isna(s_fill.values).sum())
        n_filled = n_before - n_after

        df0[v] = s_fill.values
        rows.append({"var": v, "n_missing_before": n_before, "n_filled": n_filled, "n_missing_after": n_after, "limit": limit, "max_fill_h": max_fill_h})

    rep = QCGapsFillReport(per_var=pd.DataFrame(rows), warnings=warnings)
    if not rep.per_var.empty:
        tot_filled = int(rep.per_var["n_filled"].sum())
        if tot_filled > 0:
            rep.warnings = rep.warnings + [
                f"[QC][gap_fill] filled {tot_filled} NaNs across {len(rep.per_var)} vars (limit={limit} samples, max_fill_h={max_fill_h:.1f}h)."
            ]
    return df0, rep