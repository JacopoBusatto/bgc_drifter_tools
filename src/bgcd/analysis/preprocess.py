# ================================
# File: src/bgcd/analysis/preprocess.py
# ================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class QCResult:
    """
    Container for QC outputs.

    - summary: per-platform summary table (+ optional __ALL__ row)
    - variable_stats: per-variable QC counts + descriptive stats on cleaned data
    - warnings: list of human-readable warnings
    - flag_columns: list of QC flag column names added to df_clean (optional)
    - outdir: output directory used for writing QC artifacts (if enabled)
    """
    summary: pd.DataFrame
    variable_stats: pd.DataFrame
    warnings: list[str]
    flag_columns: list[str]
    outdir: Optional[Path] = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def apply_qc(
    df: pd.DataFrame,
    config: Dict[str, Any],
    *,
    time_col: str = "time_utc",
    group_col: str = "platform_id",
    outdir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, QCResult]:
    """
    Apply QC to MASTER-like dataframe.

    Philosophy:
    - No interpolation.
    - Hard physical bounds => mask to NaN (not clipping).
    - Derivative-based spike detection for selected variables.
    - Flatline detection => flag only by default.
    - Irregular time supported; rolling operations are time-based.

    Assumption (current workflow):
    - Input MASTER file contains a single platform_id.
      If multiple platform_id are present, reports will be written under qc/__MULTI__/.

    Returns:
        df_clean: copy of df with QC applied (NaNs + optional QC flags)
        qc: QCResult with summary tables and warnings
    """
    df0 = df.copy()

    # Ensure datetime
    df0[time_col] = pd.to_datetime(df0[time_col], errors="coerce", utc=True)

    qc_cfg = (config.get("analysis", {}) or {}).get("qc", {}) or {}
    rep_cfg = (qc_cfg.get("report", {}) or {})
    write_flags = bool(rep_cfg.get("write_flag_columns", True))

    warnings: list[str] = []
    flag_cols: list[str] = []

    # -------------------------------------------------------------------------
    # 1) Time axis checks + handle duplicates (median by default)
    # -------------------------------------------------------------------------
    df1, time_summary, time_warnings = _check_time_axis_and_handle_duplicates(
        df0,
        qc_cfg=qc_cfg,
        time_col=time_col,
        group_col=group_col,
    )
    warnings.extend(time_warnings)

    # -------------------------------------------------------------------------
    # 2) Physical bounds masking (hard NaN)
    # -------------------------------------------------------------------------
    df2, bounds_stats, bounds_warnings, new_flag_cols = _apply_physical_bounds(
        df1,
        qc_cfg=qc_cfg,
        write_flags=write_flags,
    )
    warnings.extend(bounds_warnings)
    flag_cols.extend(new_flag_cols)

    # -------------------------------------------------------------------------
    # 3) Derivative-based spike detection
    # -------------------------------------------------------------------------
    df3, spike_stats, spike_warnings, new_flag_cols = _detect_derivative_spikes(
        df2,
        qc_cfg=qc_cfg,
        time_col=time_col,
        group_col=group_col,
        write_flags=write_flags,
    )
    warnings.extend(spike_warnings)
    flag_cols.extend(new_flag_cols)

    # -------------------------------------------------------------------------
    # 4) Flatline detection (flag-only by default)
    # -------------------------------------------------------------------------
    flat_stats, flat_warnings, new_flag_cols, df4 = _detect_flatline(
        df3,
        qc_cfg=qc_cfg,
        time_col=time_col,
        group_col=group_col,
        write_flags=write_flags,
    )
    warnings.extend(flat_warnings)
    flag_cols.extend(new_flag_cols)

    # -------------------------------------------------------------------------
    # 5) Overlap requirements (stored in summary; enforcement happens downstream)
    # -------------------------------------------------------------------------
    overlap_summary = _compute_overlap_overview(df4, qc_cfg=qc_cfg)

    # -------------------------------------------------------------------------
    # 6) Build report tables
    # -------------------------------------------------------------------------
    summary = _merge_summaries([time_summary, overlap_summary])
    variable_stats = _merge_variable_stats([bounds_stats, spike_stats, flat_stats], df4)

    qc = QCResult(
        summary=summary,
        variable_stats=variable_stats,
        warnings=warnings,
        flag_columns=sorted(set(flag_cols)),
        outdir=None,
    )

    # -------------------------------------------------------------------------
    # 7) Optional writing (SINGLE PLATFORM report folder)
    # -------------------------------------------------------------------------
    if rep_cfg.get("enabled", True) and outdir is not None:
        qc_outname = rep_cfg.get("outdir_name", "qc")
        qc_root = Path(outdir) / qc_outname
        qc_root.mkdir(parents=True, exist_ok=True)

        pid = _infer_single_platform_id(df4, group_col=group_col)
        pid_dir = qc_root / pid
        pid_dir.mkdir(parents=True, exist_ok=True)
        qc.outdir = pid_dir

        if rep_cfg.get("write_summary_csv", True):
            qc.summary.to_csv(pid_dir / "qc_summary.csv", index=False)

        if rep_cfg.get("write_variable_stats_csv", True):
            qc.variable_stats.to_csv(pid_dir / "variable_stats.csv", index=False)

        if rep_cfg.get("write_warnings_txt", True):
            _write_warnings(pid_dir / "warnings.txt", qc.warnings)

    return df4, qc


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
def _infer_single_platform_id(df: pd.DataFrame, *, group_col: str) -> str:
    """
    Infer platform_id for a single-platform MASTER file.

    If multiple platform_ids are present, returns '__MULTI__' so the user notices.
    """
    if group_col not in df.columns:
        return "__UNKNOWN__"

    vals = df[group_col].dropna().unique()
    if len(vals) == 0:
        return "__UNKNOWN__"
    if len(vals) > 1:
        return "__MULTI__"
    return str(vals[0])


def _check_time_axis_and_handle_duplicates(
    df: pd.DataFrame,
    *,
    qc_cfg: Dict[str, Any],
    time_col: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    - Drops invalid timestamps
    - Aggregates duplicate timestamps per platform_id
    - Computes dt stats and gap warnings per platform_id
    """
    time_cfg = qc_cfg.get("time", {}) or {}
    expected_dt_h = float(time_cfg.get("expected_dt_hours", 3.0))
    tol_h = float(time_cfg.get("dt_tolerance_hours", 0.75))
    gap_warn_h = float(time_cfg.get("gap_warn_hours", 12.0))
    dup_strategy = str(time_cfg.get("duplicates_strategy", "median")).lower()

    warnings: list[str] = []

    # Drop rows with invalid time early (can't QC without time)
    n_bad_time = int(df[time_col].isna().sum())
    if n_bad_time > 0:
        warnings.append(f"[QC][time] {n_bad_time} rows have invalid {time_col} and will be dropped.")
        df = df.loc[df[time_col].notna()].copy()

    # Identify duplicates per group
    dup_mask = df.duplicated(subset=[group_col, time_col], keep=False)
    n_dups = int(dup_mask.sum())
    if n_dups > 0:
        warnings.append(
            f"[QC][time] {n_dups} rows belong to duplicate ({group_col}, {time_col}) timestamps. "
            f"Applying duplicates_strategy={dup_strategy!r}."
        )
        df = _aggregate_duplicates(df, group_col=group_col, time_col=time_col, strategy=dup_strategy)

    # Compute dt stats per group
    rows = []
    for pid, g in df.sort_values(time_col).groupby(group_col):
        t = g[time_col].values.astype("datetime64[ns]")
        if len(t) < 2:
            rows.append(
                dict(
                    platform_id=pid,
                    n=len(t),
                    dt_median_h=np.nan,
                    dt_p10_h=np.nan,
                    dt_p90_h=np.nan,
                    n_gaps_gt_warn=0,
                    max_gap_h=np.nan,
                )
            )
            continue

        dt = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
        dt_med = float(np.nanmedian(dt))
        dt_p10 = float(np.nanpercentile(dt, 10))
        dt_p90 = float(np.nanpercentile(dt, 90))
        gaps = dt[dt > gap_warn_h]
        n_gaps = int(gaps.size)
        max_gap = float(np.nanmax(dt)) if dt.size else np.nan

        if np.isfinite(dt_med) and abs(dt_med - expected_dt_h) > tol_h:
            warnings.append(
                f"[QC][time][{pid}] median dt = {dt_med:.2f}h (expected {expected_dt_h:.2f}±{tol_h:.2f}h)."
            )
        if n_gaps > 0:
            warnings.append(
                f"[QC][time][{pid}] detected {n_gaps} gaps > {gap_warn_h:.1f}h (max gap {max_gap:.1f}h)."
            )

        rows.append(
            dict(
                platform_id=pid,
                n=len(t),
                dt_median_h=dt_med,
                dt_p10_h=dt_p10,
                dt_p90_h=dt_p90,
                n_gaps_gt_warn=n_gaps,
                max_gap_h=max_gap,
            )
        )

    time_summary = pd.DataFrame(rows)

    # Optional file-level aggregate row
    file_row = dict(
        platform_id="__ALL__",
        n=int(len(df)),
        dt_median_h=float(time_summary["dt_median_h"].median()) if not time_summary.empty else np.nan,
        dt_p10_h=np.nan,
        dt_p90_h=np.nan,
        n_gaps_gt_warn=int(time_summary["n_gaps_gt_warn"].sum()) if not time_summary.empty else 0,
        max_gap_h=float(time_summary["max_gap_h"].max()) if not time_summary.empty else np.nan,
    )
    time_summary = pd.concat([time_summary, pd.DataFrame([file_row])], ignore_index=True)

    return df, time_summary, warnings


def _aggregate_duplicates(
    df: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
    strategy: str,
) -> pd.DataFrame:
    """
    Aggregate duplicate timestamps per platform.
    Keeps non-numeric columns by taking first (except group/time).
    Numeric columns aggregated with median/first/last.
    """
    strategy = strategy.lower()
    if strategy not in ("median", "first", "last"):
        strategy = "median"

    cols = list(df.columns)
    num_cols = [c for c in cols if c not in (group_col, time_col) and pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in cols if c not in (group_col, time_col) and c not in num_cols]

    agg: dict[str, Any] = {}
    if strategy == "median":
        for c in num_cols:
            agg[c] = "median"
    else:
        for c in num_cols:
            agg[c] = strategy

    for c in other_cols:
        agg[c] = "first"

    out = (
        df.sort_values(time_col)
        .groupby([group_col, time_col], as_index=False)
        .agg(agg)
    )
    return out


def _apply_physical_bounds(
    df: pd.DataFrame,
    *,
    qc_cfg: Dict[str, Any],
    write_flags: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Hard physical bounds:
    - out-of-range => mask to NaN
    - optionally write boolean qc_out_of_bounds_<var>
    """
    bounds_cfg = qc_cfg.get("bounds", {}) or {}
    warnings: list[str] = []
    flag_cols: list[str] = []

    rows = []
    out = df.copy()

    for var, b in bounds_cfg.items():
        if var not in out.columns:
            continue

        v = out[var]
        v_ok = v.notna()
        n_total = int(v_ok.sum())
        if n_total == 0:
            rows.append(dict(var=var, n_total=0, n_out_of_bounds=0, frac_out_of_bounds=np.nan))
            continue

        vmin = b.get("min", None)
        vmax = b.get("max", None)

        mask_oob = pd.Series(False, index=out.index)
        if vmin is not None:
            mask_oob |= (v < float(vmin)) & v_ok
        if vmax is not None:
            mask_oob |= (v > float(vmax)) & v_ok

        n_oob = int(mask_oob.sum())
        frac = n_oob / n_total if n_total else np.nan
        rows.append(dict(var=var, n_total=n_total, n_out_of_bounds=n_oob, frac_out_of_bounds=frac))

        if n_oob > 0:
            warnings.append(f"[QC][bounds] {var}: masked {n_oob}/{n_total} points outside [{vmin}, {vmax}].")
            out.loc[mask_oob, var] = np.nan

        if write_flags:
            col = f"qc_out_of_bounds_{var}"
            out[col] = False
            out.loc[mask_oob, col] = True
            flag_cols.append(col)

    stats = pd.DataFrame(rows)
    return out, stats, warnings, flag_cols


def _detect_derivative_spikes(
    df: pd.DataFrame,
    *,
    qc_cfg: Dict[str, Any],
    time_col: str,
    group_col: str,
    write_flags: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Derivative-based spike detection:
    - compute dx/dt using true dt (irregular time supported)
    - spike if |dx/dt| > threshold (per variable)
    - mark the later point (i) of the jump (from i-1 to i)
    - action: mask_points or flag_only
    """
    dcfg = qc_cfg.get("derivative_spikes", {}) or {}
    enabled = bool(dcfg.get("enabled", True))
    if not enabled:
        return df, pd.DataFrame(columns=["var", "n_total", "n_spikes", "frac_spikes"]), [], []

    thr = (dcfg.get("thresholds_abs_per_s", {}) or {})
    action = str(dcfg.get("action", "mask_points")).lower()
    if action not in ("mask_points", "flag_only"):
        action = "mask_points"

    warnings: list[str] = []
    flag_cols: list[str] = []
    rows = []
    out = df.copy()

    vars_to_check = [v for v in thr.keys() if v in out.columns]
    if not vars_to_check:
        return out, pd.DataFrame(columns=["var", "n_total", "n_spikes", "frac_spikes"]), [], []

    for var in vars_to_check:
        spikes_all = pd.Series(False, index=out.index)

        for pid, g in out[[group_col, time_col, var]].sort_values(time_col).groupby(group_col):
            gg = g.dropna(subset=[time_col])
            if len(gg) < 3:
                continue

            # ensure monotonic time
            gg = gg.sort_values(time_col)

            t = gg[time_col].values.astype("datetime64[ns]")
            x = gg[var].astype(float).values

            dt_s = np.diff(t).astype("timedelta64[s]").astype(float)
            dx = np.diff(x)

            # Avoid division by 0 / negative dt (should not happen, but be safe)
            good = dt_s > 0
            if not np.any(good):
                continue

            der = np.full(dx.shape, np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                der[good] = dx[good] / dt_s[good]

            thr_abs = float(thr[var])
            spike_local = np.abs(der) > thr_abs

            idx = gg.index.values
            if spike_local.size > 0:
                spike_idx = idx[1:][spike_local]  # flag the later point in the jump
                spikes_all.loc[spike_idx] = True

        n_total = int(out[var].notna().sum())
        n_spikes = int(spikes_all.sum())
        frac = n_spikes / n_total if n_total else np.nan
        rows.append(dict(var=var, n_total=n_total, n_spikes=n_spikes, frac_spikes=frac))

        if n_spikes > 0:
            warnings.append(f"[QC][spikes] {var}: detected {n_spikes} derivative spikes (action={action}).")
            if action == "mask_points":
                out.loc[spikes_all, var] = np.nan

        if write_flags:
            col = f"qc_derivative_spike_{var}"
            out[col] = False
            out.loc[spikes_all, col] = True
            flag_cols.append(col)

    stats = pd.DataFrame(rows)
    return out, stats, warnings, flag_cols


def _detect_flatline(
    df: pd.DataFrame,
    *,
    qc_cfg: Dict[str, Any],
    time_col: str,
    group_col: str,
    write_flags: bool,
) -> Tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    """
    Flatline detection (stuck sensor heuristic):
    - compute time-based rolling std (window_hours) per platform
    - flag points where rolling std < eps_std[var]
    - default action: flag_only (do not mask)
    """
    fcfg = qc_cfg.get("flatline", {}) or {}
    enabled = bool(fcfg.get("enabled", True))
    if not enabled:
        empty = pd.DataFrame(columns=["var", "n_total", "n_flagged", "frac_flagged"])
        return empty, [], [], df

    window_h = float(fcfg.get("window_hours", 24.0))
    eps_std = (fcfg.get("eps_std", {}) or {})
    action = str(fcfg.get("action", "flag_only")).lower()
    if action not in ("flag_only", "mask_points"):
        action = "flag_only"

    warnings: list[str] = []
    flag_cols: list[str] = []
    rows = []
    out = df.copy()

    vars_to_check = [v for v in eps_std.keys() if v in out.columns]
    if not vars_to_check:
        empty = pd.DataFrame(columns=["var", "n_total", "n_flagged", "frac_flagged"])
        return empty, [], [], out

    for var in vars_to_check:
        flagged_all = pd.Series(False, index=out.index)

        for pid, g in out[[group_col, time_col, var]].groupby(group_col):
            gg = g.dropna(subset=[time_col]).sort_values(time_col).copy()
            if len(gg) < 5:
                continue

            s = gg.set_index(time_col)[var].astype(float)
            roll_std = s.rolling(f"{window_h}H", min_periods=3).std()

            eps = float(eps_std[var])
            flag = (roll_std < eps) & s.notna()

            if flag.any():
                flagged_all.loc[gg.index[flag.values]] = True

        n_total = int(out[var].notna().sum())
        n_flagged = int(flagged_all.sum())
        frac = n_flagged / n_total if n_total else np.nan
        rows.append(dict(var=var, n_total=n_total, n_flagged=n_flagged, frac_flagged=frac))

        if n_flagged > 0:
            warnings.append(
                f"[QC][flatline] {var}: flagged {n_flagged} points as potential flatline "
                f"(window={window_h}h, eps_std={eps_std[var]}, action={action})."
            )
            if action == "mask_points":
                out.loc[flagged_all, var] = np.nan

        if write_flags:
            col = f"qc_flatline_{var}"
            out[col] = False
            out.loc[flagged_all, col] = True
            flag_cols.append(col)

    stats = pd.DataFrame(rows)
    return stats, warnings, flag_cols, out


def _compute_overlap_overview(df: pd.DataFrame, *, qc_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Store overlap requirements in the summary for traceability.
    Actual enforcement is done inside analysis modules (correlations / multivariate).
    """
    ocfg = qc_cfg.get("pairwise_overlap", {}) or {}
    min_frac = float(ocfg.get("min_fraction_overlap", 0.7))
    min_pts = int(ocfg.get("min_points", 50))
    return pd.DataFrame([dict(platform_id="__ALL__", min_fraction_overlap=min_frac, min_points=min_pts)])


def _merge_summaries(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge summary tables on platform_id (one row per platform + __ALL__)."""
    out = None
    for p in parts:
        if p is None or p.empty:
            continue
        if out is None:
            out = p.copy()
        else:
            out = out.merge(p, on="platform_id", how="outer")
    return out if out is not None else pd.DataFrame()


def _merge_variable_stats(parts: list[pd.DataFrame], df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-variable QC stats and add basic descriptive statistics on cleaned data.
    """
    out = None
    for p in parts:
        if p is None or p.empty:
            continue
        if out is None:
            out = p.copy()
        else:
            out = out.merge(p, on="var", how="outer")

    if out is None:
        out = pd.DataFrame(columns=["var"])

    rows = []
    for var in sorted([c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]):
        s = df_clean[var]
        n = int(s.notna().sum())
        if n == 0:
            rows.append(
                dict(
                    var=var,
                    n_valid=0,
                    frac_valid=np.nan,
                    q01=np.nan,
                    median=np.nan,
                    q99=np.nan,
                    min=np.nan,
                    max=np.nan,
                )
            )
            continue
        rows.append(
            dict(
                var=var,
                n_valid=n,
                frac_valid=n / len(df_clean) if len(df_clean) else np.nan,
                q01=float(s.quantile(0.01)),
                median=float(s.quantile(0.50)),
                q99=float(s.quantile(0.99)),
                min=float(s.min()),
                max=float(s.max()),
            )
        )
    stats2 = pd.DataFrame(rows)

    out = out.merge(stats2, on="var", how="outer")
    return out


def _write_warnings(path: Path, warnings: list[str]) -> None:
    txt = "\n".join(warnings) + ("\n" if warnings else "")
    path.write_text(txt, encoding="utf-8")