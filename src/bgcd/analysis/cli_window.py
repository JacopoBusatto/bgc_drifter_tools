# ================================
# File: src/bgcd/analysis/cli_window.py
# ================================
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from bgcd.analysis.qc_core import qc_time, qc_duplicates, qc_bounds, qc_gap_fill, qc_coverage
from bgcd.analysis.overlap import find_overlap_windows, OverlapResult


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")


def _platform_id_from_df_or_filename(df: pd.DataFrame, master_path: Path) -> str:
    if "platform_id" in df.columns:
        s = df["platform_id"].dropna()
        if not s.empty:
            return str(s.mode().iloc[0])
    return master_path.stem


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _save_overlap_windows_csv(res: OverlapResult, out_csv: Path, topk: int = 10) -> pd.DataFrame:
    rows = []
    for w in res.windows:
        rows.append(
            {
                "start": w.start,
                "end": w.end,
                "duration_h": w.duration_h,
                "n_rows": w.n_rows,
            }
        )
    dfw = pd.DataFrame(rows)
    if dfw.empty:
        pd.DataFrame(columns=["start", "end", "duration_h", "n_rows"]).to_csv(out_csv, index=False)
        return dfw

    dfw = dfw.sort_values(["duration_h", "n_rows"], ascending=[False, False]).reset_index(drop=True)
    if topk and topk > 0:
        dfw.head(topk).to_csv(out_csv, index=False)
    else:
        dfw.to_csv(out_csv, index=False)
    return dfw

def _extract_window_subset(
    df: pd.DataFrame,
    *,
    time_col: str,
    required: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Rebuild the subset rows belonging to a window defined by (start, end) and "required valid rows".
    This matches the semantics in overlap.find_overlap_windows:
    - time is parsed UTC
    - rows are considered only if all required are non-NaN
    - window is along sorted time
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out = out.loc[out[time_col].notna()].sort_values(time_col)

    # required valid rows
    valid = pd.Series(True, index=out.index)
    for v in required:
        if v in out.columns:
            valid &= out[v].notna()
        else:
            # if required missing, caller already handles, but keep safe:
            valid &= False

    in_window = (out[time_col] >= start) & (out[time_col] <= end)
    subset = out.loc[in_window & valid].copy()
    return subset

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bgcd.analysis.cli_window",
        description="QC (optional) + best window selection (required vars) for one MASTER drifter CSV.",
    )
    ap.add_argument("--master", type=str, required=True, help="Path to MASTER CSV (one drifter).")
    ap.add_argument("--config", type=str, required=True, help="Path to analysis_config_min.yml.")
    ap.add_argument(
        "--outdir",
        type=str,
        default="OUT",
        help="Base output directory. Outputs are isolated in OUT/<platform_id>/...",
    )
    ap.add_argument(
        "--run-qc",
        type=str,
        default="true",
        help="true/false. If true: time+duplicates+bounds+coverage before window selection.",
    )
    ap.add_argument("--topk", type=int, default=10, help="How many top windows to write in windows_top.csv.")
    ap.add_argument(
        "--export-topk",
        type=str,
        default="false",
        help="true/false. If true, also export master_subset_window_<i>.csv for i=1..topk (sorted by duration).",
    )
    args = ap.parse_args()

    master_path = Path(args.master)
    cfg_path = Path(args.config)
    out_base = Path(args.outdir)

    cfg = _load_yaml(cfg_path)
    analysis_cfg = (cfg or {}).get("analysis", {}) or {}
    dw = analysis_cfg.get("data_window", {}) or {}
    qc_cfg = (analysis_cfg.get("qc", {}) or {})

    required = _dedup_keep_order(dw.get("required", []) or [])
    optional = _dedup_keep_order(dw.get("optional", []) or [])
    min_points = int(dw.get("min_points", 50))
    max_gap_h = float(dw.get("max_gap_h", 6.0))
    prefer = str(dw.get("prefer", "duration"))

    gap_cfg = (dw.get("gap_handling", {}) or {})
    gap_max_fill_h = float(gap_cfg.get("max_fill_h", 0.0))
    gap_method = str(gap_cfg.get("method", "time"))
    gap_limit_area = str(gap_cfg.get("limit_area", "inside"))
    gap_vars = _dedup_keep_order(gap_cfg.get("vars", []) or [])  # optional override
    if not gap_vars:
        gap_vars = required  # default: only required affect window continuity

    run_qc = _as_bool(args.run_qc)
    export_topk = _as_bool(args.export_topk)
    topk = int(args.topk)

    df = pd.read_csv(master_path)
    pid = _platform_id_from_df_or_filename(df, master_path)

    out_platform = out_base / pid
    out_qc = out_platform / "A_qc"
    out_win = out_platform / "B_window"

    _ensure_dir(out_qc / "data")
    _ensure_dir(out_qc / "reports")
    _ensure_dir(out_win / "data")
    _ensure_dir(out_win / "reports")

    run_log: List[str] = []
    run_log.append(f"MASTER: {master_path}")
    run_log.append(f"CONFIG: {cfg_path}")
    run_log.append(f"PLATFORM_ID: {pid}")
    run_log.append(f"RUN_QC: {run_qc}")
    run_log.append("")
    run_log.append("[data_window]")
    run_log.append(f"required: {required}")
    run_log.append(f"optional: {optional}")
    run_log.append(f"min_points: {min_points}")
    run_log.append(f"max_gap_h: {max_gap_h}")
    run_log.append(f"prefer: {prefer}")
    run_log.append("[gap_handling]")
    run_log.append(f"max_fill_h: {gap_max_fill_h}")
    run_log.append(f"method: {gap_method}")
    run_log.append(f"limit_area: {gap_limit_area}")
    run_log.append(f"vars: {gap_vars}")
    run_log.append("")

    # sanity: required columns exist
    missing_required_cols = [c for c in required if c not in df.columns]
    if missing_required_cols:
        _write_text(out_win / "run_log.txt", run_log + [f"ERROR: missing required columns: {missing_required_cols}"])
        raise SystemExit(f"Missing required columns in MASTER: {missing_required_cols}")

    df_qc = df.copy()
    warnings_all: List[str] = []

    if run_qc:
        time_cfg = qc_cfg.get("time", {}) or {}
        bounds_cfg = qc_cfg.get("bounds", {}) or {}

        # STEP 1: time
        df_qc, rep_time = qc_time(
            df_qc,
            time_col="time_utc",
            expected_dt_hours=float(time_cfg.get("expected_dt_hours", 3.0)),
            dt_tolerance_hours=float(time_cfg.get("dt_tolerance_hours", 0.75)),
            gap_warn_hours=float(time_cfg.get("gap_warn_hours", 12.0)),
        )
        warnings_all.extend(rep_time.warnings)

        # STEP 2: duplicates
        strategy = str(time_cfg.get("duplicates_strategy", "median"))
        df_qc, rep_dups = qc_duplicates(
            df_qc,
            group_col="platform_id",
            time_col="time_utc",
            strategy=strategy,
        )
        warnings_all.extend(rep_dups.warnings)

        # STEP 3: bounds
        df_qc, rep_bounds = qc_bounds(
            df_qc,
            bounds_cfg=bounds_cfg,
        )
        warnings_all.extend(rep_bounds.warnings)

        # STEP 3b: gap fill (optional)
        if gap_max_fill_h > 0:
            df_qc, rep_gap = qc_gap_fill(
                df_qc,
                time_col="time_utc",
                vars_to_fill=gap_vars,
                max_fill_h=gap_max_fill_h,
                method=gap_method,
                limit_area=gap_limit_area,
            )
            warnings_all.extend(rep_gap.warnings)

            # write gap-fill report
            print(out_qc)
            rep_gap.per_var.to_csv(out_qc / "reports" / "qc_gap_fill_per_var.csv", index=False)
            run_log.append(f"QC wrote: {out_qc / 'reports' / 'qc_gap_fill_per_var.csv'}")
            if rep_gap.warnings:
                run_log.append(f"QC gap_fill warnings: {len(rep_gap.warnings)}")
            run_log.append("")

        # STEP 4: coverage (returns report + warnings list)
        rep_cov, cov_warnings = qc_coverage(
            df_qc,
            time_col="time_utc",
            vars_to_check=_dedup_keep_order(required + optional),
        )
        warnings_all.extend(rep_cov.warnings)
        warnings_all.extend(cov_warnings)

        # write QC artifacts
        df_qc.to_csv(out_qc / "data" / "qc_clean_master.csv", index=False)
        rep_cov.per_var.to_csv(out_qc / "reports" / "qc_coverage_per_var.csv", index=False)

        # compact warnings
        if warnings_all:
            _write_text(out_qc / "warnings.txt", warnings_all)

        run_log.append(f"QC wrote: {out_qc / 'data' / 'qc_clean_master.csv'}")
        run_log.append(f"QC wrote: {out_qc / 'reports' / 'qc_coverage_per_var.csv'}")
        if warnings_all:
            run_log.append(f"QC warnings: {out_qc / 'warnings.txt'}")
        run_log.append("")

    # STEP 5: window selection
    res_win: OverlapResult = find_overlap_windows(
        df_qc,
        time_col="time_utc",
        required=required,
        optional=optional,
        min_points=min_points,
        max_gap_h=max_gap_h,
        prefer=prefer,
    )

    windows_csv = out_win / "reports" / "windows_top.csv"
    df_windows = _save_overlap_windows_csv(res_win, windows_csv, topk=topk)

    run_log.append(f"WINDOWS wrote: {windows_csv}")
    run_log.append(f"WINDOWS found: {len(res_win.windows)}")
    run_log.append("")

    if res_win.best is None:
        _write_text(
            out_win / "best_window.txt",
            [
                "NO VALID WINDOW FOUND",
                f"required={required}",
                f"min_points={min_points}",
                f"max_gap_h={max_gap_h}",
            ],
        )
        _write_text(out_win / "run_log.txt", run_log + ["NO VALID WINDOW FOUND"])
        raise SystemExit("No valid window found (check required vars / min_points / max_gap_h).")


    best = res_win.best

    subset = _extract_window_subset(
        df_qc,
        time_col="time_utc",
        required=required,
        start=best.start,
        end=best.end,
    )

    subset_path = out_win / "data" / "master_subset_best_window.csv"
    subset.to_csv(subset_path, index=False)

    best_lines = [
        "BEST WINDOW",
        f"start: {best.start}",
        f"end:   {best.end}",
        f"duration_h: {best.duration_h:.3f}",
        f"n_rows (expected): {best.n_rows}",
        f"n_rows (exported): {len(subset)}",
        "",
        f"required: {required}",
        f"optional: {optional}",
        f"min_points: {min_points}",
        f"max_gap_h: {max_gap_h}",
        f"prefer: {prefer}",
    ]
    _write_text(out_win / "best_window.txt", best_lines)

    run_log.append(f"BEST WINDOW wrote: {out_win / 'best_window.txt'}")
    run_log.append(f"SUBSET wrote: {subset_path}")
    run_log.append("")

    # optionally export topK subsets (in the same order as res_win.windows)
    if export_topk and len(res_win.windows) > 0:
        n_export = min(topk, len(res_win.windows))
        for i in range(n_export):
            w = res_win.windows[i]
            df_i = _extract_window_subset(
                df_qc,
                time_col="time_utc",
                required=required,
                start=w.start,
                end=w.end,
            )
            out_i = out_win / "data" / f"master_subset_window_{i+1:02d}.csv"
            df_i.to_csv(out_i, index=False)
        run_log.append(f"EXPORTED topK subsets: {n_export}")
        run_log.append("")

    _write_text(out_win / "run_log.txt", run_log)

    print(f"[OK] platform_id={pid}")
    print(f"[OK] best window: {best.start} -> {best.end}  (n={best.n_rows}, dur_h={best.duration_h:.1f})")
    print(f"[OK] outputs in: {out_platform}")


if __name__ == "__main__":
    main()
"""
python -m bgcd.analysis.cli_window `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065378180.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --run-qc true `
  --topk 10 `
  --export-topk false

python -m bgcd.analysis.cli_window `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065379230.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --run-qc true `
  --topk 10 `
  --export-topk false

python -m bgcd.analysis.cli_window `
  --master "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master/master_300534065470010.csv" `
  --config "analysis_config/analysis_config_min.yml" `
  --outdir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/OUT" `
  --run-qc true `
  --topk 10 `
  --export-topk false
"""