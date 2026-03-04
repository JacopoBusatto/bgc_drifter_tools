from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from bgcd.analysis.qc_core import qc_time, qc_duplicates, qc_bounds, qc_coverage
from bgcd.analysis.overlap import find_overlap_windows


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def _dedup_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def main() -> None:
    p = argparse.ArgumentParser(description="QC sandbox runner (step-by-step).")
    p.add_argument("--master", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--time-col", default="time_utc")
    p.add_argument("--group-col", default="platform_id")
    p.add_argument("--csv-sep", default=",")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument(
        "--stop-after",
        default="overlap",
        choices=["time", "dups", "bounds", "coverage", "overlap"],
    )

    args = p.parse_args()

    df = pd.read_csv(args.master, sep=args.csv_sep, encoding=args.encoding)
    cfg = _load_yaml(Path(args.config))

    qc_cfg = (cfg.get("analysis", {}) or {}).get("qc", {}) or {}
    time_cfg = qc_cfg.get("time", {}) or {}
    bounds_cfg = qc_cfg.get("bounds", {}) or {}

    # STEP 1: TIME
    df1, rep1 = qc_time(
        df,
        time_col=args.time_col,
        expected_dt_hours=float(time_cfg.get("expected_dt_hours", 3.0)),
        dt_tolerance_hours=float(time_cfg.get("dt_tolerance_hours", 0.75)),
        gap_warn_hours=float(time_cfg.get("gap_warn_hours", 12.0)),
    )
    print("\n=== QC STEP 1: TIME ===")
    print(f"rows in:   {rep1.n_rows_in}")
    print(f"bad time:  {rep1.n_bad_time}")
    print(f"rows out:  {rep1.n_rows_out}")
    print(f"dt median: {rep1.dt_median_h}")
    print(f"dt p10:    {rep1.dt_p10_h}")
    print(f"dt p90:    {rep1.dt_p90_h}")
    print(f"gaps >thr: {rep1.n_gaps_gt_warn}")
    print(f"max gap h: {rep1.max_gap_h}")
    print("Warnings:", "none" if not rep1.warnings else "")
    for w in rep1.warnings:
        print(" - " + w)

    if args.stop_after == "time":
        return

    # STEP 2: DUPLICATES
    dup_strategy = str(time_cfg.get("duplicates_strategy", "median"))
    df2, rep2 = qc_duplicates(df1, group_col=args.group_col, time_col=args.time_col, strategy=dup_strategy)
    print("\n=== QC STEP 2: DUPLICATES ===")
    print(f"rows in:         {rep2.n_rows_in}")
    print(f"dup rows:        {rep2.n_dup_rows}")
    print(f"dup key groups:  {rep2.n_groups_with_dups}")
    print(f"strategy:        {rep2.strategy}")
    print(f"rows out:        {rep2.n_rows_out}")
    print("Warnings:", "none" if not rep2.warnings else "")
    for w in rep2.warnings:
        print(" - " + w)

    if args.stop_after == "dups":
        return

    # STEP 3: BOUNDS
    df3, rep3 = qc_bounds(df2, bounds_cfg=bounds_cfg)
    print("\n=== QC STEP 3: BOUNDS ===")
    if rep3.per_var.empty:
        print("No bounds configured or no matching variables found.")
    else:
        any_oob = rep3.per_var.loc[rep3.per_var["n_oob"] > 0].copy()
        if any_oob.empty:
            print("Out-of-bounds: none.")
        else:
            print(any_oob.to_string(index=False))
    print("Warnings:", "none" if not rep3.warnings else "")
    for w in rep3.warnings:
        print(" - " + w)

    if args.stop_after == "bounds":
        return

    # STEP 4: COVERAGE
    rep4, _ = qc_coverage(
        df3,
        time_col=args.time_col,
        vars_to_check=None,
        warn_frac_valid_below=0.5,
    )
    print("\n=== QC STEP 4: COVERAGE ===")
    low = rep4.per_var.loc[rep4.per_var["frac_valid"] < 0.8].sort_values("frac_valid")
    if low.empty:
        print("All variables have frac_valid >= 0.8")
    else:
        cols = ["var", "n_valid", "frac_valid", "n_valid_runs", "longest_missing_run_h", "first_valid_time", "last_valid_time"]
        print(low[cols].to_string(index=False))
    print("Warnings:", "none" if not rep4.warnings else "")
    for w in rep4.warnings:
        print(" - " + w)

    if args.stop_after == "coverage":
        return

    # STEP 5: OVERLAP (YAML-driven)
    ov_cfg = ((cfg.get("analysis", {}) or {}).get("data_window", {}) or {})
    required = _dedup_keep_order([str(x) for x in (ov_cfg.get("required", []) or [])])
    optional = _dedup_keep_order([str(x) for x in (ov_cfg.get("optional", []) or [])])

    min_points = int(ov_cfg.get("min_points", 50))
    max_gap_h = float(ov_cfg.get("max_gap_h", 6.0))
    prefer = str(ov_cfg.get("prefer", "duration"))

    print("\n=== QC STEP 5: OVERLAP ===")
    print("required:", required)
    print("optional:", optional)
    print(f"min_points={min_points}, max_gap_h={max_gap_h}, prefer={prefer!r}")

    if not required:
        print("No required variables configured in YAML -> skipping overlap.")
        return

    res = find_overlap_windows(
        df3,
        time_col=args.time_col,
        required=required,
        optional=optional,
        min_points=min_points,
        max_gap_h=max_gap_h,
        prefer=prefer,
    )

    if res.best is None:
        print("No overlap window found for REQUIRED variables (all-NaN overlap or too short).")
        return

    b = res.best
    print("\nBEST WINDOW")
    print(f"start:     {b.start}")
    print(f"end:       {b.end}")
    print(f"n_rows:    {b.n_rows}")
    print(f"duration_h:{b.duration_h:.1f}  (~{b.duration_h/24.0:.1f} days)")

    # Show top 5 windows
    ws = res.windows[:]
    ws_sorted = sorted(ws, key=lambda w: (-w.duration_h, -w.n_rows))
    print("\nTOP WINDOWS (up to 5)")
    for i, w in enumerate(ws_sorted[:5], start=1):
        print(f"{i:02d}) {w.start} -> {w.end} | n={w.n_rows} | {w.duration_h/24.0:.2f} days")

    if args.stop_after == "overlap":
        return


if __name__ == "__main__":
    main()