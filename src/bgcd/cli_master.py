from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .master import MasterPaths, build_master_for_platform, ExtraSource

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bgcd.cli_master",
        description="Build per-platform MASTER tables by merging canonical drifter, wind and MAT databases.",
    )

    p.add_argument("--platform-id", action="append", default=[], help="Platform ID (repeatable).")
    p.add_argument("--platform-ids-file", default=None, help="Text file with one platform_id per line.")

    p.add_argument("--drifter-db-dir", required=True, help="Directory containing drifter_<pid>.csv|parquet")
    p.add_argument("--wind-db-dir", required=True, help="Directory containing wind_<pid>.csv|parquet")
    p.add_argument("--mat-db-dir", required=True, help="Directory containing mat_timeseries_<pid>.csv|parquet")

    p.add_argument("--output-dir", required=True, help="Where to write master outputs.")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format.")
    p.add_argument("--mode", choices=["per-platform", "single"], default="per-platform",
                   help="Write one file per platform, or a single concatenated file.")
    p.add_argument("--target-time", choices=["drifter", "hourly"], default="drifter",
                   help="Base timeline for the master table.")
    p.add_argument("--wind-tolerance", default="30min", help="merge_asof tolerance for wind onto base timeline.")
    p.add_argument("--mat-tolerance", default="30min", help="merge_asof tolerance for MAT onto base timeline.")
    p.add_argument("--hourly-freq", default="1H", help="Hourly frequency when target-time=hourly.")

    # optional diagnostics columns
    p.add_argument("--with-dt", action="store_true", help="Include matched source timestamps and |Δt| diagnostics columns (dt_*_min).")
    p.add_argument("--no-bbox-filter", action="store_true", help="Disable automatic Mediterranean bbox filter.")
    p.add_argument("--segment-filter", action="store_true", help="Keep only largest contiguous time segment.")
    p.add_argument("--segment-max-gap", default="7D", help="Maximum allowed time gap inside contiguous segment (default: 7D).")

    p.add_argument("--oxygen-db-dir", default=None, help="Directory containing oxygen_<pid>.csv|parquet")
    p.add_argument("--oxygen-tolerance", default="30min", help="merge_asof tolerance for oxygen onto base timeline.")
    p.add_argument("--oxygen-cols", nargs="+", default=["DO2_c", "oxy_comp_mgL_c"], help="Oxygen columns to merge into master (easy to extend).")

    return p.parse_args(argv)


def _load_platform_ids(args: argparse.Namespace) -> list[str]:
    pids = [str(x).strip() for x in args.platform_id if str(x).strip()]

    if args.platform_ids_file:
        fp = Path(args.platform_ids_file)
        if not fp.exists():
            raise FileNotFoundError(f"platform-ids-file not found: {fp}")
        for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = ln.strip()
            if s and not s.startswith("#"):
                pids.append(s)

    # unique, preserve order
    seen = set()
    out = []
    for pid in pids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)

    if not out:
        raise ValueError("No platform ids provided. Use --platform-id or --platform-ids-file.")
    return out


def _write(df: pd.DataFrame, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path.with_suffix(".csv"), index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    else:
        raise ValueError("fmt must be csv or parquet")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.output_dir)

    pids = _load_platform_ids(args)

    extra_db_dirs = {}
    extras = []

    if args.oxygen_db_dir:
        extra_db_dirs["oxygen"] = args.oxygen_db_dir
        extras.append(
            ExtraSource(
                name="oxygen",
                base_name="oxygen",
                tolerance=args.oxygen_tolerance,
                keep_cols=args.oxygen_cols,
            )
        )

    paths = MasterPaths(
        drifter_db_dir=args.drifter_db_dir,
        wind_db_dir=args.wind_db_dir,
        mat_db_dir=args.mat_db_dir,
        extra_db_dirs=extra_db_dirs or None,
    )

    masters: list[pd.DataFrame] = []

    for pid in pids:
        try:
            df = build_master_for_platform(
                pid,
                paths,
                target_time=args.target_time,
                wind_tolerance=args.wind_tolerance,
                mat_tolerance=args.mat_tolerance,
                hourly_freq=args.hourly_freq,
                extras=extras or None,
                with_dt=args.with_dt,
                apply_bbox_filter=not args.no_bbox_filter,
                apply_segment_filter=args.segment_filter,
                segment_max_gap=args.segment_max_gap,
            )
        except TypeError:
            df = build_master_for_platform(
                pid,
                paths,
                target_time=args.target_time,
                wind_tolerance=args.wind_tolerance,
                mat_tolerance=args.mat_tolerance,
                hourly_freq=args.hourly_freq,
            )

        if args.mode == "per-platform":
            out_base = out_dir / f"master_{pid}"
            _write(df, out_base, args.format)
            print(f"Wrote {out_base.with_suffix('.' + args.format)} ({len(df)} rows)")
        else:
            masters.append(df)

    if args.mode == "single":
        big = pd.concat(masters, ignore_index=True) if masters else pd.DataFrame()
        out_base = out_dir / "master_all"
        _write(big, out_base, args.format)
        print(f"Wrote {out_base.with_suffix('.' + args.format)} ({len(big)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
pip install -e .

python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --platform-id 300534065379230 `
  --platform-id 300534065470010 `
  --drifter-db-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter" `
  --wind-db-dir    "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_wind" `
  --mat-db-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --output-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master" `
  --format csv `
  --mat-tolerance 2H `
  --target-time drifter `
  --mode per-platform `
  --segment-filter `
  --segment-max-gap 7D


SOLO BBOX
python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --platform-id 300534065379230 `
  --platform-id 300534065470010 `
  --drifter-db-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter" `
  --wind-db-dir    "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_wind" `
  --mat-db-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --output-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master" `
  --format csv `
  --mat-tolerance 2H `
  --target-time drifter `
  --mode per-platform

NESSUN FILTRO
python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --drifter-db-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter" `
  --wind-db-dir    "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_wind" `
  --mat-db-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --output-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master" `
  --format csv `
  --mat-tolerance 2H `
  --target-time drifter `
  --mode per-platform `
  --no-bbox-filter


MASTER con OXYGEN + controllo Δt
python -m bgcd.cli_master `
  --platform-id 300534065378180 `
  --platform-id 300534065379230 `
  --platform-id 300534065470010 `
  --drifter-db-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter" `
  --wind-db-dir    "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_wind" `
  --mat-db-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --oxygen-db-dir  "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_oxygen" `
  --oxygen-cols DO2_c oxy_comp_mgL_c `
  --oxygen-tolerance 30min `
  --output-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master" `
  --format csv `
  --target-time drifter `
  --mode per-platform `
  --with-dt `
  --segment-filter `
  --segment-max-gap 7D

PER AGGIUNGERE COLONNE> --oxygen-cols DO2_c oxy_comp_mgL_c oxy_comp_saturation_c TdegC_c
"""