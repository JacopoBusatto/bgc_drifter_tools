from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .master import MasterPaths, build_master_for_platform


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
    p.add_argument("--with-dt", action="store_true",
                   help="Include matched source timestamps and |Δt| diagnostics columns (dt_*_min).")

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

    paths = MasterPaths(
        drifter_db_dir=args.drifter_db_dir,
        wind_db_dir=args.wind_db_dir,
        mat_db_dir=args.mat_db_dir,
        extra_db_dirs=None,
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
                # NOTE: build_master_for_platform should accept with_dt.
                # If not yet implemented, just keep this line commented and run without --with-dt.
                with_dt=args.with_dt,  # type: ignore[arg-type]
            )
        except TypeError:
            # fallback if with_dt is not implemented yet
            if args.with_dt:
                print("NOTE: --with-dt requested but master builder does not support it yet. Ignoring.", file=sys.stderr)
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
  --drifter-db-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter" `
  --wind-db-dir    "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_wind" `
  --mat-db-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" `
  --output-dir     "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_master_cli" `
  --format csv `
  --target-time drifter



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
"""