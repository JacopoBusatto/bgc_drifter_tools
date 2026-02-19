# src/bgcd/cli_prepare_bbp.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .bbp_io import bbp_csv_to_dataframe


def _write(df: pd.DataFrame, out_base: Path, fmt: str) -> Path:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "csv":
        out_path = out_base.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        return out_path
    if fmt == "parquet":
        out_path = out_base.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    raise ValueError("format must be csv or parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build per-platform BBP DBs from bbp_raw/*.csv files.")
    ap.add_argument("--input-dir", required=True, help="Directory containing raw bbp csv files (e.g. 3005340....csv)")
    ap.add_argument("--output-dir", required=True, help="Directory to write bbp_<pid>.csv|parquet")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern for bbp raw files")
    args = ap.parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.glob}")

    for fp in files:
        pid = fp.stem  # filename is the platform id
        try:
            df = bbp_csv_to_dataframe(fp, platform_id=pid)
        except Exception as e:
            print(f"[skip] {fp.name}: {e}")
            continue

        out_base = out_dir / f"bbp_{pid}"
        out_path = _write(df, out_base, args.format)
        print(f"[ok] {pid}: {len(df)} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Example:

python -m bgcd.cli_prepare_bbp `
  --input-dir  "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/bbp_raw" `
  --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_bbp" `
  --format csv
"""
