from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .raw_split import split_raw_csv_into_platform_chunks
from .io import canonicalize_drifter_df


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bgcd.cli_prepare_drifter",
        description="Split and normalize raw drifter_data.csv into per-platform canonical files.",
    )
    p.add_argument("--input-file", required=True, help="Path to raw drifter_data.csv")
    p.add_argument("--output-dir", required=True, help="Directory where canonical files will be written")
    p.add_argument("--format", choices=["csv", "parquet"], default="csv")
    return p.parse_args(argv)


def _write(df: pd.DataFrame, out_base: Path, fmt: str) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_base.with_suffix(".csv"), index=False)
    else:
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)


def main(argv=None) -> int:
    args = _parse_args(argv)

    input_path = Path(args.input_file)
    out_dir = Path(args.output_dir)

    res = split_raw_csv_into_platform_chunks(input_path)
    print(f"chunks: {len(res.chunks)}")

    for pid, raw_df in res.chunks.items():
        try:
            canon = canonicalize_drifter_df(raw_df)
        except Exception as e:
            print(f"SKIP {pid} (canonicalize error): {e}")
            continue

        if canon.empty:
            print(f"SKIP {pid} (empty after canonicalize)")
            continue

        out_base = out_dir / f"drifter_{pid}"
        _write(canon, out_base, args.format)
        print(f"Wrote {out_base.with_suffix('.' + args.format)} rows {len(canon)} cols {canon.shape[1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
python -m bgcd.cli_prepare_drifter `
  --input-file ""C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/drifter_data.csv" `
  --output-dir ""C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_drifter_new"
"""