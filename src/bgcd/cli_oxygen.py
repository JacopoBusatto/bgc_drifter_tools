from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .oxygen_io import oxygen_mat_to_dataframe, preprocess_oxygen_df

import warnings
from scipy.io.matlab import MatReadWarning

warnings.filterwarnings("ignore", category=MatReadWarning)

DEFAULT_KEEP_COLS = [
    # scriviamo *tutto* quello che abbiamo visto nei .mat (facile da cambiare)
    "AirSat_c",
    "CalPhase_c",
    "DO2_H2O_c",
    "DO2_c",
    "Depth_c",
    "Spsu_c",
    "TdegC_c",
    "oxy_comp_mgL_c",
    "oxy_comp_mlL_c",
    "oxy_comp_molar_c",
    "oxy_comp_saturation_c",
]


def _platform_id_from_filename(p: Path) -> str | None:
    m = re.search(r"(\d{10,})", p.name)
    return m.group(1) if m else None


def write_df(df: pd.DataFrame, out_base: Path, fmt: str) -> Path:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-platform oxygen DBs from oxigen_*.mat files.")
    ap.add_argument("--input-dir", required=True, help="Directory containing oxigen_*.mat files")
    ap.add_argument("--output-dir", required=True, help="Directory to write oxygen_<pid>.csv|parquet")
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    ap.add_argument(
        "--keep-cols",
        nargs="+",
        default=DEFAULT_KEEP_COLS,
        help="Columns to extract from MAT (keys as in .mat).",
    )
    ap.add_argument(
        "--zero-is-nan-cols",
        nargs="+",
        default=["DO2_c", "oxy_comp_mgL_c"],
        help="Convert 0 -> NaN for these columns (if present).",
    )
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    mats = sorted(in_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {in_dir}")

    for mp in mats:
        pid = _platform_id_from_filename(mp)
        if not pid:
            print(f"[skip] cannot parse platform_id from filename: {mp.name}")
            continue

        df = oxygen_mat_to_dataframe(mp, platform_id=pid, keep_cols=args.keep_cols)
        df = preprocess_oxygen_df(df, zero_is_nan_cols=args.zero_is_nan_cols)

        out_base = out_dir / f"oxygen_{pid}"
        out_path = write_df(df, out_base, args.format)
        print(f"[ok] {pid}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
