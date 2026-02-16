from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .mat_io import load_mat_any


# Updated to match your .mat keys (case-insensitive matching is applied)
CANDIDATES = {
    "time": ["Time", "time"],
    "lat": ["Lat", "lat"],
    "lon": ["Lon", "lon"],
    "u": ["u"],
    "v": ["v"],
    "vorticity": ["vorti", "vorticity"],
    "strain": ["stra", "strain"],
}


def _pick_key(d: dict, names: list[str]) -> str | None:
    """Pick the first matching key from dict d, case-insensitive."""
    lower_to_orig = {k.lower(): k for k in d.keys()}
    for nm in names:
        key = lower_to_orig.get(nm.lower())
        if key is not None:
            return key
    return None


def _to_1d(x) -> pd.Series:
    """Flatten arrays and return as a pandas Series."""
    s = pd.Series(x)
    # if it's a numpy array with shape (n,1) or similar, flatten
    try:
        if getattr(x, "ndim", 1) > 1:
            s = pd.Series(x.ravel())
    except Exception:
        pass
    return s


def _matlab_datenum_to_datetime(dn: pd.Series) -> pd.Series:
    """
    Convert MATLAB datenum (days) to pandas datetime.
    """
    from datetime import datetime, timedelta
    import numpy as np

    arr = pd.to_numeric(dn, errors="coerce").to_numpy(dtype=float)
    out = []
    for x in arr:
        if not np.isfinite(x):
            out.append(pd.NaT)
            continue
        day = int(x)
        frac = x - day
        try:
            dt = datetime.fromordinal(day) + timedelta(days=frac) - timedelta(days=366)
        except Exception:
            dt = None
        out.append(dt)
    return pd.to_datetime(out, errors="coerce")


def _platform_id_from_filename(p: Path) -> str | None:
    """
    Extract platform id from filename like: a300534065378180_vort.mat -> 300534065378180
    """
    m = re.search(r"(\d{10,})", p.name)
    return m.group(1) if m else None


def mat_to_dataframe(mat_path: str | Path, platform_id: str | None = None) -> pd.DataFrame:
    """
    Convert one .mat file to a canonical DataFrame:
    platform_id, time_utc, lat, lon, u, v, vorticity, strain
    """
    d = load_mat_any(mat_path)

    k_time = _pick_key(d, CANDIDATES["time"])
    k_lat = _pick_key(d, CANDIDATES["lat"])
    k_lon = _pick_key(d, CANDIDATES["lon"])
    k_u = _pick_key(d, CANDIDATES["u"])
    k_v = _pick_key(d, CANDIDATES["v"])
    k_vort = _pick_key(d, CANDIDATES["vorticity"])
    k_str = _pick_key(d, CANDIDATES["strain"])

    required = {"time": k_time, "lat": k_lat, "lon": k_lon}
    missing = [name for name, key in required.items() if key is None]
    if missing:
        raise RuntimeError(
            f"{mat_path}: missing required variables {missing}. "
            f"Available keys: {sorted(d.keys())}"
        )

    df = pd.DataFrame(
        {
            "time_raw": _to_1d(d[k_time]),
            "lat": _to_1d(d[k_lat]),
            "lon": _to_1d(d[k_lon]),
        }
    )

    if k_u is not None:
        df["u"] = _to_1d(d[k_u])
    if k_v is not None:
        df["v"] = _to_1d(d[k_v])
    if k_vort is not None:
        df["vorticity"] = _to_1d(d[k_vort])
    if k_str is not None:
        df["strain"] = _to_1d(d[k_str])

    # time conversion: these files use MATLAB datenum
    df.insert(0, "time_utc", _matlab_datenum_to_datetime(df["time_raw"]))
    df.drop(columns=["time_raw"], inplace=True)

    if platform_id is not None:
        df.insert(0, "platform_id", str(platform_id))

    # basic cleanup
    for c in ["lat", "lon", "u", "v", "vorticity", "strain"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["time_utc", "lat", "lon"]).reset_index(drop=True)

    # if "u" in df and df["u"].abs().max() > 100:
    #     print(f"Warning: large |u| values in {mat_path} (max={df['u'].abs().max():.2f})")
    # if "v" in df and df["v"].abs().max() > 100:
    #     print(f"Warning: large |v| values in {mat_path} (max={df['v'].abs().max():.2f})")

    # Okubo–Weiss parameter:
    # Here "strain" is assumed to be the magnitude sqrt(sn^2 + ss^2),
    # so OW = strain^2 - vorticity^2
    if "strain" in df.columns and "vorticity" in df.columns:
        df["okubo_weiss"] = df["strain"] ** 2 - df["vorticity"] ** 2
    else:
        # keep column absent if inputs not available
        pass

    return df


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
    ap = argparse.ArgumentParser(description="Build per-platform DBs from MAT timeseries files.")
    ap.add_argument("--input-dir", required=True, help="Directory containing .mat files")
    ap.add_argument("--output-dir", required=True, help="Directory to write outputs")
    ap.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    ap.add_argument("--mode", choices=["per-platform", "single"], default="per-platform")
    ap.add_argument("--glob", default="*.mat", help="Glob pattern for MAT files")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {in_dir} matching {args.glob}")

    all_frames = []

    for fp in files:
        pid = _platform_id_from_filename(fp)
        df = mat_to_dataframe(fp, platform_id=pid)

        if args.mode == "per-platform":
            base = out_dir / f"mat_timeseries_{pid if pid else fp.stem}"
            out_path = write_df(df, base, args.format)
            print(f"Wrote {out_path}  ({len(df)} rows)")
        else:
            all_frames.append(df)
            print(f"Loaded {fp.name} ({len(df)} rows, pid={pid})")

    if args.mode == "single":
        big = pd.concat(all_frames, ignore_index=True)
        if "platform_id" in big.columns:
            big = big.sort_values(["platform_id", "time_utc"])
        else:
            big = big.sort_values(["time_utc"])
        out_path = write_df(big, out_dir / "mat_timeseries_all", args.format)
        print(f"Wrote combined DB: {out_path}  ({len(big)} rows)")


if __name__ == "__main__":
    main()



"""
python -m bgcd.cli_matdb --input-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/timeseries" --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" --format csv --mode per-platform

oppure
python -m bgcd.cli_matdb --input-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/timeseries" --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" --format csv --mode per-platform --glob "a300534065379230*.mat"
python -m bgcd.cli_matdb --input-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/timeseries" --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" --format csv --mode per-platform --glob "a300534065378180*.mat"
python -m bgcd.cli_matdb --input-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/timeseries" --output-dir "C:/Users/Jacopo/OneDrive - CNR/BGC-SVP/DATI_PLATFORMS/db_mat" --format csv --mode per-platform --glob "a300534065470010*.mat"

"""