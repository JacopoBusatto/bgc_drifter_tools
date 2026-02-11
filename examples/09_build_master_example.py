from __future__ import annotations

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# CONFIG: edit these paths on your machine
# ---------------------------------------------------------------------
PID = "300534065378180"

DRIFTER_DB_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_drifter")
WIND_DB_DIR    = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_wind")
MAT_DB_DIR     = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_mat")

OUT_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_master")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FORMAT = "csv"      # "csv" | "parquet"
WIND_TOL = "30min"
MAT_TOL  = "30min"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def read_per_platform(db_dir: Path, base: str, pid: str) -> pd.DataFrame:
    parquet = db_dir / f"{base}_{pid}.parquet"
    csv     = db_dir / f"{base}_{pid}.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Missing {base}_{pid} in {db_dir}")

    if "time_utc" in df.columns:
        df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df["platform_id"] = df["platform_id"].astype(str).str.strip()
    df = df.dropna(subset=["platform_id", "time_utc"])
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df


def merge_asof_with_source_time(
    base: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tol: str,
    source_label: str,
) -> pd.DataFrame:
    """
    Merge 'right' onto 'base' by nearest time, keeping the matched timestamp.
    Adds:
      - time_<source_label> (timestamp of matched row)
      - dt_<source_label>_min (abs time diff, minutes)
    """
    r = right.copy()
    r[f"time_{source_label}"] = r["time_utc"]  # keep original time as a separate column

    merged = pd.merge_asof(
        base.sort_values("time_utc"),
        r.drop(columns=["platform_id"]),
        on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta(tol),
    )

    merged[f"dt_{source_label}_min"] = (
        (merged["time_utc"] - merged[f"time_{source_label}"])
        .abs()
        .dt.total_seconds()
        / 60.0
    )
    return merged


# ---------------------------------------------------------------------
# Load canonical per-platform DBs
# ---------------------------------------------------------------------
d = read_per_platform(DRIFTER_DB_DIR, "drifter", PID)
w = read_per_platform(WIND_DB_DIR, "wind", PID)
m = read_per_platform(MAT_DB_DIR, "mat_timeseries", PID)

# Select what we want from each dataset (current spec)
d_keep = ["platform_id", "time_utc", "lat", "lon", "sst_c", "slp_mb"]
d = d[[c for c in d_keep if c in d.columns]].copy()

# wind: keep all (already canonical), except keys will be merged
# mat: keep rest
m_keep = ["platform_id", "time_utc", "u", "v", "vorticity", "strain"]
m = m[[c for c in m_keep if c in m.columns]].copy()

if d.empty:
    raise RuntimeError(f"Drifter DB is empty for PID={PID}")
if m.empty:
    raise RuntimeError(f"MAT DB is empty for PID={PID}")

# ---------------------------------------------------------------------
# Build MASTER on drifter timeline
# ---------------------------------------------------------------------
base = d.sort_values("time_utc").reset_index(drop=True)

master = base.copy()
master = merge_asof_with_source_time(master, w, tol=WIND_TOL, source_label="wind")
master = merge_asof_with_source_time(master, m, tol=MAT_TOL, source_label="mat")

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
print("\nMASTER diagnostics")
print("PID:", PID)
print("rows:", len(master))
print("time range:", master["time_utc"].min(), "->", master["time_utc"].max())

# completeness
mat_missing = master["vorticity"].isna().mean() if "vorticity" in master.columns else float("nan")
wind_missing = master["wspd"].isna().mean() if "wspd" in master.columns else float("nan")
print(f"MAT missing fraction:  {mat_missing:.3f}")
print(f"WIND missing fraction: {wind_missing:.3f}")

# dt stats (only where match exists)
for lab in ["wind", "mat"]:
    dtc = f"dt_{lab}_min"
    if dtc in master.columns:
        s = master[dtc].dropna()
        if len(s):
            print(f"{lab.upper()} |Δt| minutes: median={s.median():.2f}, p90={s.quantile(0.90):.2f}, max={s.max():.2f}")
        else:
            print(f"{lab.upper()} |Δt| minutes: no matched rows within tolerance")

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
out_base = OUT_DIR / f"master_{PID}"
if OUT_FORMAT == "csv":
    master.to_csv(out_base.with_suffix(".csv"), index=False)
elif OUT_FORMAT == "parquet":
    master.to_parquet(out_base.with_suffix(".parquet"), index=False)
else:
    raise ValueError("OUT_FORMAT must be 'csv' or 'parquet'")

print("\nWrote:", out_base.with_suffix("." + OUT_FORMAT))
print(master.head(5))
