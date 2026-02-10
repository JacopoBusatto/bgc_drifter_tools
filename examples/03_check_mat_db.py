from __future__ import annotations

from pathlib import Path
import pandas as pd

DB_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_mat")

for fp in sorted(DB_DIR.glob("mat_timeseries_*.csv")):
    df = pd.read_csv(fp, parse_dates=["time_utc"])
    print("\n", fp.name)
    print("rows:", len(df))
    print("time range:", df["time_utc"].min(), "->", df["time_utc"].max())
    print("NaT frac:", df["time_utc"].isna().mean())
    print("cols:", list(df.columns))
