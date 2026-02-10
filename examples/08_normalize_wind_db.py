from __future__ import annotations

from pathlib import Path
import pandas as pd
from bgcd.io import canonicalize_wind_df

IN_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_wind_raw")
OUT_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_wind")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for fp in sorted(IN_DIR.glob("wind_raw_*.csv")):
    df = pd.read_csv(fp)

    try:
        canon = canonicalize_wind_df(df)
    except Exception as e:
        print(f"SKIP (error) {fp.name}: {e}")
        continue

    if canon.empty:
        print(f"SKIP (empty) {fp.name}: produced 0 rows after parsing")
        continue

    pid = canon["platform_id"].iloc[0]
    out = OUT_DIR / f"wind_{pid}.csv"
    canon.to_csv(out, index=False)
    print("Wrote", out.name, "rows", len(canon), "cols", len(canon.columns))
