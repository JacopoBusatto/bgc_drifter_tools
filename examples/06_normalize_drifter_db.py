from __future__ import annotations

from pathlib import Path
import pandas as pd
import bgcd
from bgcd.io import canonicalize_drifter_df

IN_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_drifter_raw")
OUT_DIR = Path(r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_drifter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for fp in sorted(IN_DIR.glob("drifter_raw_*.csv")):
    df = pd.read_csv(fp)
    canon = canonicalize_drifter_df(df)
    pid = canon["platform_id"].iloc[0]
    out = OUT_DIR / f"drifter_{pid}.csv"
    canon.to_csv(out, index=False)
    print("Wrote", out.name, "rows", len(canon), "cols", len(canon.columns))
