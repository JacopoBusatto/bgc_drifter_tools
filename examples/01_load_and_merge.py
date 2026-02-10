from __future__ import annotations

from pathlib import Path
import bgcd

DRIFTER_CSV = r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\drifter_data.csv"
WIND_CSV    = r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\wind_data.csv"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

d = bgcd.read_drifter_csv(DRIFTER_CSV)
w = bgcd.read_wind_csv(WIND_CSV)

m = bgcd.merge_drifter_wind(d, w, tolerance="30min")

print("drifter rows:", len(d), "platforms:", d["platform_id"].nunique())
print("wind rows:", len(w), "platforms:", w["platform_id"].nunique())
print("merged rows:", len(m), "platforms:", m["platform_id"].nunique())
print("time range:", m["time_utc"].min(), "->", m["time_utc"].max())
print(m.head())

OUTPUT_FORMAT = "csv"  # "csv" or "parquet"

out_path = OUT_DIR / f"merged.{OUTPUT_FORMAT}"

if OUTPUT_FORMAT == "parquet":
    try:
        m.to_parquet(out_path, index=False)
    except ImportError as e:
        raise RuntimeError(
            "Parquet support requires pyarrow. "
            "Install with: pip install -e .[parquet]"
        ) from e
else:
    m.to_csv(out_path, index=False)

print("Wrote:", out_path)