from __future__ import annotations

from pathlib import Path
import bgcd
from bgcd.master import MasterPaths, build_master_for_platform

PATHS = MasterPaths(
    drifter_csv=r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\drifter_data.csv",
    wind_csv=r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\wind_data.csv",
    mat_db_dir=r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_mat",
)

PID = "300534065378180"

# Choose: "drifter" or "hourly"
TARGET = "drifter"

master = build_master_for_platform(
    PID,
    PATHS,
    target_time=TARGET,
    wind_tolerance="30min",
    mat_tolerance="30min",
)

print(master.head())
print("rows:", len(master))
print("time range:", master["time_utc"].min(), "->", master["time_utc"].max())
print("columns:", list(master.columns))

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)
master.to_csv(out_dir / f"master_{PID}_{TARGET}.csv", index=False)
print("Wrote:", out_dir / f"master_{PID}_{TARGET}.csv")
