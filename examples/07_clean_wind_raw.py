from __future__ import annotations

from bgcd.raw_split import split_into_chunks, write_per_platform_from_chunks

INP = r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\wind_data.csv"
OUT = r"C:\Users\Jacopo\OneDrive - CNR\BGC-SVP\DATI_PLATFORMS\db_wind_raw"

chunks = split_into_chunks(INP)
print("chunks:", len(chunks))

write_per_platform_from_chunks(chunks, OUT, base_name="wind_raw", fmt="csv")
print("done")
