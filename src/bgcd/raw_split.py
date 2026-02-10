from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import re

@dataclass(frozen=True)
class Chunk:
    header: str
    lines: List[str]


def _clean_lines(path: str | Path) -> List[str]:
    """
    Remove empty lines and '</br>' tokens. Does NOT try to interpret columns.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = s.replace("</br>", "").strip()
        if not s:
            continue
        out.append(s)
    return out


def split_into_chunks(path: str | Path, header_prefix: str = "Platform-ID") -> List[Chunk]:
    """
    Split a CSV file into chunks each starting with a header line.
    We do NOT assume any fixed set of columns beyond the header_prefix.
    """
    lines = _clean_lines(path)

    chunks: List[Chunk] = []
    cur_header: Optional[str] = None
    cur_lines: List[str] = []

    for ln in lines:
        if ln.startswith(header_prefix):
            # flush previous
            if cur_header is not None and cur_lines:
                chunks.append(Chunk(header=cur_header, lines=cur_lines))
            cur_header = ln
            cur_lines = [ln]
        else:
            if cur_header is not None:
                cur_lines.append(ln)

    if cur_header is not None and cur_lines:
        chunks.append(Chunk(header=cur_header, lines=cur_lines))

    return chunks


def read_chunk_df(chunk: Chunk) -> pd.DataFrame:
    """
    Read one chunk into a DataFrame using pandas.
    Keeps the original column names (stripped).
    """
    txt = "\n".join(chunk.lines)
    df = pd.read_csv(
        StringIO(txt),
        skipinitialspace=True,
        engine="python",
        on_bad_lines="skip",
    )
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")].copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def write_per_platform_from_chunks(
    chunks: List[Chunk],
    out_dir: str | Path,
    base_name: str,
    fmt: str = "csv",
) -> None:
    """
    For each chunk:
      - read it as DataFrame
      - require a 'Platform-ID' column
      - group by Platform-ID and append to per-platform files

    Output files:
      out_dir/{base_name}_{platform_id}.csv|parquet
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # accumulate frames per platform in memory (simple, ok for your sizes)
    acc: dict[str, list[pd.DataFrame]] = {}

    for ch in chunks:
        df = read_chunk_df(ch)
        if "Platform-ID" not in df.columns:
            # skip weird blocks
            continue

        # normalize ID to string
        df["Platform-ID"] = df["Platform-ID"].astype(str).str.strip()
        valid = df["Platform-ID"].str.fullmatch(r"\d{10,}")
        df = df[valid].copy()
        if df.empty:
            continue

        for pid, g in df.groupby("Platform-ID", sort=False):
            acc.setdefault(pid, []).append(g)

    # write
    for pid, frames in acc.items():
        big = pd.concat(frames, ignore_index=True, sort=False)

        # optional: sort if a time column exists
        for tcol in ("Timestamp(UTC)", "GPS-Timestamp(utc)", "Time", "time"):
            if tcol in big.columns:
                big[tcol] = pd.to_datetime(big[tcol], errors="coerce")
                big = big.sort_values(tcol)
                break

        base = out / f"{base_name}_{pid}"
        if fmt == "csv":
            big.to_csv(base.with_suffix(".csv"), index=False)
        elif fmt == "parquet":
            big.to_parquet(base.with_suffix(".parquet"), index=False)
        else:
            raise ValueError("fmt must be csv or parquet")
