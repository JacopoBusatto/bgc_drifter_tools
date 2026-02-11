from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    """
    Output of the raw CSV splitter.

    chunks: dict platform_id -> raw DataFrame (as parsed with that chunk header)
    """
    chunks: dict[str, pd.DataFrame]


def _clean_line(line: str) -> str:
    # remove html junk, strip spaces
    return line.replace("</br>", "").strip()


def split_raw_csv_into_platform_chunks(path: str | Path) -> SplitResult:
    """
    Split a multi-header CSV container into per-platform raw DataFrames.

    This handles files like:
      headerA
      rows...
      </br>
      headerB
      rows...
      ...

    Strategy:
    - read file as text
    - remove empty lines and '</br>'
    - detect header lines (currently: lines starting with 'Platform-ID')
    - for each header block, collect subsequent rows until next header
    - parse each block with pandas.read_csv(StringIO(block_text))
    - group by Platform-ID and concatenate across blocks (even if different schemas)

    Notes
    -----
    - Output DataFrames are "raw": columns depend on the header block.
    - Later steps (canonicalization) will normalize names/types.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [_clean_line(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empty

    # Find header indices
    header_idx: list[int] = [i for i, ln in enumerate(lines) if ln.startswith("Platform-ID")]
    if not header_idx:
        raise ValueError("No header lines detected (expected lines starting with 'Platform-ID').")

    # Add sentinel end
    header_idx.append(len(lines))

    blocks: list[list[str]] = []
    for i in range(len(header_idx) - 1):
        start = header_idx[i]
        end = header_idx[i + 1]
        block = lines[start:end]
        # must contain header + at least one row
        if len(block) >= 2:
            blocks.append(block)

    # Parse blocks and collect per platform
    per_pid: dict[str, list[pd.DataFrame]] = {}

    for block in blocks:
        header = block[0]
        rows = block[1:]

        # Some exports have trailing commas → pandas creates unnamed columns; ok.
        csv_text = "\n".join([header, *rows])

        try:
            df = pd.read_csv(
                pd.io.common.StringIO(csv_text),
                skipinitialspace=True,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            # skip completely broken blocks
            continue

        # If Platform-ID column missing, skip
        if "Platform-ID" not in df.columns:
            continue

        # Drop completely empty/garbage columns
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        df.columns = [c.strip() for c in df.columns]

        # Normalize platform id to string
        df["Platform-ID"] = df["Platform-ID"].astype(str).str.strip()

        for pid, g in df.groupby("Platform-ID", sort=False):
            per_pid.setdefault(pid, []).append(g.copy())

    # Concatenate lists
    chunks: dict[str, pd.DataFrame] = {}
    for pid, parts in per_pid.items():
        # concat with union of columns (different schemas) → NaN where missing
        chunks[pid] = pd.concat(parts, ignore_index=True, sort=False)

    return SplitResult(chunks=chunks)
