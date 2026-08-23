"""Parse raw BCRP snapshots into long-format DataFrames.

Single source of truth for snapshot parsing — imported by both the EDA
notebook and the BigQuery loader, so the two can never drift apart.

Long format, one row per (series, month):
    series_code | col_name | obs_date | value | pulled_at | source_batch
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data_collection.config import COL_NAME_BY_CODE

# project_root/src/data_collection/parse.py -> project_root/data/raw/bcrp
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "bcrp"

# Spanish month abbreviations used by the BCRP API -> month number.
# The API returns "Sep"; "Set" (setiembre) is accepted too in case it ever
# switches to the other common Peruvian abbreviation.
MONTH_MAP = {
    "Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
    "Sep": 9, "Set": 9, "Oct": 10, "Nov": 11, "Dic": 12,
}


def parse_snapshot(path: Path) -> pd.DataFrame:
    """Return a long-format DataFrame from one raw BCRP snapshot.

    Each snapshot holds exactly ONE series. The BCRP API identifies series
    only by a long `name` string, never by code, and does not return batched
    series in request order — so the fetch script requests one code at a time,
    which makes the code -> values mapping correct by construction.
    """
    snapshot = json.loads(path.read_text(encoding="utf-8"))

    if "code" not in snapshot:
        raise ValueError(
            f"{path.name}: outdated snapshot format (no 'code'). "
            "Re-run the fetch script, then delete this file."
        )

    code = snapshot["code"]
    pulled_at = pd.Timestamp(snapshot["pulled_at"], tz="UTC")
    payload = snapshot["response"]

    n_series = len(payload["config"]["series"])
    if n_series != 1:
        raise ValueError(f"{path.name}: expected 1 series, found {n_series}.")

    rows = []
    for period in payload["periods"]:
        month_abbr, year_str = period["name"].split(".")  # e.g. "Ene.2024"
        if month_abbr not in MONTH_MAP:
            raise ValueError(
                f"{path.name}: unrecognised month abbreviation {month_abbr!r} in "
                f"period {period['name']!r}. Add it to MONTH_MAP."
            )
        raw_val = period["values"][0]
        rows.append(
            {
                "series_code": code,
                "col_name": COL_NAME_BY_CODE.get(code, code),
                "obs_date": pd.Timestamp(
                    year=int(year_str), month=MONTH_MAP[month_abbr], day=1
                ),
                "value": None if raw_val in ("n.d.", "", None) else float(raw_val),
            }
        )

    df = pd.DataFrame(rows)
    df["pulled_at"] = pulled_at
    df["source_batch"] = path.name
    return df


def load_all_snapshots(raw_dir: Path = RAW_DIR, verbose: bool = True) -> pd.DataFrame:
    """Parse every snapshot in `raw_dir` into one long DataFrame.

    Files in an outdated format are skipped with a warning rather than
    aborting the run — an old file shouldn't block a fresh pull.
    """
    files = sorted(raw_dir.glob("*.json"))
    if verbose:
        print(f"Found {len(files)} file(s) in {raw_dir}\n")

    frames, skipped = [], []
    for f in files:
        try:
            frames.append(parse_snapshot(f))
            if verbose:
                print(f"  OK   {f.name}")
        except ValueError as e:
            skipped.append(f.name)
            if verbose:
                print(f"  SKIP {f.name} — {e}")

    if not frames:
        raise SystemExit(
            "\nNo parseable snapshots. Run:\n"
            "    python -m src.data_collection.fetch_target_series"
        )

    if skipped and verbose:
        print(
            f"\n{len(skipped)} file(s) skipped (old format). Safe to delete once "
            "the current pull covers the same series."
        )

    df = pd.concat(frames, ignore_index=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Latest pull per (series, month), pivoted to one column per series.

    BCRP revises recent months, so when the same (series, month) appears in
    more than one pull, the newest pull wins by design.
    """
    latest = (
        df.sort_values(["series_code", "obs_date", "pulled_at"])
        .drop_duplicates(subset=["series_code", "obs_date"], keep="last")
    )
    wide = latest.pivot(index="obs_date", columns="col_name", values="value").sort_index()
    wide.columns.name = None
    return wide
