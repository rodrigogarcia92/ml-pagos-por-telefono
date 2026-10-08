"""Google Trends CSV -> validated monthly DataFrame, plus the `.meta.json` sidecar.

Manual route (plan 4.5, O-15): the owner exports ONE Trends request
holding two ENTITIES -- Yape ("Aplicación") and Plin ("Tema") -- and drops

    data/raw/google_trends/{pull_date}_yape_plin.csv
    data/raw/google_trends/{pull_date}_yape_plin.meta.json

into the repo. This module owns everything between that file and the loader.

    python -m src.data_collection.trends validate      # parse + report every pull
    python -m src.data_collection.trends reconcile     # final-month check (>= 2 pulls)

WHAT IS DELIBERATELY NOT ASSUMED
--------------------------------
Nothing about the export's layout is hardcoded. Trends' CSV changes with the UI
language and the export route: a preamble of title lines before the header, a
different delimiter, "2017-01" or "2017-01-01" dates, "<1" for sub-1 values. So the
parser FINDS the data (the first line whose first cell is a date), takes the line
above it as the header, and maps the two value columns to yape / plin by the
entity name in the header. If anything does not fit it raises with the offending
line, because a silently mis-read Trends file produces a plausible-looking feature.

A VALUE OF 0 IS NOT "<1". Trends prints "<1" for a value that rounds below 1 and
the plan (4.5) reads it as 0.5. A literal 0 is kept as 0 -- it is not this module's
place to decide that a zero means 0.5, and the consequence (log of a zero) is the
data gate's and dataset.py's to report, loudly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "google_trends"

SERIES_CODE = "GT_YAPE_PLIN"          # config.TRENDS_SERIES
PULL_RE = re.compile(r"^(?P<pull>\d{4}-\d{2}-\d{2})_yape_plin\.csv$")

SUBONE_VALUE = 0.5                    # plan 4.5: "<1" is read as 0.5
FIRST_MONTH = pd.Timestamp("2017-01-01")
FINAL_MONTH_TOLERANCE = 10.0          # index points; plan 4.5 / v1.7.1

# The request, as fixed by the plan (4.5, v1.7.1). Every pull must match it, and
# every pull must match every other pull: Trends rescales per request, so two pulls
# with different selections are not two measurements of one series.
CANONICAL_SELECTIONS = (
    {"column": "yape", "name": "Yape", "type": "Aplicación"},
    {"column": "plin", "name": "Plin", "type": "Tema"},
)
CANONICAL_REQUEST = {
    "geo": "PE",
    "period_start": "2017-01-01",
    "category": "All categories",
    "property": "Web search",
    "request_count": 1,
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
_NUM_RE = re.compile(r"^\d+([.,]\d+)?$")


class TrendsFormatError(ValueError):
    """The file is not the export this module understands. Carries the line."""


# --------------------------------------------------------------------------- #
# CSV -> DataFrame
# --------------------------------------------------------------------------- #
def _fail(path: Path, lineno: int, line: str, why: str) -> TrendsFormatError:
    return TrendsFormatError(f"{path.name}, line {lineno}: {why}\n    > {line!r}")


def _split(line: str, delimiter: str) -> list[str]:
    return [c.strip() for c in next(csv.reader([line], delimiter=delimiter))]


def _guess_delimiter(header: str) -> str:
    counts = {d: len(_split(header, d)) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best


def _parse_value(cell: str, path: Path, lineno: int, line: str) -> float:
    c = cell.strip().strip('"')
    if c.replace(" ", "") in ("<1", "&lt;1"):
        return SUBONE_VALUE
    if not _NUM_RE.match(c):
        raise _fail(path, lineno, line, f"value {cell!r} is neither a number nor '<1'")
    return float(c.replace(",", "."))


def _entity_column(header_cells: list[str], needle: str, path: Path, header: str, lineno: int) -> int:
    hits = [i for i, h in enumerate(header_cells[1:], start=1) if needle in h.lower()]
    if len(hits) != 1:
        raise _fail(path, lineno, header,
                    f"expected exactly one header cell mentioning {needle!r}, found {len(hits)}")
    return hits[0]


def parse_trends_csv(path: str | Path) -> pd.DataFrame:
    """Return DataFrame[date, yape, plin], one row per month, validated.

    Validation (plan 4.5): monthly, contiguous, starts 2017-01, no duplicates,
    values within 0..100. The incomplete current month is NOT trimmed here --
    that depends on the pull date, see `complete_months`.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    # The data starts at the first line whose first cell is a date; the header is the
    # non-blank line above it; everything before is preamble.
    first = next((i for i, ln in enumerate(lines) if _first_cell_is_date(ln)), None)
    if first is None:
        raise TrendsFormatError(f"{path.name}: no data line found (no line starts with a date).")
    header_idx = next((i for i in range(first - 1, -1, -1) if lines[i].strip()), None)
    if header_idx is None:
        raise _fail(path, first + 1, lines[first], "data starts at the top of the file: no header line")

    header = lines[header_idx]
    delim = _guess_delimiter(header)
    head = _split(header, delim)
    if len(head) != 3:
        raise _fail(path, header_idx + 1, header,
                    f"expected a date column and exactly two entity columns, found {len(head)} cells")
    c_yape = _entity_column(head, "yape", path, header, header_idx + 1)
    c_plin = _entity_column(head, "plin", path, header, header_idx + 1)

    rows = []
    for i in range(first, len(lines)):
        ln = lines[i]
        if not ln.strip():
            continue
        cells = _split(ln, delim)
        if len(cells) != 3 or not _DATE_RE.match(cells[0]):
            raise _fail(path, i + 1, ln, "not a data line (expected: date, value, value)")
        rows.append({
            "date": pd.Timestamp(cells[0][:7] + "-01") if len(cells[0]) == 7 else pd.Timestamp(cells[0]),
            "yape": _parse_value(cells[c_yape], path, i + 1, ln),
            "plin": _parse_value(cells[c_plin], path, i + 1, ln),
            "_line": i + 1,
        })
    df = pd.DataFrame(rows)
    _validate(df, path)
    return df.drop(columns="_line").reset_index(drop=True)


def _first_cell_is_date(line: str) -> bool:
    if not line.strip():
        return False
    for d in (",", ";", "\t"):
        cells = _split(line, d)
        if len(cells) > 1 and _DATE_RE.match(cells[0]):
            return True
    return False


def _validate(df: pd.DataFrame, path: Path) -> None:
    def bad(msg: str, row: pd.Series | None = None):
        where = f", line {int(row['_line'])}" if row is not None else ""
        raise TrendsFormatError(f"{path.name}{where}: {msg}")

    not_first = df[df["date"].dt.day != 1]
    if len(not_first):
        bad(f"not monthly: {not_first['date'].iloc[0]:%Y-%m-%d} is not the first of a month",
            not_first.iloc[0])
    dup = df[df["date"].duplicated(keep=False)]
    if len(dup):
        bad(f"duplicate month {dup['date'].iloc[0]:%Y-%m}", dup.iloc[0])
    if df["date"].iloc[0] != FIRST_MONTH:
        bad(f"first month is {df['date'].iloc[0]:%Y-%m}, expected {FIRST_MONTH:%Y-%m}", df.iloc[0])
    expected = pd.date_range(FIRST_MONTH, periods=len(df), freq="MS")
    off = df.index[df["date"].to_numpy() != expected.to_numpy()]
    if len(off):
        r = df.loc[off[0]]
        bad(f"not contiguous: expected {expected[off[0]]:%Y-%m}, found {r['date']:%Y-%m}", r)
    for col in ("yape", "plin"):
        out = df[(df[col] < 0) | (df[col] > 100)]
        if len(out):
            bad(f"{col} = {out[col].iloc[0]} is outside 0..100", out.iloc[0])


def consolidated(df: pd.DataFrame) -> pd.Series:
    """yape + plin, in index points (plan 4.5), indexed by month."""
    return (df["yape"] + df["plin"]).set_axis(df["date"]).rename("gt_yape_plin")


def pull_date_of(path: str | Path) -> str:
    m = PULL_RE.match(Path(path).name)
    if not m:
        raise TrendsFormatError(
            f"{Path(path).name}: expected a name like 2026-10-05_yape_plin.csv")
    return m.group("pull")


def complete_months(df: pd.DataFrame, pull_date: str) -> pd.DataFrame:
    """Drop the pull month (and anything later): Trends reports it incomplete.

    The plan's period is "2017-01-01 -> last complete month". A pull's last kept
    month must therefore be the month BEFORE the pull; anything else means a stale
    or truncated export and raises.
    """
    pull_month = pd.Timestamp(pull_date).to_period("M").to_timestamp()
    out = df[df["date"] < pull_month].reset_index(drop=True)
    last_complete = pull_month - pd.DateOffset(months=1)
    if out.empty or out["date"].iloc[-1] != last_complete:
        got = f"{out['date'].iloc[-1]:%Y-%m}" if len(out) else "nothing"
        raise TrendsFormatError(
            f"pull {pull_date}: last complete month should be {last_complete:%Y-%m}, got {got}.")
    return out


# --------------------------------------------------------------------------- #
# Sidecar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pull:
    pull_date: str
    csv: Path
    meta: dict
    raw: pd.DataFrame            # as exported, validated
    data: pd.DataFrame           # complete months only, final-month rule applied

    @property
    def selections(self) -> tuple:
        return tuple(tuple(sorted(s.items())) for s in self.meta["selections"])


def meta_path(csv_path: str | Path) -> Path:
    p = Path(csv_path)
    return p.with_name(p.stem + ".meta.json")


def sha256(path: Path) -> str:
    # Line endings normalised: git may check the file out as CRLF on Windows.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_meta(csv_path: Path, *, source: str = "manual", notes: str = "") -> dict:
    """The sidecar for one export. The selections are the plan's, not read from the
    file: a Trends CSV does not say which entity TYPE a column was, and that is
    exactly what the sidecar exists to record."""
    pull = pull_date_of(csv_path)
    pull_month = pd.Timestamp(pull).to_period("M").to_timestamp()
    last_complete = (pull_month - pd.DateOffset(months=1)).strftime("%Y-%m")
    return {
        "pull_date": pull,
        "source": source,
        "csv_file": Path(csv_path).name,
        "csv_sha256": sha256(Path(csv_path)),
        "selections": [dict(s) for s in CANONICAL_SELECTIONS],
        **CANONICAL_REQUEST,
        "period_end": last_complete,
        "final_month_check": {"status": "pending",
                              "note": "needs a second pull on a different day: `trends reconcile`"},
        "notes": notes,
    }


REQUIRED_META_KEYS = ("pull_date", "selections", "geo", "period_start", "category",
                      "property", "request_count")


def load_pull(csv_path: str | Path) -> Pull:
    """CSV + sidecar -> Pull. Refuses a CSV without a sidecar."""
    csv_path = Path(csv_path)
    pull = pull_date_of(csv_path)
    mp = meta_path(csv_path)
    if not mp.exists():
        raise TrendsFormatError(
            f"{csv_path.name}: no sidecar {mp.name}. The selections of a pull must be "
            "recorded (plan 4.5); refusing to load a CSV whose request is unknown.")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise TrendsFormatError(f"{mp.name}: missing key(s) {missing}")
    if meta["pull_date"] != pull:
        raise TrendsFormatError(f"{mp.name}: pull_date {meta['pull_date']!r} != file name {pull!r}")
    if meta.get("csv_sha256") and meta["csv_sha256"] != sha256(csv_path):
        raise TrendsFormatError(f"{csv_path.name}: file changed since its sidecar was written (sha256).")
    if int(meta["request_count"]) != 1:
        raise TrendsFormatError(
            f"{mp.name}: request_count={meta['request_count']}. Both entities must come from ONE "
            "request, or they do not share a scale and their sum is meaningless.")
    for key, want in CANONICAL_REQUEST.items():
        if meta[key] != want:
            raise TrendsFormatError(f"{mp.name}: {key}={meta[key]!r}, the plan fixes {want!r} (4.5).")
    canon = tuple(tuple(sorted(s.items())) for s in CANONICAL_SELECTIONS)
    got = tuple(tuple(sorted(s.items())) for s in meta["selections"])
    if got != canon:
        raise TrendsFormatError(
            f"{mp.name}: selections {meta['selections']} differ from the plan's "
            f"{list(CANONICAL_SELECTIONS)} (4.5).")

    raw = parse_trends_csv(csv_path)
    data = complete_months(raw, pull)
    check = meta.get("final_month_check", {})
    if check.get("last_month_dropped"):
        dropped = pd.Timestamp(check["month"] + "-01")
        if data["date"].iloc[-1] != dropped:
            raise TrendsFormatError(f"{mp.name}: final_month_check names {check['month']}, "
                                    f"but this pull's last complete month is {data['date'].iloc[-1]:%Y-%m}")
        data = data[data["date"] < dropped].reset_index(drop=True)
    return Pull(pull_date=pull, csv=csv_path, meta=meta, raw=raw, data=data)


def load_pulls(raw_dir: Path = RAW_DIR) -> list[Pull]:
    """Every pull in the directory, oldest first, selections cross-checked."""
    files = sorted(p for p in Path(raw_dir).glob("*_yape_plin.csv") if PULL_RE.match(p.name))
    pulls = [load_pull(f) for f in files]
    check_same_selections(pulls)
    return pulls


def check_same_selections(pulls: list[Pull]) -> None:
    """Pulls with different selections are not repeat measurements of one series."""
    if not pulls:
        return
    ref = pulls[0]
    for p in pulls[1:]:
        for key in ("selections", "geo", "period_start", "category", "property"):
            if p.meta[key] != ref.meta[key]:
                raise TrendsFormatError(
                    f"pull {p.pull_date} differs from pull {ref.pull_date} on {key!r}: "
                    f"{p.meta[key]!r} vs {ref.meta[key]!r}. Identical selections are required (4.5).")


# --------------------------------------------------------------------------- #
# Final-month check between two pulls
# --------------------------------------------------------------------------- #
def final_month_check(a: Pull, b: Pull, tolerance: float = FINAL_MONTH_TOLERANCE) -> dict:
    """Compare two pulls on the last complete month both contain.

    The newest month is the least settled; if two pulls disagree on it by more than
    `tolerance` index points (on the consolidated series) it is not usable, and the
    month is dropped from BOTH pulls (plan 4.5, v1.7.1).
    """
    ca, cb = consolidated(a.raw), consolidated(b.raw)
    last = min(a.data["date"].iloc[-1], b.data["date"].iloc[-1])
    va, vb = float(ca[last]), float(cb[last])
    diff = abs(va - vb)
    return {
        "status": "checked",
        "month": f"{last:%Y-%m}",
        "pulls": [a.pull_date, b.pull_date],
        "values": [va, vb],
        "abs_difference": diff,
        "tolerance": tolerance,
        "last_month_dropped": bool(diff > tolerance),
    }


def reconcile(raw_dir: Path = RAW_DIR, write: bool = True) -> dict:
    pulls = load_pulls(raw_dir)
    if len(pulls) < 2:
        return {"status": "pending", "note": f"{len(pulls)} pull(s); two are needed"}
    a, b = pulls[-2], pulls[-1]
    res = final_month_check(a, b)
    if write:
        for p in pulls[-2:]:
            meta = dict(p.meta)
            meta["final_month_check"] = res
            meta_path(p.csv).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["validate", "reconcile"])
    ap.add_argument("--dir", default=str(RAW_DIR))
    a = ap.parse_args()

    if a.command == "validate":
        pulls = load_pulls(Path(a.dir))
        if not pulls:
            raise SystemExit(f"no *_yape_plin.csv in {a.dir}")
        for p in pulls:
            s = consolidated(p.data)
            print(f"{p.csv.name}: {len(p.raw)} months exported, {len(p.data)} complete "
                  f"({s.index[0]:%Y-%m} .. {s.index[-1]:%Y-%m}); "
                  f"yape mean {p.data['yape'].mean():.1f}, plin mean {p.data['plin'].mean():.1f}; "
                  f"final-month check: {p.meta.get('final_month_check', {}).get('status')}")
    else:
        print(json.dumps(reconcile(Path(a.dir)), indent=2))


if __name__ == "__main__":
    main()
