"""Drift monitor: did the forecasts we made turn out to be right?

    python -m src.forecasting.monitor --snapshot data/processed/panel_<version>.parquet

Joins forecasts/history.csv with the actuals in a snapshot. For every target month that now has an
actual, it scores the forecast that was made FOR it -- the earliest one if several vintages exist
(a later vintage may have seen part of the answer) -- as a percentage error and as "inside the 80%
/ 90% band or not".

  status  ok        nothing to worry about (also: nothing scored yet)
          warning   the latest scored month missed by more than 10%
          alert     the two latest scored months are consecutive calendar months and BOTH
                    missed by more than 10%

Writes forecasts/monitor.json. Exit code 0 for ok and warning, 2 for alert, so CI can react.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.forecasting import production
from src.model_training import dataset, tracking
from src.model_training.snapshot import latest_snapshot

OUT_DIR = Path("forecasts")
THRESHOLD_PCT = 10.0          # a "miss"
ROLLING_MONTHS = 12           # window of the rolling MAPE
EXIT = {"ok": 0, "warning": 0, "alert": 2}


def load_history(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["target_month", "data_version", "point_forecast"])
    return pd.read_csv(path, dtype={"target_month": str, "data_version": str, "origin": str})


def actuals(panel: pd.DataFrame, target_id: str) -> pd.Series:
    """The target series by month ('YYYY-MM'), published months only."""
    cols = dataset.TARGETS[target_id]["cols"]
    s = panel[cols].sum(axis=1, min_count=len(cols)).dropna()
    s.index = s.index.strftime("%Y-%m")
    return s


def score(history: pd.DataFrame, actual: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """One row per scored target month, plus the forecast months still waiting for an actual."""
    if history.empty:
        return pd.DataFrame(columns=["target_month"]), []
    h = history.copy()
    sort = [c for c in ("created_at", "data_version") if c in h.columns]
    first = h.sort_values(sort, kind="stable").drop_duplicates("target_month", keep="first")
    first = first.sort_values("target_month")

    known = first["target_month"].isin(actual.index)
    pending = sorted(first.loc[~known, "target_month"])
    s = first[known].copy()
    s["actual"] = s["target_month"].map(actual).astype(float)
    s["forecast"] = s["point_forecast"].astype(float)
    s["pct_error"] = (s["forecast"] - s["actual"]) / s["actual"] * 100       # + = over-forecast
    s["abs_pct_error"] = s["pct_error"].abs()
    s["in_80"] = (s["actual"] >= s["lo80"]) & (s["actual"] <= s["hi80"])
    s["in_90"] = (s["actual"] >= s["lo90"]) & (s["actual"] <= s["hi90"])
    keep = ["target_month", "origin", "data_version", "forecast", "actual", "pct_error",
            "abs_pct_error", "in_80", "in_90"]
    return s[[c for c in keep if c in s.columns]].reset_index(drop=True), pending


def _consecutive(a: str, b: str) -> bool:
    return (pd.Period(b, "M") - pd.Period(a, "M")).n == 1


def status_of(scored: pd.DataFrame, threshold: float = THRESHOLD_PCT) -> str:
    """ok / warning / alert from the scored months (rules in the module docstring)."""
    if scored.empty:
        return "ok"
    s = scored.sort_values("target_month")
    miss = s["abs_pct_error"] > threshold
    if not bool(miss.iloc[-1]):
        return "ok"
    if len(s) >= 2 and bool(miss.iloc[-2]) and _consecutive(s["target_month"].iloc[-2],
                                                            s["target_month"].iloc[-1]):
        return "alert"
    return "warning"


def summarize(history: pd.DataFrame, actual: pd.Series, *, snapshot_version: str,
              threshold: float = THRESHOLD_PCT, window: int = ROLLING_MONTHS) -> dict:
    scored, pending = score(history, actual)
    status = status_of(scored, threshold)
    recent = scored.sort_values("target_month").tail(window) if len(scored) else scored
    out = {
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "snapshot_data_version": snapshot_version,
        "threshold_pct": threshold,
        "n_scored": int(len(scored)),
        "latest_scored_month": None if scored.empty else str(scored["target_month"].max()),
        "pending_months": pending,
        "rolling_window_months": window,
        "rolling_mape_pct": None if recent.empty else round(float(recent["abs_pct_error"].mean()), 3),
        "coverage": {
            "n": int(len(scored)),
            "within_80": None if scored.empty else round(float(scored["in_80"].mean()), 4),
            "within_90": None if scored.empty else round(float(scored["in_90"].mean()), 4),
            "nominal_80": 0.80, "nominal_90": 0.90,
        },
        "scored": [] if scored.empty else json.loads(
            scored.round(4).to_json(orient="records")),
    }
    if status != "ok":
        last = scored.sort_values("target_month").tail(2)
        out["reason"] = (f"{status}: latest scored month(s) "
                         + ", ".join(f"{m} ({e:+.1f}%)" for m, e in
                                     zip(last["target_month"], last["pct_error"], strict=True))
                         + f" against a {threshold:.0f}% threshold")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: newest)")
    ap.add_argument("--history", default=str(OUT_DIR / "history.csv"))
    ap.add_argument("--out", default=str(OUT_DIR / "monitor.json"))
    ap.add_argument("--config", default=str(production.DEFAULT_CONFIG))
    ap.add_argument("--no-write", action="store_true", help="print, write nothing")
    a = ap.parse_args(argv)

    snap = Path(a.snapshot) if a.snapshot else latest_snapshot()
    panel, _ = dataset.load_snapshot(snap)
    cfg = production.load(a.config)
    res = summarize(load_history(a.history), actuals(panel, cfg["target_id"]),
                    snapshot_version=tracking.data_version(snap))

    print(f"monitor: {res['status'].upper()}   scored {res['n_scored']} month(s), "
          f"pending {res['pending_months'] or 'none'}")
    if res["n_scored"]:
        c = res["coverage"]
        print(f"  rolling MAPE ({res['rolling_window_months']}m) {res['rolling_mape_pct']:.2f}%   "
              f"coverage 80%: {c['within_80']:.0%}  90%: {c['within_90']:.0%}")
    if "reason" in res:
        print(f"  {res['reason']}")
    if not a.no_write:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return EXIT[res["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
