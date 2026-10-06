"""The forecast command: the frozen model, refitted on everything published, one month ahead.

    python -m src.forecasting.predict --snapshot data/processed/panel_20261005T000000Z.parquet
    python -m src.forecasting.predict --snapshot ... --out forecasts/    # the default --out
    python -m src.forecasting.predict --snapshot ... --no-write          # print only

What it does
  1. loads configs/production/t3_ens3.yaml (frozen members, columns, error band);
  2. fits the three members on ALL labelled rows of the window in the snapshot -- the CV period
     and the holdout months alike. This is the production fit, not an evaluation: it scores
     nothing, and no error is computed or printed for any month;
  3. forecasts the latest admissible origin: the most recent month t for which every FS3 feature
     is available under its kappa (payments 2, pbi 2, circulante 1). The target month is
     t - kappa + h = t + 1 for t3. On snapshot 20261005 that is origin 2026-09 -> target 2026-10;
  4. wraps the point forecast in the frozen CV error band (80% and 90%).

Output: forecasts/forecast_{target_month}_{data_version}.json and one row in
forecasts/history.csv (created with a header if missing, never a duplicate (target_month,
data_version)).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import ensemble, production
from src.model_training import dataset, registry, tracking
from src.model_training.snapshot import latest_snapshot

OUT_DIR = Path("forecasts")
HISTORY = "history.csv"
HISTORY_COLUMNS = ["target_month", "origin", "data_version", "point_forecast",
                   "lo80", "hi80", "lo90", "hi90", "last_actual_month", "last_actual",
                   "model_id", "config_hash", "git_sha", "created_at"]
UNIT = "millions of transfers per month"


def latest_origin(pred_frames: dict[str, dataset.Frame]) -> pd.Timestamp:
    """The most recent origin that every member can be built for."""
    sets = [set(f.X.index) for f in pred_frames.values()]
    common = set.intersection(*sets)
    if not common:
        raise RuntimeError("no admissible forecast origin: every feature is unavailable")
    return max(common)


def _row(frame: dataset.Frame, origin: pd.Timestamp) -> dataset.Frame:
    return replace(frame, X=frame.X.loc[[origin]], y=frame.y.loc[[origin]],
                   ctx=frame.ctx.loc[[origin]])


def forecast(panel: pd.DataFrame, meta: pd.DataFrame, cfg: dict, *, data_version: str,
             cfg_path: str | Path = production.DEFAULT_CONFIG) -> dict:
    """The forecast record for the latest admissible origin of `panel`."""
    target_id, horizon, window = cfg["target_id"], cfg["horizon"], cfg["window"]
    kappa = ensemble.kappa_of(meta, target_id)
    if kappa != cfg["kappa"]:
        raise RuntimeError(f"snapshot kappa={kappa} for {target_id}, frozen config says {cfg['kappa']}")
    params = production.member_params(cfg)

    frames = ensemble.member_frames(panel, meta, target_id=target_id, horizon=horizon,
                                    window=window, columns=cfg["columns"])
    pred = {
        fam: dataset.build_prediction_rows(
            panel, meta, target_id=target_id, horizon=horizon, window=window,
            columns=registry.model_columns(fam, cfg["columns"]),
            encoding=registry.default_encoding(fam))
        for fam in ensemble.MEMBERS
    }
    origin = latest_origin(pred)
    row = {fam: _row(f, origin) for fam, f in pred.items()}
    anchor = float(row[ensemble.MEMBERS[0]].ctx["anchor"].iloc[0])
    target_month = origin + pd.DateOffset(months=horizon - kappa)

    z_ens, z_members = ensemble.fit_predict(frames, row, params, horizon=horizon)
    point = anchor * float(np.exp(z_ens))
    band = cfg["error_band"]
    iv = ensemble.apply_band(point, band)

    series = dataset.TARGETS[target_id]["cols"]
    actual = panel[series].sum(axis=1, min_count=len(series)).dropna()

    return {
        "target_month": f"{target_month:%Y-%m}",
        "origin": f"{origin:%Y-%m}",
        "data_version": data_version,
        "point_forecast": round(point, 3),
        "unit": UNIT,
        "band_80": [round(v, 3) for v in iv["80"]],
        "band_90": [round(v, 3) for v in iv["90"]],
        "band_basis": (f"empirical quantiles of log(actual/forecast) over {band['n_folds']} CV folds "
                       f"({band.get('first_target', '?')}..{band.get('last_target', '?')}); "
                       f"CV MAPE {band['cv_mape_pct']:.2f}%"),
        "last_actual": {"month": f"{actual.index[-1]:%Y-%m}", "value": round(float(actual.iloc[-1]), 3)},
        "anchor": {"month": f"{origin - pd.DateOffset(months=kappa):%Y-%m}", "value": round(anchor, 3)},
        "members": {fam: round(anchor * float(np.exp(z)), 3) for fam, z in z_members.items()},
        "model_id": production.model_id(cfg),
        "config_hash": production.config_hash(cfg_path),
        "git_sha": tracking.git_sha(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def forecast_path(out_dir: str | Path, rec: dict) -> Path:
    return Path(out_dir) / f"forecast_{rec['target_month']}_{rec['data_version']}.json"


def append_history(path: str | Path, rec: dict) -> bool:
    """Append one row; False (and no write) when (target_month, data_version) is already there."""
    path = Path(path)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["target_month"] == rec["target_month"] and r["data_version"] == rec["data_version"]:
                    return False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)
    row = {
        "target_month": rec["target_month"], "origin": rec["origin"],
        "data_version": rec["data_version"], "point_forecast": rec["point_forecast"],
        "lo80": rec["band_80"][0], "hi80": rec["band_80"][1],
        "lo90": rec["band_90"][0], "hi90": rec["band_90"][1],
        "last_actual_month": rec["last_actual"]["month"], "last_actual": rec["last_actual"]["value"],
        "model_id": rec["model_id"], "config_hash": rec["config_hash"],
        "git_sha": rec["git_sha"], "created_at": rec["created_at"],
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=HISTORY_COLUMNS).writerow(row)
    return True


def write_outputs(rec: dict, out_dir: str | Path) -> tuple[Path, bool]:
    """forecast_*.json (left untouched if it exists) + the history row. Returns (path, new_row)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = forecast_path(out_dir, rec)
    if not path.exists():
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, append_history(out_dir / HISTORY, rec)


def render(rec: dict) -> str:
    lo80, hi80 = rec["band_80"]
    lo90, hi90 = rec["band_90"]
    return "\n".join([
        f"forecast for {rec['target_month']}   (origin {rec['origin']}, snapshot {rec['data_version']})",
        f"  point   {rec['point_forecast']:>9,.1f}   {rec['unit']}",
        f"  80%     {lo80:>9,.1f} .. {hi80:,.1f}",
        f"  90%     {lo90:>9,.1f} .. {hi90:,.1f}",
        f"  last observed {rec['last_actual']['month']}: {rec['last_actual']['value']:,.1f}",
        f"  members {rec['members']}",
        f"  band    {rec['band_basis']}",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: newest)")
    ap.add_argument("--config", default=str(production.DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(OUT_DIR),
                    help="directory for forecast_*.json and history.csv (default: forecasts/)")
    ap.add_argument("--no-write", action="store_true", help="print the forecast, write nothing")
    a = ap.parse_args(argv)

    snap = Path(a.snapshot) if a.snapshot else latest_snapshot()
    panel, meta = dataset.load_snapshot(snap)
    cfg = production.load(a.config)
    rec = forecast(panel, meta, cfg, data_version=tracking.data_version(snap), cfg_path=a.config)
    print(render(rec))
    if not a.no_write:
        path, new = write_outputs(rec, a.out)
        print(f"wrote {path}   history: {'row added' if new else 'row already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
