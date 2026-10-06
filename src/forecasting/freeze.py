"""Freeze the selected model into configs/production/t3_ens3.yaml.

    python scripts/freeze_production.py [--snapshot ...] [--db mlflow/mlflow.db]

Read-only on MLflow (SQLite opened mode=ro) and on the snapshot. It creates no run, tunes
nothing and selects nothing: the hyperparameters are READ from the protocol-1.7 ens3 / FS3 /
t3 / w2019 run that Stage A + B already logged.

It also computes the forecast error band, from the CV backtest only: the three members are
refitted with the frozen hyperparameters on the sweep's expanding folds (splits.make_folds on
the CV rows, i.e. without the final HOLDOUT_MONTHS target months), and the ensemble's fold
errors log(actual / forecast) are summarised by their empirical quantiles. The holdout months
are never indexed (asserted), and nothing here prints an error on them.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.forecasting import ensemble, mlflow_ro, production
from src.model_training import dataset, registry, tracking, train
from src.model_training.snapshot import latest_snapshot

TARGET, HORIZON, WINDOW, FEATURE_SET = "t3", 3, "w2019", "FS3_activity"
PROTOCOL = "1.7"
KAPPA = 2

HEADER = """\
# Frozen production model: ens3 on FS3_activity, t3 (h=3), w2019.
# Written by scripts/freeze_production.py -- do not edit by hand; regenerate and re-record.
# Hyperparameters are READ from the MLflow run below (Stage A on the CV period, protocol 1.7),
# the error band is computed on CV folds only. Selection record and the pre-registered holdout
# rule: docs/training_plan.md section 11, marker "PRODUCTION-FREEZE t3_ens3 v1".
"""


def build_config(panel, meta, snapshot_path: str | Path, con: sqlite3.Connection) -> dict:
    version = tracking.data_version(snapshot_path)
    run_id = mlflow_ro.find_parent_run(
        con, model_family=ensemble.ENSEMBLE, feature_set=FEATURE_SET, target_id=TARGET,
        window=WINDOW, stage="cv", data_version=version, protocol_version=PROTOCOL)
    params = mlflow_ro.run_params(con, run_id)
    metrics = mlflow_ro.run_metrics(con, run_id)

    member_hp = ensemble.parse_member_params(params.items())
    encodings = dict(kv.split("=") for kv in params["member_encodings"].split(","))
    if encodings != registry.member_encodings(ensemble.ENSEMBLE):
        raise RuntimeError(f"logged member encodings {encodings} differ from the registry's")
    for fam, hp in member_hp.items():       # fail now, not at forecast time, if a name is stale
        registry.MODELS[fam][0](**hp)

    kappa = ensemble.kappa_of(meta, TARGET)
    if kappa != KAPPA:
        raise RuntimeError(f"snapshot says kappa={kappa} for {TARGET}; this freeze is kappa={KAPPA}")
    columns = train.load_feature_sets()[FEATURE_SET]

    frames = ensemble.member_frames(
        panel, meta, target_id=TARGET, horizon=HORIZON, window=WINDOW, columns=columns)
    fc = ensemble.cv_backtest(frames, member_hp, window=WINDOW, horizon=HORIZON, kappa=kappa)
    holdout_start = ensemble.first_holdout_target(frames, window=WINDOW, horizon=HORIZON,
                                                  kappa=kappa)
    if fc.index.max() >= holdout_start:
        raise AssertionError(f"the error band reached the holdout ({fc.index.max():%Y-%m})")
    band = ensemble.error_band(fc)
    if band["n_folds"] != int(params["n_folds"]):
        raise RuntimeError(
            f"CV backtest has {band['n_folds']} folds, the MLflow run logged {params['n_folds']}")

    return {
        "target_id": TARGET,
        "horizon": HORIZON,
        "kappa": kappa,
        "window": WINDOW,
        "feature_set": FEATURE_SET,
        "columns": list(columns),
        "model_family": ensemble.ENSEMBLE,
        "members": {fam: {"encoding": encodings[fam], "params": hp}
                    for fam, hp in member_hp.items()},
        "mlflow_run_id": run_id,
        "data_version": version,
        "protocol_version": PROTOCOL,
        "selected_on": "cv",
        "selection_record": f"docs/training_plan.md section 11, marker {production.FREEZE_MARKER!r}",
        "cv_mase_mean": float(metrics["mase_mean"]),
        "error_band": {
            "basis": ("empirical quantiles of log(actual / forecast) over the CV backtest folds "
                      "only; the final 12 target months are excluded"),
            **band,
            "first_target": f"{fc.index.min():%Y-%m}",
            "last_target": f"{fc.index.max():%Y-%m}",
        },
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: newest)")
    ap.add_argument("--db", default=str(root / "mlflow" / "mlflow.db"), help="MLflow SQLite backend")
    ap.add_argument("--out", default=str(production.DEFAULT_CONFIG))
    a = ap.parse_args()

    snap = Path(a.snapshot) if a.snapshot else latest_snapshot()
    print(f"snapshot {snap.name} | MLflow {a.db} | output {a.out}")
    panel, meta = dataset.load_snapshot(snap)
    con = mlflow_ro.connect_ro(a.db)
    try:
        print("refitting the three members on the CV folds (1-3 minutes)...")
        cfg = build_config(panel, meta, snap, con)
    finally:
        con.close()
    production.dump(cfg, a.out, HEADER)

    b = cfg["error_band"]
    print(f"run {cfg['mlflow_run_id']}  data_version {cfg['data_version']}")
    for fam, m in cfg["members"].items():
        print(f"  {fam:8s} [{m['encoding']}] {m['params']}")
    print(f"band over {b['n_folds']} CV folds {b['first_target']}..{b['last_target']}: "
          + "  ".join(f"{k}={b[k]:+.4f}" for k in ("p05", "p10", "p50", "p90", "p95"))
          + f"   CV MAPE {b['cv_mape_pct']:.2f}%   (MLflow MASE {cfg['cv_mase_mean']:.3f})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
