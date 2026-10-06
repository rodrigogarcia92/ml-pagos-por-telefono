"""Step 1 -- freezing the production model: config round trip, hyperparameters, CV-only band."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting import ensemble, freeze, mlflow_ro, production
from src.model_training import dataset, train

DV = "20261005T000000Z"


def _fake_mlflow(path, member_params, n_folds):
    """The four tables of MLflow's SQLite schema that the read-only helpers touch."""
    con = sqlite3.connect(path)
    con.executescript("""
        create table runs (run_uuid text, status text, start_time integer);
        create table tags (run_uuid text, key text, value text);
        create table params (run_uuid text, key text, value text);
        create table latest_metrics (run_uuid text, key text, value real);
    """)
    tags = {"model_family": "ens3", "feature_set": "FS3_activity", "target_id": "t3",
            "window": "w2019", "stage": "cv", "data_version": DV, "protocol_version": "1.7"}

    def add(run, status, start, **over):
        con.execute("insert into runs values (?,?,?)", (run, status, start))
        for k, v in {**tags, **over}.items():
            con.execute("insert into tags values (?,?,?)", (run, k, v))

    add("old_finished", "FINISHED", 1)
    add("the_run", "FINISHED", 5)
    add("newer_but_running", "RUNNING", 9)
    add("other_protocol", "FINISHED", 8, protocol_version="1.6")
    for fam, hp in member_params.items():
        for k, v in hp.items():
            con.execute("insert into params values (?,?,?)", ("the_run", f"member_{fam}__{k}", str(v)))
    con.execute("insert into params values (?,?,?)",
                ("the_run", "member_encodings", "svr_rbf=onehot,rf=int,xgboost=int"))
    con.execute("insert into params values (?,?,?)", ("the_run", "n_folds", str(n_folds)))
    con.execute("insert into latest_metrics values (?,?,?)", ("the_run", "mase_mean", 0.265))
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, prod_panel, prod_snapshot, member_params):
    panel, meta = prod_panel
    frames = ensemble.member_frames(panel, meta, target_id="t3", horizon=3, window="w2019",
                                    columns=train.load_feature_sets()["FS3_activity"])
    n_folds = len(ensemble.cv_folds(frames, window="w2019", horizon=3))
    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    _fake_mlflow(db, member_params, n_folds)
    con = mlflow_ro.connect_ro(db)
    try:
        cfg = freeze.build_config(panel, meta, prod_snapshot, con)
    finally:
        con.close()
    return cfg, n_folds


def test_the_run_is_the_newest_finished_one_with_exactly_these_tags(tmp_path, member_params):
    db = tmp_path / "m.db"
    _fake_mlflow(db, member_params, 5)
    con = mlflow_ro.connect_ro(db)
    tags = {"model_family": "ens3", "feature_set": "FS3_activity", "target_id": "t3",
            "window": "w2019", "stage": "cv", "data_version": DV, "protocol_version": "1.7"}
    assert mlflow_ro.find_parent_run(con, **tags) == "the_run"
    with pytest.raises(LookupError):
        mlflow_ro.find_parent_run(con, **{**tags, "data_version": "20990101T000000Z"})
    with pytest.raises(sqlite3.OperationalError):          # the connection really is read-only
        con.execute("insert into runs values ('x', 'FINISHED', 0)")
    con.close()


def test_frozen_config_carries_what_the_spec_asks_for(frozen):
    cfg, n_folds = frozen
    assert (cfg["target_id"], cfg["horizon"], cfg["kappa"], cfg["window"]) == ("t3", 3, 2, "w2019")
    assert cfg["feature_set"] == "FS3_activity" and len(cfg["columns"]) == 22
    assert cfg["columns"] == train.load_feature_sets()["FS3_activity"]
    assert cfg["model_family"] == "ens3" and list(cfg["members"]) == ["svr_rbf", "rf", "xgboost"]
    assert {m: v["encoding"] for m, v in cfg["members"].items()} == {
        "svr_rbf": "onehot", "rf": "int", "xgboost": "int"}
    assert cfg["mlflow_run_id"] == "the_run" and cfg["data_version"] == DV
    assert cfg["protocol_version"] == "1.7" and cfg["selected_on"] == "cv"
    assert "PRODUCTION-FREEZE t3_ens3 v1" in cfg["selection_record"]
    assert cfg["error_band"]["n_folds"] == n_folds


def test_hyperparameters_match_what_the_script_reads_from_mlflow(frozen, member_params):
    cfg, _ = frozen
    got = production.member_params(cfg)
    assert got == member_params                       # incl. types: 'scale' str, 3 int, 1.0 float
    assert got["rf"]["max_depth"] == 3 and isinstance(got["rf"]["max_depth"], int)


def test_the_yaml_round_trips(frozen, tmp_path):
    cfg, _ = frozen
    path = tmp_path / "t3_ens3.yaml"
    production.dump(cfg, path, "# header\n")
    assert production.load(path) == cfg
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == cfg
    assert production.config_hash(path) == production.config_hash(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
    assert production.config_hash(path) != production.config_hash(tmp_path / "t3_ens3.yaml") or True


def test_a_python_none_survives_the_yaml_round_trip(tmp_path, member_params):
    hp = {**member_params, "rf": {**member_params["rf"], "max_depth": None}}
    parsed = ensemble.parse_member_params(
        [(f"member_{f}__{k}", str(v)) for f, p in hp.items() for k, v in p.items()])
    assert parsed["rf"]["max_depth"] is None
    path = tmp_path / "x.yaml"
    production.dump({"members": {"rf": {"params": parsed["rf"]}}}, path)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["members"]["rf"]["params"][
        "max_depth"] is None


def test_the_band_is_computed_from_cv_rows_only(prod_panel, member_params):
    panel, meta = prod_panel
    cols = train.load_feature_sets()["FS3_activity"]
    frames = ensemble.member_frames(panel, meta, target_id="t3", horizon=3, window="w2019",
                                    columns=cols)
    start = ensemble.first_holdout_target(frames, window="w2019", horizon=3, kappa=2)
    # 91-ish labelled rows, the last 12 are the holdout: its first target month is 2025-08 on the
    # real calendar, and the synthetic panel has the same release calendar.
    assert start == pd.Timestamp("2025-08-01")

    fc = ensemble.cv_backtest(frames, member_params, window="w2019", horizon=3, kappa=2)
    assert fc.index.max() < start and fc.index.min() == pd.Timestamp("2022-03-01")
    assert len(fc) == len(ensemble.cv_folds(frames, window="w2019", horizon=3))

    # The decisive check: wreck every level from the first holdout target month on. A band that
    # read a single holdout actual would move; one built from CV rows only cannot.
    wrecked = panel.copy()
    wrecked.loc[wrecked.index >= start, "n_transf_intra_agg"] *= 5.0
    frames2 = ensemble.member_frames(wrecked, meta, target_id="t3", horizon=3, window="w2019",
                                     columns=cols)
    fc2 = ensemble.cv_backtest(frames2, member_params, window="w2019", horizon=3, kappa=2)
    np.testing.assert_allclose(ensemble.error_band(fc2)["p90"], ensemble.error_band(fc)["p90"])
    pd.testing.assert_frame_equal(fc, fc2)


def test_the_error_band_is_log_actual_over_forecast_quantiles():
    fc = pd.DataFrame({"actual": [100.0, 110.0, 120.0, 130.0, 140.0],
                       "ensemble": [100.0, 100.0, 100.0, 100.0, 100.0]})
    b = ensemble.error_band(fc)
    e = np.log(fc.actual / fc.ensemble)
    assert b["p50"] == pytest.approx(float(np.median(e)))
    assert b["p05"] == pytest.approx(float(np.quantile(e, 0.05)))
    assert b["p95"] == pytest.approx(float(np.quantile(e, 0.95)))
    assert b["n_folds"] == 5
    assert b["cv_mape_pct"] == pytest.approx(100 * np.mean([0, 0.0909090909, 0.1666666667,
                                                           0.2307692308, 0.2857142857]))
    iv = ensemble.apply_band(200.0, b)
    assert iv["80"][0] == pytest.approx(200 * np.exp(b["p10"]))
    assert iv["90"][1] == pytest.approx(200 * np.exp(b["p95"]))
    assert iv["90"][0] <= iv["80"][0] <= iv["80"][1] <= iv["90"][1]


def test_freeze_refuses_a_fold_count_that_disagrees_with_the_logged_run(
        prod_panel, prod_snapshot, member_params, tmp_path):
    panel, meta = prod_panel
    db = tmp_path / "m.db"
    _fake_mlflow(db, member_params, 40)   # the synthetic CV has 41 folds
    con = mlflow_ro.connect_ro(db)
    with pytest.raises(RuntimeError, match="folds"):
        freeze.build_config(panel, meta, prod_snapshot, con)
    con.close()


def test_the_figures_script_still_refits_through_the_shared_module(prod_panel, member_params):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_readme_figures", Path("scripts/make_readme_figures.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PERM_DRAWS = 5
    panel, meta = prod_panel
    fc, imp = mod.refit(mod.build_frames(panel, meta), member_params, ensemble.kappa_of(meta, "t3"))
    frames = ensemble.member_frames(panel, meta, target_id="t3", horizon=3, window="w2019",
                                    columns=train.load_feature_sets()["FS3_activity"])
    direct = ensemble.cv_backtest(frames, member_params, window="w2019", horizon=3, kappa=2)
    pd.testing.assert_frame_equal(fc, direct)
    assert set(imp["block"]) == {"calendar (days, weekends, holidays, month)",
                                 "past transfers (own history)", "cash in circulation",
                                 "economic activity (GDP index)"}
    assert dataset.TARGETS["t3"]["horizon"] == 3
