"""Protocol 1.7: T9 (ens3), reproducibility, protocol tagging, and the s1b / s5 sweeps.

plan 7.4, 9.2. Offline: a synthetic panel written to a tmp snapshot, and a
throwaway SQLite MLflow store for the one test that logs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.model_training import registry, sweep, tracking
from src.model_training.train import (
    RunConfig,
    _score_fold,
    _tune,
    check_protocol,
    context_for,
    fit_config,
    hyperparameter_params,
)

SPECS = Path("configs/sweeps")

# Small grids so the whole file runs in seconds. They are OVERRIDES of the search space only;
# nothing about the ensemble logic changes.
SMALL = {
    "svr_rbf": {"C": [1, 10], "gamma": ["scale"], "epsilon": [0.01]},
    "rf": {"n_estimators": [20], "max_depth": [3, None], "min_samples_leaf": [3],
           "max_features": [0.6]},
    "xgboost": {"max_depth": [2], "learning_rate": [0.1], "n_estimators": [30, 60],
                "subsample": [1.0], "colsample_bytree": [0.8], "min_child_weight": [1],
                "reg_lambda": [1]},
}


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    """panel_*.parquet + series_meta_*.csv with the FS3 series and a positive Trends index."""
    rng = np.random.default_rng(29)
    idx = pd.date_range("2013-01-01", "2026-09-01", freq="MS")
    n = len(idx)
    month = idx.month.to_numpy() - 1
    p = pd.DataFrame({
        "n_transf_intra_agg": np.exp(np.linspace(np.log(10), np.log(700), n)
                                     + 0.05 * np.sin(2 * np.pi * month / 12)
                                     + 0.015 * rng.standard_normal(n)),
        "circulante": np.exp(np.linspace(np.log(40000), np.log(90000), n)
                             + 0.01 * rng.standard_normal(n)),
        "pbi_idx": 100 + np.linspace(0, 60, n) + rng.standard_normal(n),
        "gt_yape_plin": np.nan,
    }, index=idx)
    k = int((idx >= "2017-01-01").sum())
    p.loc["2017-01-01":, "gt_yape_plin"] = np.clip(
        8 * np.exp(0.03 * np.arange(k) + 0.1 * rng.standard_normal(k)), 6, 190)
    p.loc["2026-08-01":, "n_transf_intra_agg"] = np.nan
    meta = pd.DataFrame({"col_name": p.columns, "kappa": [2, 1, 2, 0],
                         "transform": ["log_diff"] * 4})
    d = tmp_path_factory.mktemp("snap")
    path = d / "panel_20261005T000000Z_gt20261005.parquet"
    p.to_parquet(path)
    meta.to_csv(d / "series_meta_20261005T000000Z_gt20261005.csv", index=False)
    return str(path)


def _cfg(snapshot, model="ens3", fs="FS3_activity", target="t10", h=5, tune=None):
    return RunConfig(target, h, "w2019", model, fs, snapshot=snapshot,
                     tune=dict(SMALL) if tune is None else tune)


@pytest.fixture(scope="module")
def fitted(snapshot):
    return fit_config(_cfg(snapshot))


# --------------------------------------------------------------------------- #
# T9 -- the forecast IS the mean of the members' z-forecasts
# --------------------------------------------------------------------------- #
def test_t9_ens3_is_registered_as_three_members_with_their_own_encodings():
    assert registry.ENSEMBLES["ens3"] == ("svr_rbf", "rf", "xgboost")
    assert registry.member_encodings("ens3") == {"svr_rbf": "onehot", "rf": "int", "xgboost": "int"}
    assert registry.default_encoding("ens3") == "mixed"
    assert "ens3" not in registry.NAIVE_FAMILIES


def test_t9_ensemble_forecast_equals_the_mean_of_the_members_z_forecasts(fitted):
    cfg, frame = fitted.cfg, fitted.frame
    best = fitted.best_params
    assert set(best) == {"svr_rbf", "rf", "xgboost"}

    for fold in fitted.eval_folds[:6] + fitted.eval_folds[-3:]:
        ens = registry.build("ens3", best, horizon=cfg.horizon)
        _, yhat_ens = _score_fold(ens, frame, fold.train, fold.test)

        # The members, each scored through the ORDINARY single-model path (own encoding,
        # own scaler), then averaged on z: that is what ens3 is defined to be (plan 7.4).
        anchor = frame.ctx["anchor"].to_numpy()[fold.test]
        z = []
        for fam in registry.ENSEMBLES["ens3"]:
            member = registry.build(fam, best[fam], horizon=cfg.horizon)
            enc_frame = frame.for_encoding(registry.default_encoding(fam))
            _, yhat_m = _score_fold(member, enc_frame, fold.train, fold.test)
            z.append(np.log(yhat_m / anchor))
        np.testing.assert_allclose(yhat_ens, anchor * np.exp(np.mean(z, axis=0)), rtol=1e-9)
        # equal weights, on z -- NOT a mean of levels (they differ, by Jensen)
        assert not np.allclose(yhat_ens, np.mean([anchor * np.exp(v) for v in z], axis=0), rtol=1e-12)
        # ...and the adapter kept each member's own z-forecast
        np.testing.assert_allclose(np.mean(list(ens.member_z_.values()), axis=0),
                                   np.log(yhat_ens / anchor), rtol=1e-9, atol=1e-12)


def test_t9_members_see_their_own_encoding_of_the_same_rows(fitted):
    f = fitted.frame
    onehot = f.for_encoding("onehot")
    assert f.encoding == "int" and "cal_month" in f.X.columns
    assert onehot.encoding == "onehot" and "cal_month" not in onehot.X.columns
    assert onehot.X.shape[1] == f.X.shape[1] + 10                    # cal_month -> 11 dummies
    assert onehot.X.index.equals(f.X.index) and onehot.y.equals(f.y) and onehot.ctx.equals(f.ctx)
    assert fitted.encoding_tag == "mixed"


def test_t9_members_are_tuned_independently(snapshot, fitted):
    """Each member's winner is exactly what a stand-alone Stage A picks, and changing ANOTHER
    member's grid does not move it."""
    cfg, frame = fitted.cfg, fitted.frame
    for fam in registry.ENSEMBLES["ens3"]:
        solo = replace(cfg, model_family=fam, tune=SMALL[fam])
        best_solo, table_solo = _tune(frame.for_encoding(registry.default_encoding(fam)), solo,
                                      fitted.n_cv)
        assert fitted.best_params[fam] == best_solo, fam
        mine = fitted.tuning_table[fitted.tuning_table["member"] == fam]
        assert len(mine) == len(table_solo)

    other = fit_config(_cfg(snapshot, tune={**SMALL, "xgboost": {**SMALL["xgboost"],
                                                                 "n_estimators": [30],
                                                                 "max_depth": [2, 3]}}))
    assert other.best_params["svr_rbf"] == fitted.best_params["svr_rbf"]
    assert other.best_params["rf"] == fitted.best_params["rf"]
    pd.testing.assert_frame_equal(
        other.tuning_table[other.tuning_table["member"] == "svr_rbf"].reset_index(drop=True),
        fitted.tuning_table[fitted.tuning_table["member"] == "svr_rbf"].reset_index(drop=True))


def test_t9_tuning_table_has_a_member_column_and_one_block_per_member(fitted):
    t = fitted.tuning_table
    assert t.columns[0] == "member"
    assert set(t["member"]) == {"svr_rbf", "rf", "xgboost"}
    assert (t.groupby("member")["n_inner_folds"].first() == 10).all()   # same inner folds for all
    assert len(t[t["member"] == "svr_rbf"]) == 2 and len(t[t["member"] == "xgboost"]) == 2


def test_member_hyperparameters_are_logged_as_member_family_param(fitted):
    p = hyperparameter_params(fitted.cfg, fitted.best_params)
    assert set(p) >= {"member_svr_rbf__C", "member_rf__max_depth", "member_xgboost__n_estimators"}
    assert not any(k.startswith("hp_") for k in p)
    single = hyperparameter_params(replace(fitted.cfg, model_family="ridge"), {"alpha": 1.0})
    assert single == {"hp_alpha": 1.0}


def test_every_fold_carries_each_members_own_mase(fitted):
    for m in fitted.per_fold:
        assert {"mase_svr_rbf", "mase_rf", "mase_xgboost", "mase"} <= set(m)
        assert all(np.isfinite(m[k]) for k in ("mase_svr_rbf", "mase_rf", "mase_xgboost", "mase"))


# --------------------------------------------------------------------------- #
# Reproducibility: same seed, same output
# --------------------------------------------------------------------------- #
def test_ens3_is_reproducible_same_seed_same_output(snapshot, fitted):
    again = fit_config(_cfg(snapshot))
    assert again.best_params == fitted.best_params
    np.testing.assert_array_equal(np.asarray(again.preds), np.asarray(fitted.preds))
    assert [f.test.tolist() for f in again.eval_folds] == [f.test.tolist() for f in fitted.eval_folds]
    for a, b in zip(again.per_fold, fitted.per_fold, strict=True):
        assert a == b


def test_t10_fold_count_through_the_real_pipeline(fitted):
    """39 folds at h=5 (plan 3), through fit_config rather than splits alone."""
    assert len(fitted.eval_folds) == 39


# --------------------------------------------------------------------------- #
# FS3_gt end to end, and the gt guard
# --------------------------------------------------------------------------- #
def test_fs3_gt_runs_through_the_pipeline_with_25_tree_columns(snapshot):
    fit = fit_config(_cfg(snapshot, model="xgboost", fs="FS3_gt", target="t10", h=5,
                          tune={"xgboost": SMALL["xgboost"]}))
    assert fit.frame.X.shape[1] == 25
    assert {"gt_d0", "gt_ma3", "gt_ma12"} <= set(fit.frame.X.columns)
    assert len(fit.eval_folds) <= 39            # gt_ma12 needs 2017+ history: fewer rows is allowed


# --------------------------------------------------------------------------- #
# Protocol tag
# --------------------------------------------------------------------------- #
def test_protocol_17_runs_are_tagged_17_and_not_resumable_from_16(snapshot):
    c17 = _cfg(snapshot, model="naive_drift", fs="none", target="t3", h=3)
    c16 = replace(c17, protocol_version="1.6")
    t17, t16 = context_for(c17, snapshot).tags(), context_for(c16, snapshot).tags()
    assert (t17["protocol_version"], t16["protocol_version"]) == ("1.7", "1.6")
    assert {k: v for k, v in t17.items() if k != "protocol_version"} == \
        {k: v for k, v in t16.items() if k != "protocol_version"}
    assert context_for(replace(c17, model_family="ens3", feature_set="FS3_gt"), snapshot
                       ).tags()["encoding"] == "mixed"


def test_a_16_run_cannot_use_a_17_target_model_or_feature_set(snapshot):
    for kw in ({"target": "t10", "h": 5}, {"model": "ens3"}, {"fs": "FS3_gt", "model": "ridge"}):
        cfg = replace(_cfg(snapshot, **kw), protocol_version="1.6")
        with pytest.raises(ValueError, match="exist only from protocol 1.7"):
            check_protocol(cfg)
        with pytest.raises(ValueError, match="exist only from protocol 1.7"):
            fit_config(cfg)
    with pytest.raises(ValueError, match="protocol_version"):
        tracking.RunContext("t3", 3, "w2019", "none", "naive_last", "none", "cv", "panel_x.parquet",
                            protocol_version="1.5")


def test_horizons_5_and_6_are_accepted_by_the_run_context():
    for h in (1, 3, 5, 6):
        tracking.RunContext("t3", h, "w2019", "none", "naive_last", "none", "cv", "panel_x.parquet")
    with pytest.raises(ValueError, match="horizon"):
        tracking.RunContext("t3", 4, "w2019", "none", "naive_last", "none", "cv", "panel_x.parquet")


# --------------------------------------------------------------------------- #
# The two sweeps (plan 7.4)
# --------------------------------------------------------------------------- #
def _expand(name):
    spec = yaml.safe_load((SPECS / name).read_text(encoding="utf-8"))
    return spec, sweep.expand(spec, "data/processed/panel_x.parquet")


def test_s1b_expands_to_15_naive_parents_under_protocol_17():
    spec, runs = _expand("s1b_baselines_h56.yaml")
    assert spec["protocol_version"] == "1.7" and sweep.protocol_of(spec) == "1.7"
    assert len(runs) == 15
    assert {(r.target_id, r.horizon) for r in runs} == {("t3", 3), ("t10", 5), ("t11", 6)}
    assert {r.window for r in runs} == {"w2019"} and {r.stage for r in runs} == {"cv"}
    assert {r.model_family for r in runs} == {"naive_last", "naive_drift", "naive_seasonal",
                                              "naive_seasdrift", "naive_calendar"}
    assert {r.feature_set for r in runs} == {"none"}
    assert {r.protocol_version for r in runs} == {"1.7"}
    assert len({(r.target_id, r.model_family) for r in runs}) == 15


def test_s5_expands_to_12_parents_three_targets_four_families_fs3_only():
    # Trends gate G1/G3 FAILED (plan 9.2, 2026-10-06): FS3_gt is not swept; the code stays.
    spec, runs = _expand("s5_trends_ens.yaml")
    assert spec["protocol_version"] == "1.7"
    assert len(runs) == 12
    assert len({(r.target_id, r.model_family, r.feature_set) for r in runs}) == 12
    assert {(r.target_id, r.horizon) for r in runs} == {("t3", 3), ("t10", 5), ("t11", 6)}
    assert {r.model_family for r in runs} == {"svr_rbf", "rf", "xgboost", "ens3"}
    assert {r.feature_set for r in runs} == {"FS3_activity"}
    assert {r.window for r in runs} == {"w2019"} and {r.protocol_version for r in runs} == {"1.7"}
    # every cell of the 3 x 4 design, once
    assert {(t, m, "FS3_activity") for t in ("t3", "t10", "t11")
            for m in ("svr_rbf", "rf", "xgboost", "ens3")} == {
        (r.target_id, r.model_family, r.feature_set) for r in runs}


def test_mlflow_experiment_names_and_run_names_of_the_new_sweeps():
    _, s1b = _expand("s1b_baselines_h56.yaml")
    _, s5 = _expand("s5_trends_ens.yaml")
    snap = "data/processed/panel_x.parquet"
    exps = {context_for(r, snap).experiment for r in s1b + s5}
    assert exps == {f"{tracking.EXPERIMENT_PREFIX}t3_h3", f"{tracking.EXPERIMENT_PREFIX}t10_h5",
                    f"{tracking.EXPERIMENT_PREFIX}t11_h6"}
    names = {context_for(r, snap).run_name for r in s5}
    assert "ens3__FS3_activity__w2019" in names and "svr_rbf__FS3_activity__w2019" in names


def test_a_17_target_in_a_spec_without_the_protocol_key_is_refused():
    spec = {"name": "x", "targets": ["t10"], "windows": ["w2019"], "models": ["naive_last"]}
    with pytest.raises(ValueError, match="exist only from protocol 1.7"):
        sweep.expand(spec, "data/processed/panel_x.parquet")
    with pytest.raises(ValueError, match="protocol_version"):
        sweep.protocol_of({"protocol_version": "9.9"})


# --------------------------------------------------------------------------- #
# End to end: one ens3 parent through MLflow (throwaway SQLite store)
# --------------------------------------------------------------------------- #
def test_ens3_logs_its_members_hyperparameters_tags_and_member_metrics(tmp_path, snapshot, fitted):
    import mlflow

    from src.model_training.train import log_config

    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    try:
        exp = context_for(fitted.cfg, snapshot).experiment
        mlflow.create_experiment(exp, artifact_location=(tmp_path / "artifacts").as_uri())
        res = log_config(fitted)
        run = mlflow.get_run(res.run_id)
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        assert tags["model_family"] == "ens3" and tags["encoding"] == "mixed"
        assert tags["protocol_version"] == "1.7" and tags["feature_set"] == "FS3_activity"
        assert tags["data_version"] == "20261005T000000Z_gt20261005"
        assert params["member_encodings"] == "svr_rbf=onehot,rf=int,xgboost=int"
        for fam in registry.ENSEMBLES["ens3"]:
            assert any(k.startswith(f"member_{fam}__") for k in params), fam
        assert not any(k.startswith("hp_") for k in params)
        assert {"mase_mean", "mase_svr_rbf_mean", "mase_rf_mean", "mase_xgboost_mean",
                "mase_svr_rbf_std"} <= set(metrics)
        children = mlflow.search_runs(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{res.run_id}'", output_format="list")
        assert len(children) == 39
        assert "mase_rf" in children[0].data.metrics
        files = {a.path for a in mlflow.MlflowClient().list_artifacts(res.run_id)}
        assert "tuning_results.csv" in files
        assert tracking.already_done(context_for(fitted.cfg, snapshot))
        assert not tracking.already_done(context_for(replace(fitted.cfg, protocol_version="1.6"),
                                                     snapshot))
    finally:
        mlflow.set_tracking_uri(old)
