"""Step 2 -- the one-time holdout: s6 spec, frozen hyperparameters in Stage C, and every guard.

Offline: a synthetic snapshot and a throwaway SQLite MLflow store (never the server in .env).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.forecasting import holdout, production
from src.model_training import registry, splits, sweep, tracking
from src.model_training.train import RunConfig, context_for, fit_config

SPEC = Path("configs/sweeps/s6_holdout.yaml")
DV = "20261005T000000Z"


@pytest.fixture
def mlflow_store(tmp_path):
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment(f"{tracking.EXPERIMENT_PREFIX}t3_h3",
                             artifact_location=(tmp_path / "art").as_uri())
    yield tmp_path
    mlflow.set_tracking_uri(old)


@pytest.fixture
def frozen_yaml(tmp_path, member_params):
    cfg = {
        "target_id": "t3", "horizon": 3, "kappa": 2, "window": "w2019",
        "feature_set": "FS3_activity", "columns": [], "model_family": "ens3",
        "members": {f: {"encoding": registry.default_encoding(f), "params": p}
                    for f, p in member_params.items()},
        "mlflow_run_id": "abc", "data_version": DV, "protocol_version": "1.7",
        "selected_on": "cv",
        "error_band": {"p05": -.05, "p10": -.04, "p50": 0.0, "p90": .1, "p95": .11,
                       "n_folds": 41, "cv_mape_pct": 4.4},
    }
    path = tmp_path / "t3_ens3.yaml"
    production.dump(cfg, path)
    return path


@pytest.fixture
def spec_path(tmp_path, frozen_yaml):
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    spec["frozen"] = {"ens3": str(frozen_yaml)}
    path = tmp_path / "s6.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture
def plan_with_marker(tmp_path):
    p = tmp_path / "plan.md"
    p.write_text(f"| 2026-10-06 | **{production.FREEZE_MARKER}** | ... |\n", encoding="utf-8")
    return p


@pytest.fixture
def plan_without_marker(tmp_path):
    p = tmp_path / "plan_no.md"
    p.write_text("| 2026-10-06 | something else |\n", encoding="utf-8")
    return p


def _no_fit(monkeypatch):
    def boom(cfg):
        raise AssertionError("a holdout fit was started")
    monkeypatch.setattr(sweep, "_safe_fit", boom)


def _holdout_runs():
    exp = mlflow.get_experiment_by_name(f"{tracking.EXPERIMENT_PREFIX}t3_h3")
    return mlflow.search_runs([exp.experiment_id], filter_string="tags.stage = 'holdout'")


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
def test_s6_expands_to_exactly_the_two_parents():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    runs = sweep.expand(spec, f"data/processed/panel_{DV}.parquet")
    assert len(runs) == 2
    assert {(r.model_family, r.feature_set) for r in runs} == {
        ("ens3", "FS3_activity"), ("naive_drift", "none")}
    assert {(r.target_id, r.horizon, r.window, r.stage, r.protocol_version) for r in runs} == {
        ("t3", 3, "w2019", "holdout", "1.7")}
    ens = next(r for r in runs if r.model_family == "ens3")
    assert ens.frozen_params == production.member_params(production.load(production.DEFAULT_CONFIG))
    assert next(r for r in runs if r.model_family == "naive_drift").frozen_params is None


def test_a_spec_with_a_third_parent_or_the_wrong_stage_is_refused(spec_path, tmp_path,
                                                                  prod_snapshot):
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    for bad in ({**spec, "models": [*spec["models"], "naive_last"]},
                {**spec, "targets": ["t3", "t10"]}):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="exactly the two parents|frozen ens3 config"):
            holdout.load_runs(p, prod_snapshot)
    p = tmp_path / "cv.yaml"
    p.write_text(yaml.safe_dump({k: v for k, v in spec.items() if k != "frozen"} | {"stage": "cv"}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="exactly the two parents"):
        holdout.load_runs(p, prod_snapshot)


def test_a_frozen_config_from_another_data_version_is_refused(spec_path, prod_snapshot, tmp_path):
    other = tmp_path / "panel_20270105T000000Z.parquet"
    other.write_bytes(Path(prod_snapshot).read_bytes())
    with pytest.raises(ValueError, match="frozen on data_version"):
        holdout.load_runs(spec_path, str(other))


# --------------------------------------------------------------------------- #
# Stage C uses the frozen hyperparameters and tunes nothing
# --------------------------------------------------------------------------- #
def test_stage_c_with_frozen_params_runs_no_stage_a_and_stays_on_the_holdout(
        prod_snapshot, member_params):
    cfg = RunConfig("t3", 3, "w2019", "ens3", "FS3_activity", stage="holdout",
                    snapshot=prod_snapshot, protocol_version="1.7",
                    frozen_params=member_params, frozen_source="frozen:test")
    fit = fit_config(cfg)
    assert fit.best_params == member_params and len(fit.tuning_table) == 0
    assert len(fit.eval_folds) == 12
    assert min(int(f.test[0]) for f in fit.eval_folds) == fit.n_cv      # first holdout origin
    assert fit.eval_folds[0].train.max() == fit.n_cv - 1 - 2            # purge h-1 = 2
    assert fit.dates[0] + pd.DateOffset(months=1) == pd.Timestamp("2025-08-01")


def test_frozen_params_are_refused_outside_stage_c(prod_snapshot, member_params):
    cfg = RunConfig("t3", 3, "w2019", "ens3", "FS3_activity", stage="cv", snapshot=prod_snapshot,
                    protocol_version="1.7", frozen_params=member_params)
    with pytest.raises(ValueError, match="Stage C"):
        fit_config(cfg)


def test_holdout_folds_are_the_final_twelve_months_expanding_and_purged():
    folds = splits.holdout_folds(91, n_cv=79, horizon=3)
    assert len(folds) == 12 and folds[0].test[0] == 79 and folds[-1].test[0] == 90
    assert all(f.train.max() == f.test[0] - 3 for f in folds)


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #
def test_without_confirm_it_prints_the_plan_and_exits_0_running_nothing(
        monkeypatch, capsys, mlflow_store, spec_path, prod_snapshot, plan_with_marker):
    _no_fit(monkeypatch)
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot,
                       "--plan", str(plan_with_marker)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ens3__FS3_activity__w2019" in out and "naive_drift__none__w2019" in out
    assert "Without --confirm nothing is run" in out and "[NO ] (b) --confirm" in out
    assert len(_holdout_runs()) == 0


def test_guard_a_refuses_when_the_freeze_marker_is_missing(
        monkeypatch, capsys, mlflow_store, spec_path, prod_snapshot, plan_without_marker):
    _no_fit(monkeypatch)
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
                       "--plan", str(plan_without_marker)])
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSED" in out and "MISSING" in out
    assert len(_holdout_runs()) == 0


def test_guard_a_refuses_when_the_plan_file_does_not_exist(
        monkeypatch, capsys, mlflow_store, spec_path, prod_snapshot, tmp_path):
    _no_fit(monkeypatch)
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
                       "--plan", str(tmp_path / "nope.md")])
    assert rc == 2 and "REFUSED" in capsys.readouterr().out


def test_guard_c_refuses_when_mlflow_already_has_a_finished_holdout(
        monkeypatch, capsys, mlflow_store, spec_path, prod_snapshot, plan_with_marker):
    _no_fit(monkeypatch)
    monkeypatch.setattr(holdout, "finished_holdouts",
                        lambda configs, snapshot: ["ens3__FS3_activity__w2019"])
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
                       "--plan", str(plan_with_marker)])
    out = capsys.readouterr().out
    assert rc == 2 and "ALREADY EVALUATED" in out


def test_guard_c_refuses_when_mlflow_cannot_be_checked_and_confirm_is_given(
        monkeypatch, capsys, mlflow_store, spec_path, prod_snapshot, plan_with_marker):
    _no_fit(monkeypatch)

    def down(configs, snapshot):
        raise ConnectionError("tracking server unreachable")
    monkeypatch.setattr(holdout, "finished_holdouts", down)
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
                       "--plan", str(plan_with_marker)])
    assert rc == 2 and "could not check MLflow" in capsys.readouterr().out
    # ...but a plain preview (no --confirm) still exits 0 and says it could not check
    rc = holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot,
                       "--plan", str(plan_with_marker)])
    assert rc == 0


def test_the_real_run_logs_two_holdout_parents_once_and_never_twice(
        capsys, mlflow_store, spec_path, prod_snapshot, plan_with_marker, member_params):
    argv = ["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
            "--plan", str(plan_with_marker)]
    assert holdout.main(argv) == 0
    out = capsys.readouterr().out

    runs = _holdout_runs()
    parents = runs                                  # children carry no stage tag
    assert set(parents["tags.model_family"]) == {"ens3", "naive_drift"} and len(parents) == 2
    assert set(parents["tags.stage"]) == {"holdout"}
    assert set(parents["tags.protocol_version"]) == {"1.7"}
    ens = parents[parents["tags.model_family"] == "ens3"].iloc[0]
    assert ens["params.hp_source"].startswith("frozen:")
    assert ens["params.member_rf__n_estimators"] == str(member_params["rf"]["n_estimators"])
    assert ens["params.member_svr_rbf__C"] == str(member_params["svr_rbf"]["C"])
    assert ens["params.n_folds"] == "12" and ens["params.holdout_months"] == "12"
    drift = parents[parents["tags.model_family"] == "naive_drift"].iloc[0]
    assert pd.isna(drift["params.hp_source"])             # Stage A on the CV rows, nothing frozen

    assert "HOLDOUT RESULT" in out and "paired per-month difference" in out
    assert "Paste into docs/methodology.md (decision log)" in out and "Paste into README.md" in out
    assert "2025-08 .. 2026-07" in out and "VERDICT" in out

    # the second attempt is refused by guard (c) and starts nothing
    n = len(runs)
    assert holdout.main(argv) == 2
    assert "ALREADY EVALUATED" in capsys.readouterr().out
    assert len(_holdout_runs()) == n


def test_the_resume_key_sees_the_finished_holdout_runs(mlflow_store, spec_path, prod_snapshot,
                                                       plan_with_marker):
    _, snap, configs = holdout.load_runs(spec_path, prod_snapshot)
    assert holdout.finished_holdouts(configs, snap) == []
    assert holdout.main(["--config", str(spec_path), "--snapshot", prod_snapshot, "--confirm",
                         "--plan", str(plan_with_marker)]) == 0
    assert sorted(holdout.finished_holdouts(configs, snap)) == [
        "ens3__FS3_activity__w2019", "naive_drift__none__w2019"]
    # ...and a CV run of the same configuration does not count as a holdout
    cv = context_for(RunConfig("t3", 3, "w2019", "ens3", "FS3_activity", stage="cv",
                               snapshot=snap, protocol_version="1.7"), snap)
    assert not tracking.already_done(cv)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def test_paired_difference_and_its_standard_error():
    a, b = pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series([2.0, 2.5, 3.0, 6.0])
    mean, se = holdout.paired(a, b)
    d = np.array([-1.0, -0.5, 0.0, -2.0])
    assert mean == pytest.approx(d.mean()) and se == pytest.approx(d.std(ddof=1) / 2)


def _fake(c_mase, b_mase):
    return {"n_months": 12, "first_month": "2025-08", "last_month": "2026-07",
            "data_version": DV, "run_ids": {"ens3": "r1", "naive_drift": "r2"},
            "ens3": {"mase": c_mase, "mape": 4.0}, "naive_drift": {"mase": b_mase, "mape": 5.0},
            "diff_mase": (c_mase - b_mase, 0.05), "diff_mape": (-1.0, 0.4),
            "challenger_wins": c_mase < b_mase}


def test_the_report_states_the_pre_registered_fallback_when_ens3_loses():
    text = holdout.render(_fake(0.40, 0.30))
    assert "does NOT beat naive_drift" in text and "production fallback" in text
    assert "MASE 0.400 vs 0.300" in text
    won = holdout.render(_fake(0.20, 0.30))
    assert "ens3 beats naive_drift" in won
    assert "NOT" not in won.split("VERDICT")[1].split("\n")[0]
