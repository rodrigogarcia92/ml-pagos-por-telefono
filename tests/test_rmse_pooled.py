"""Backlog 8.1 -- a true RMSE in reports. One test point per fold made the logged `rmse` a MAE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlflow
from src.model_training import metrics, report, tracking
from src.model_training.train import RunConfig, fit_config, log_config


@pytest.fixture
def store(tmp_path):
    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.create_experiment(f"{tracking.EXPERIMENT_PREFIX}t3_h3",
                             artifact_location=(tmp_path / "art").as_uri())
    yield tmp_path
    mlflow.set_tracking_uri(old)


@pytest.fixture
def logged(store, prod_snapshot):
    cfg = RunConfig("t3", 3, "w2019", "naive_drift", "none", snapshot=prod_snapshot,
                    protocol_version="1.7")
    return fit_config(cfg), log_config(fit_config(cfg))


def test_pooled_rmse_is_the_root_of_the_mean_squared_fold_errors():
    assert metrics.rmse_pooled([3.0, 4.0]) == pytest.approx(np.sqrt((9 + 16) / 2))
    assert metrics.rmse_pooled([3.0, 4.0]) > np.mean([3.0, 4.0])        # a mean of |e| understates it
    assert metrics.rmse_pooled([5.0]) == 5.0                           # one point: RMSE == MAE
    assert metrics.rmse_pooled([3.0, np.nan, 4.0]) == pytest.approx(metrics.rmse_pooled([3.0, 4.0]))
    assert np.isnan(metrics.rmse_pooled([np.nan, np.nan]))
    assert metrics.rmse_pooled(iter([1.0, 1.0])) == 1.0


def test_the_logged_per_fold_rmse_really_is_the_mae(logged):
    fit, _ = logged
    assert all(d["rmse"] == pytest.approx(d["mae"]) for d in fit.per_fold)   # the defect, pinned


def test_new_runs_log_rmse_pooled_on_the_parent(logged):
    fit, res = logged
    run = mlflow.get_run(res.run_id)
    want = float(np.sqrt(np.mean([d["mae"] ** 2 for d in fit.per_fold])))
    assert run.data.metrics["rmse_pooled"] == pytest.approx(want)
    assert run.data.metrics["rmse_pooled"] > run.data.metrics["rmse_mean"] * 0.999   # >= the MAE
    assert run.data.metrics["rmse_mean"] == pytest.approx(run.data.metrics["mae_mean"])


def test_report_shows_rmse_pooled_and_labels_the_old_column(logged):
    fit, _ = logged
    df = report.fetch("t3", 3, protocol="1.7")
    row = df.iloc[0]
    want = float(np.sqrt(np.mean([d["mae"] ** 2 for d in fit.per_fold])))
    assert row["rmse_pooled"] == pytest.approx(want)
    assert "rmse_mean" not in df.columns and "rmse_mean_is_mae" in df.columns
    assert row["rmse_mean_is_mae"] == pytest.approx(row["mae_mean"])
    assert row["rmse_pooled"] >= row["mae_mean"]


def test_it_is_derived_from_the_child_runs_for_runs_that_never_logged_it(logged):
    """Every run already in MLflow predates the metric: no re-run, the children carry the folds."""
    fit, res = logged
    exp = mlflow.get_experiment_by_name(f"{tracking.EXPERIMENT_PREFIX}t3_h3")
    parents = pd.DataFrame({"run_id": [res.run_id]})            # no `metrics.rmse_pooled` column
    got = report.pooled_rmse_from_children(parents, exp.experiment_id)
    want = float(np.sqrt(np.mean([d["mae"] ** 2 for d in fit.per_fold])))
    assert got.iloc[0] == pytest.approx(want)
    # and a NaN in the logged column also falls back to the children
    parents = pd.DataFrame({"run_id": [res.run_id], "metrics.rmse_pooled": [np.nan]})
    assert report.pooled_rmse_from_children(parents, exp.experiment_id).iloc[0] == pytest.approx(want)
