"""Step 5 -- monthly refresh orchestrator, with the stages mocked (and the cheap ones real)."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from src.data_collection import fetch_target_series
from src.forecasting import predict, production, refresh
from src.model_training import registry

FS3 = None


def _cfg_file(tmp_path, member_params):
    from src.model_training import train

    cfg = {
        "target_id": "t3", "horizon": 3, "kappa": 2, "window": "w2019",
        "feature_set": "FS3_activity", "columns": train.load_feature_sets()["FS3_activity"],
        "model_family": "ens3",
        "members": {f: {"encoding": registry.default_encoding(f), "params": p}
                    for f, p in member_params.items()},
        "mlflow_run_id": "x", "data_version": "20261005T000000Z", "protocol_version": "1.7",
        "selected_on": "cv",
        "error_band": {"p05": -.05, "p10": -.04, "p50": 0.0, "p90": .1, "p95": .11,
                       "n_folds": 41, "cv_mape_pct": 4.4},
    }
    path = tmp_path / "cfg.yaml"
    production.dump(cfg, path)
    return path


@pytest.fixture
def ctx(tmp_path, prod_snapshot, member_params, monkeypatch):
    monkeypatch.setenv("DBT_BIN", str(tmp_path / "fake-dbt"))
    return refresh.Ctx(out_dir=tmp_path / "forecasts", config=_cfg_file(tmp_path, member_params),
                       snapshot=Path(prod_snapshot), root=tmp_path)


class FakeRun:
    """Stands in for refresh.run_subprocess: records every call, answers from a script."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail or {}

    def __call__(self, cmd, cwd=None):
        self.calls.append((list(cmd), cwd))
        for key, (code, out, err) in self.fail.items():
            if key in " ".join(str(c) for c in cmd):
                return subprocess.CompletedProcess(cmd, code, out, err)
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")


def _summary(ctx):
    return json.loads((ctx.out_dir / refresh.LAST_REFRESH).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Plan and dry run
# --------------------------------------------------------------------------- #
def test_the_plan_for_each_entry_point():
    assert refresh.plan(False, None) == list(refresh.STAGES)
    assert refresh.plan(True, None) == ["predict", "monitor"]
    assert refresh.plan(False, "dbt") == ["dbt", "snapshot", "predict", "monitor"]
    assert refresh.plan(False, "monitor") == ["monitor"]
    assert refresh.STAGES == ("fetch", "load", "dbt", "snapshot", "predict", "monitor")
    with pytest.raises(SystemExit):                                  # mutually exclusive
        refresh.main(["--skip-etl", "--from-stage", "dbt", "--dry-run"])


def test_dry_run_prints_the_plan_runs_nothing_and_writes_nothing(
        monkeypatch, tmp_path, capsys, prod_snapshot, member_params):
    cfg = _cfg_file(tmp_path, member_params)
    out = tmp_path / "forecasts"
    monkeypatch.setenv("DBT_BIN", "/opt/dbt/bin/dbt")
    monkeypatch.setattr(refresh, "run_subprocess",
                        lambda *a, **k: pytest.fail("dry-run started a subprocess"))
    monkeypatch.setattr(refresh, "run_stage", lambda *a, **k: pytest.fail("dry-run ran a stage"))
    assert refresh.main(["--dry-run", "--out-dir", str(out), "--config", str(cfg),
                         "--snapshot", prod_snapshot]) == 0
    text = capsys.readouterr().out
    for stage in refresh.STAGES:
        assert stage in text
    assert "src.data_collection.fetch_target_series" in text and "load_to_bigquery" in text
    assert "/opt/dbt/bin/dbt build" in text and "[cwd pagos_dbt]" in text
    assert "src.model_training.snapshot" in text and "--dry-run: nothing run" in text
    assert "it WOULD forecast" in text
    assert not out.exists()                      # no forecasts/, no last_refresh.json


def test_dry_run_skip_etl_lists_only_the_forecast_stages(monkeypatch, tmp_path, capsys,
                                                         prod_snapshot, member_params):
    cfg = _cfg_file(tmp_path, member_params)
    assert refresh.main(["--dry-run", "--skip-etl", "--out-dir", str(tmp_path / "f"),
                         "--config", str(cfg), "--snapshot", prod_snapshot]) == 0
    text = capsys.readouterr().out
    assert "predict" in text and "monitor" in text and "fetch" not in text.split("predict preview")[0]


# --------------------------------------------------------------------------- #
# dbt is a subprocess with an explicit executable
# --------------------------------------------------------------------------- #
def test_dbt_runs_as_a_subprocess_from_the_dbt_project_with_the_explicit_executable(
        ctx, monkeypatch, tmp_path):
    fake = FakeRun()
    monkeypatch.setattr(refresh, "run_subprocess", fake)
    refresh.run_stage("dbt", ctx)
    (cmd, cwd), = fake.calls
    assert cmd == [str(tmp_path / "fake-dbt"), "build"]
    assert Path(cwd) == tmp_path / "pagos_dbt"


def test_a_missing_dbt_environment_is_a_named_stage_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DBT_BIN", raising=False)
    with pytest.raises(refresh.DbtNotFoundError, match="--dbt-bin"):
        refresh.resolve_dbt_bin(None, tmp_path)
    assert refresh.resolve_dbt_bin("C:/x/dbt.exe", tmp_path) == "C:/x/dbt.exe"
    (tmp_path / ".venv-dbt" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv-dbt" / "Scripts" / "dbt.exe").write_text("")
    assert refresh.resolve_dbt_bin(None, tmp_path).endswith("dbt.exe")


# --------------------------------------------------------------------------- #
# Idempotence by the last published month
# --------------------------------------------------------------------------- #
def test_first_run_forecasts_and_the_second_stops_at_no_new_month(ctx, monkeypatch, capsys):
    monkeypatch.setattr(refresh, "run_subprocess", FakeRun())
    assert refresh.refresh(list(refresh.STAGES), ctx) == 0
    first = _summary(ctx)
    assert first["outcome"] == "forecast" and first["target_month"] == "2026-10"
    assert [s["status"] for s in first["stages"]] == ["ok"] * 6
    f = ctx.out_dir / "forecast_2026-10_20261005T000000Z.json"
    assert f.exists() and (ctx.out_dir / "monitor.json").exists()
    before = f.read_bytes()
    rows = list(csv.DictReader((ctx.out_dir / "history.csv").open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["last_actual_month"] == "2026-07"

    # A NEW data_version with the same last published month is not a new month.
    ctx2 = refresh.Ctx(out_dir=ctx.out_dir, config=ctx.config, root=ctx.root,
                       snapshot=ctx.snapshot.with_name("panel_20261105T000000Z.parquet"))
    ctx2.snapshot.write_bytes(ctx.snapshot.read_bytes())
    (ctx.snapshot.parent / "series_meta_20261105T000000Z.csv").write_text(
        (ctx.snapshot.parent / "series_meta_20261005T000000Z.csv").read_text())
    capsys.readouterr()
    assert refresh.refresh(list(refresh.STAGES), ctx2) == 0
    out = capsys.readouterr().out
    assert "no new month published" in out
    second = _summary(ctx2)
    assert second["outcome"] == "no_new_month" and second["target_month"] is None
    assert [s["status"] for s in second["stages"]][-2:] == ["ok", "ok"]      # monitor still ran
    assert sorted(p.name for p in ctx.out_dir.glob("forecast_*.json")) == [f.name]   # no new file
    assert f.read_bytes() == before
    assert len(list(csv.DictReader((ctx.out_dir / "history.csv").open(encoding="utf-8")))) == 1


def test_no_new_month_without_the_etl_stages_runs_predict_and_monitor_only(ctx, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(refresh, "run_subprocess", fake)
    assert refresh.refresh(["predict", "monitor"], ctx) == 0
    assert fake.calls == []                                           # no ETL subprocess at all
    assert refresh.refresh(["predict", "monitor"], ctx) == 0          # second: stops, writes nothing new
    assert _summary(ctx)["outcome"] == "no_new_month"
    assert len(list(ctx.out_dir.glob("forecast_*.json"))) == 1


def test_a_snapshot_older_than_the_last_forecast_is_not_a_new_month(ctx):
    predict.append_history(ctx.out_dir / predict.HISTORY, {
        "target_month": "2026-11", "origin": "2026-10", "data_version": "v9",
        "point_forecast": 1.0, "band_80": [1, 2], "band_90": [1, 2],
        "last_actual": {"month": "2026-08", "value": 1.0}, "model_id": "m", "config_hash": "c",
        "git_sha": "g", "created_at": "t"})
    assert refresh.last_forecast_actual_month(ctx.out_dir / predict.HISTORY) == "2026-08"
    refresh.run_stage("predict", ctx)
    assert ctx.no_new_month and ctx.target_month is None


# --------------------------------------------------------------------------- #
# Failures: which stage, what command, the last 20 lines -- no stack trace
# --------------------------------------------------------------------------- #
def test_a_failing_stage_stops_the_chain_and_reports_stage_command_and_20_lines(
        ctx, monkeypatch, capsys):
    err = "\n".join(f"line {i}" for i in range(1, 31))
    fake = FakeRun(fail={"load_to_bigquery": (1, "", err)})
    monkeypatch.setattr(refresh, "run_subprocess", fake)
    assert refresh.refresh(list(refresh.STAGES), ctx) == 1
    captured = capsys.readouterr()
    assert "REFRESH FAILED at stage 'load'" in captured.err
    assert "src.data_collection.load_to_bigquery" in captured.err and "exit code: 1" in captured.err
    assert "line 30" in captured.err and "line 11" in captured.err and "line 10\n" not in captured.err
    assert "Traceback" not in captured.err
    ran = [" ".join(c) for c, _ in fake.calls]
    assert len(ran) == 2 and "fetch_target_series" in ran[0]          # fetch ran, load failed, rest never
    s = _summary(ctx)
    assert s["outcome"] == "failed" and s["error"]["stage"] == "load"
    assert [r["status"] for r in s["stages"]][:3] == ["ok", "failed", "not_run"]
    assert not list(ctx.out_dir.glob("forecast_*.json"))               # nothing forecast after a failure


def test_bcrp_bot_protection_surfaces_as_a_named_error_not_a_stack_trace(
        ctx, monkeypatch, capsys):
    out = ("  x PN42200EM  n_transf_intra_agg  FAILED: BCRP returned a non-JSON body\n"
           "BCRP answered 2 requests in a row with a non-JSON body (bot protection?). Stopping.")
    monkeypatch.setattr(refresh, "run_subprocess",
                        FakeRun(fail={"fetch_target_series": (1, out, "")}))
    assert refresh.refresh(list(refresh.STAGES), ctx) == 1
    err = capsys.readouterr().err
    assert "[BcrpBlockedError]" in err and "stage 'fetch'" in err and "O-13" in err
    assert "Traceback" not in err
    assert _summary(ctx)["error"]["kind"] == "BcrpBlockedError"


def test_an_exception_inside_an_in_process_stage_is_a_stage_error_too(ctx, monkeypatch, capsys):
    def boom(c):
        raise ValueError("snapshot has no payments column")
    monkeypatch.setitem(refresh.IN_PROCESS, "predict", boom)
    assert refresh.refresh(["predict", "monitor"], ctx) == 1
    err = capsys.readouterr().err
    assert "stage 'predict'" in err and "ValueError: snapshot has no payments column" in err
    assert "Traceback" not in err
    assert "Traceback" in _summary(ctx)["error"]["traceback"]         # kept for the record only


def test_a_missing_executable_is_a_stage_error(ctx, monkeypatch, capsys):
    def nope(cmd, cwd=None):
        raise FileNotFoundError("no such file")
    monkeypatch.setattr(refresh, "run_subprocess", nope)
    assert refresh.refresh(["snapshot"], ctx) == 1
    assert "executable not found" in capsys.readouterr().err


def test_a_monitor_alert_is_exit_2_but_the_forecast_is_still_made(ctx, monkeypatch):
    monkeypatch.setattr(refresh, "run_subprocess", FakeRun())

    def alert(c):
        c.monitor_exit, c.monitor_status = 2, "alert"
    monkeypatch.setitem(refresh.IN_PROCESS, "monitor", alert)
    assert refresh.refresh(["predict", "monitor"], ctx) == 2
    s = _summary(ctx)
    assert s["outcome"] == "forecast" and s["monitor_status"] == "alert" and s["monitor_exit"] == 2


def test_the_run_summary_records_stages_timings_and_outcome(ctx, monkeypatch):
    monkeypatch.setattr(refresh, "run_subprocess", FakeRun())
    refresh.refresh(["snapshot", "predict", "monitor"], ctx)
    s = _summary(ctx)
    assert set(s) >= {"outcome", "started_at", "finished_at", "seconds", "stages", "data_version",
                      "target_month", "forecast_file", "point_forecast", "monitor_status", "error"}
    assert [r["stage"] for r in s["stages"]] == list(refresh.STAGES)
    assert [r["status"] for r in s["stages"]] == ["skipped"] * 3 + ["ok"] * 3
    assert all(isinstance(r["seconds"], float) for r in s["stages"])
    assert s["data_version"] == "20261005T000000Z" and s["error"] is None


def test_the_fetch_window_always_reaches_the_end_of_the_current_year():
    assert f"{date.today().year}-12" == fetch_target_series.END
