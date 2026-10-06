"""Step 4 -- the drift monitor: scoring, band coverage, status rules, exit codes."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.forecasting import monitor, production


def _history(rows):
    """rows: (target_month, forecast, data_version, created_at); band +/-5% / +/-8%."""
    return pd.DataFrame([{
        "target_month": m, "origin": "x", "data_version": dv, "point_forecast": f,
        "lo80": f * 0.95, "hi80": f * 1.05, "lo90": f * 0.92, "hi90": f * 1.08, "created_at": ca,
    } for m, f, dv, ca in rows])


def _actual(**months):
    return pd.Series({m.replace("_", "-"): v for m, v in months.items()}, dtype=float)


def _status(hist, act):
    return monitor.summarize(hist, act, snapshot_version="v")["status"]


H = [("2026-04", 100.0, "v1", "2026-05-01"), ("2026-05", 100.0, "v1", "2026-06-01"),
     ("2026-06", 100.0, "v1", "2026-07-01")]


def test_percentage_error_and_band_membership():
    scored, pending = monitor.score(_history(H), _actual(**{"2026_04": 100.0, "2026_05": 96.0,
                                                           "2026_06": 120.0}))
    assert pending == [] and list(scored["target_month"]) == ["2026-04", "2026-05", "2026-06"]
    assert list(scored["pct_error"].round(3)) == [0.0, round((100 - 96) / 96 * 100, 3),
                                                  round((100 - 120) / 120 * 100, 3)]
    assert list(scored["in_80"]) == [True, True, False]       # 96 in [95,105]; 120 outside
    assert list(scored["in_90"]) == [True, True, False]


def test_90_band_is_wider_than_80():
    scored, _ = monitor.score(_history(H[:1]), _actual(**{"2026_04": 93.0}))   # 7% under
    assert bool(scored["in_80"].iloc[0]) is False and bool(scored["in_90"].iloc[0]) is True


def test_ok_when_the_latest_month_is_within_the_threshold():
    assert _status(_history(H), _actual(**{"2026_04": 100.0, "2026_05": 100.0, "2026_06": 109.0})) == "ok"
    assert monitor.status_of(pd.DataFrame()) == "ok"
    out = monitor.summarize(_history(H[:0]), _actual(), snapshot_version="v")
    assert out["status"] == "ok" and out["n_scored"] == 0 and out["rolling_mape_pct"] is None


def test_warning_when_only_the_latest_month_missed_by_more_than_10_percent():
    act = _actual(**{"2026_04": 150.0, "2026_05": 100.0, "2026_06": 150.0})   # miss, hit, miss
    assert _status(_history(H), act) == "warning"
    only_last = _actual(**{"2026_04": 100.0, "2026_05": 100.0, "2026_06": 120.0})
    assert _status(_history(H), only_last) == "warning"


def test_alert_when_the_two_latest_consecutive_months_both_missed():
    act = _actual(**{"2026_04": 100.0, "2026_05": 130.0, "2026_06": 130.0})
    out = monitor.summarize(_history(H), act, snapshot_version="v")
    assert out["status"] == "alert" and "reason" in out and "2026-05" in out["reason"]


def test_two_misses_with_a_calendar_gap_are_not_consecutive():
    hist = _history([("2026-04", 100.0, "v1", "a"), ("2026-06", 100.0, "v1", "b")])
    assert _status(hist, _actual(**{"2026_04": 150.0, "2026_06": 150.0})) == "warning"


def test_an_old_miss_followed_by_a_good_month_is_ok():
    act = _actual(**{"2026_04": 150.0, "2026_05": 150.0, "2026_06": 101.0})
    assert _status(_history(H), act) == "ok"


def test_exactly_ten_percent_is_not_a_miss():
    hist = _history([("2026-04", 110.0, "v1", "c")])
    assert _status(hist, _actual(**{"2026_04": 100.0})) == "ok"          # +10.0% over: not > 10
    assert _status(hist, _actual(**{"2026_04": 99.0})) == "warning"      # +11.1%


def test_the_earliest_forecast_for_a_month_is_the_one_scored():
    hist = _history([("2026-06", 100.0, "v2", "2026-08-01"),      # a later vintage, closer to the truth
                     ("2026-06", 130.0, "v1", "2026-07-01")])
    scored, _ = monitor.score(hist, _actual(**{"2026_06": 100.0}))
    assert len(scored) == 1 and scored["forecast"].iloc[0] == 130.0
    assert scored["data_version"].iloc[0] == "v1"


def test_months_without_a_forecast_are_not_scored_and_forecasts_without_an_actual_are_pending():
    hist = _history([("2026-05", 100.0, "v1", "a"), ("2026-09", 100.0, "v1", "b")])
    act = _actual(**{"2026_04": 100.0, "2026_05": 100.0, "2026_06": 100.0})   # 04 and 06: no forecast
    out = monitor.summarize(hist, act, snapshot_version="v")
    assert [r["target_month"] for r in out["scored"]] == ["2026-05"]
    assert out["pending_months"] == ["2026-09"] and out["n_scored"] == 1
    assert out["status"] == "ok"


def test_rolling_mape_uses_the_last_window_and_coverage_the_whole_history():
    rows = [(f"2026-{m:02d}", 100.0, "v", f"c{m:02d}") for m in range(1, 7)]
    act = _actual(**{f"2026_{m:02d}": a for m, a in zip(range(1, 7), [100, 100, 100, 100, 104, 96], strict=True)})
    out = monitor.summarize(_history(rows), act, snapshot_version="v", window=2)
    # last two: |100-104|/104 and |100-96|/96
    assert out["rolling_mape_pct"] == pytest.approx((4 / 104 * 100 + 4 / 96 * 100) / 2, abs=1e-3)
    assert out["coverage"]["n"] == 6 and out["coverage"]["within_80"] == 1.0
    assert out["rolling_window_months"] == 2


# --------------------------------------------------------------------------- #
# CLI: monitor.json and exit codes
# --------------------------------------------------------------------------- #
def _cli(tmp_path, prod_snapshot, member_params, hist_rows, monkeypatch=None):
    h = _history(hist_rows)
    hp = tmp_path / "history.csv"
    h.to_csv(hp, index=False)
    cfg = tmp_path / "cfg.yaml"
    production.dump({"target_id": "t3", "horizon": 3, "kappa": 2, "window": "w2019",
                     "feature_set": "F", "columns": [], "model_family": "ens3", "members": {},
                     "mlflow_run_id": "x", "data_version": "v", "protocol_version": "1.7",
                     "selected_on": "cv",
                     "error_band": {k: 0 for k in production.BAND_KEYS}}, cfg)
    out = tmp_path / "monitor.json"
    rc = monitor.main(["--snapshot", prod_snapshot, "--history", str(hp), "--out", str(out),
                       "--config", str(cfg)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_cli_exit_codes_0_0_2_and_the_json_it_writes(tmp_path, prod_snapshot, prod_panel, member_params):
    panel, _ = prod_panel
    last = panel["n_transf_intra_agg"].dropna()
    m1, m2 = last.index[-2].strftime("%Y-%m"), last.index[-1].strftime("%Y-%m")
    a1, a2 = float(last.iloc[-2]), float(last.iloc[-1])

    ok_rows = [(m1, a1 * 1.01, "v1", "c1"), (m2, a2 * 0.99, "v1", "c2")]
    rc, res = _cli(tmp_path, prod_snapshot, member_params, ok_rows)
    assert rc == 0 and res["status"] == "ok" and res["n_scored"] == 2
    assert res["snapshot_data_version"] == "20261005T000000Z"

    warn_rows = [(m1, a1 * 1.01, "v1", "c1"), (m2, a2 * 1.2, "v1", "c2")]
    rc, res = _cli(tmp_path, prod_snapshot, member_params, warn_rows)
    assert rc == 0 and res["status"] == "warning"

    alert_rows = [(m1, a1 * 1.3, "v1", "c1"), (m2, a2 * 1.3, "v1", "c2")]
    rc, res = _cli(tmp_path, prod_snapshot, member_params, alert_rows)
    assert rc == 2 and res["status"] == "alert"
    assert set(res) >= {"status", "scored", "rolling_mape_pct", "coverage", "pending_months"}
    assert res["scored"][0]["target_month"] == m1


def test_cli_with_no_history_file_is_ok_and_scores_nothing(tmp_path, prod_snapshot, capsys):
    cfg = tmp_path / "cfg.yaml"
    production.dump({"target_id": "t3", "horizon": 3, "kappa": 2, "window": "w", "feature_set": "F",
                     "columns": [], "model_family": "ens3", "members": {}, "mlflow_run_id": "x",
                     "data_version": "v", "protocol_version": "1.7", "selected_on": "cv",
                     "error_band": {k: 0 for k in production.BAND_KEYS}}, cfg)
    rc = monitor.main(["--snapshot", prod_snapshot, "--history", str(tmp_path / "none.csv"),
                       "--out", str(tmp_path / "m.json"), "--config", str(cfg)])
    assert rc == 0 and "OK" in capsys.readouterr().out
