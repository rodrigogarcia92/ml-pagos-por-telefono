"""Step 3 -- the forecast command: origin selection under kappa, prediction rows, band, history."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting import ensemble, predict, production
from src.model_training import dataset, registry, train

FS3 = train.load_feature_sets()["FS3_activity"]
BAND = {"p05": -0.05, "p10": -0.04, "p50": 0.01, "p90": 0.10, "p95": 0.11,
        "n_folds": 41, "cv_mape_pct": 4.4, "first_target": "2022-03", "last_target": "2025-07"}
REAL = Path("data/processed/panel_20261005T000000Z.parquet")


def _cfg(member_params):
    return {
        "target_id": "t3", "horizon": 3, "kappa": 2, "window": "w2019",
        "feature_set": "FS3_activity", "columns": FS3, "model_family": "ens3",
        "members": {f: {"encoding": registry.default_encoding(f), "params": p}
                    for f, p in member_params.items()},
        "mlflow_run_id": "x", "data_version": "20261005T000000Z", "protocol_version": "1.7",
        "selected_on": "cv", "error_band": dict(BAND),
    }


def _forecast(panel, meta, member_params, tmp_path):
    cfg = _cfg(member_params)
    path = tmp_path / "cfg.yaml"
    production.dump(cfg, path)
    return predict.forecast(panel, meta, cfg, data_version="20261005T000000Z", cfg_path=path)


def _rows(panel, meta, enc="int", cols=FS3):
    return dataset.build_prediction_rows(panel, meta, target_id="t3", horizon=3, window="w2019",
                                         columns=cols, encoding=enc)


# --------------------------------------------------------------------------- #
# Origin selection under kappa
# --------------------------------------------------------------------------- #
def test_the_latest_origin_is_two_months_after_the_last_payment_and_the_target_is_t_plus_1(
        prod_panel, member_params, tmp_path):
    panel, meta = prod_panel
    # payments / pbi end 2026-07 (kappa 2), circulante 2026-08 (kappa 1): the binding series are
    # the kappa-2 ones, so t - 2 <= 2026-07 -> t = 2026-09; target = t - kappa + h = t + 1.
    rec = _forecast(panel, meta, member_params, tmp_path)
    assert (rec["origin"], rec["target_month"]) == ("2026-09", "2026-10")
    assert rec["last_actual"]["month"] == "2026-07"
    assert rec["anchor"] == {"month": "2026-07", "value": rec["last_actual"]["value"]}


@pytest.mark.parametrize("col,last_month,origin,target", [
    ("circulante", "2026-07-01", "2026-08", "2026-09"),          # kappa 1: origin <= last + 1
    ("n_transf_intra_agg", "2026-06-01", "2026-08", "2026-09"),  # kappa 2: origin <= last + 2
    ("pbi_idx", "2026-05-01", "2026-07", "2026-08"),
])
def test_origin_follows_the_slowest_feature_under_its_own_kappa(
        prod_panel, member_params, tmp_path, col, last_month, origin, target):
    panel, meta = prod_panel
    p = panel.copy()
    p.loc[p.index > pd.Timestamp(last_month), col] = np.nan
    rec = _forecast(p, meta, member_params, tmp_path)
    assert (rec["origin"], rec["target_month"]) == (origin, target)


def test_prediction_rows_are_exactly_the_unpublished_targets(prod_panel):
    panel, meta = prod_panel
    pr = _rows(panel, meta)
    # origins 2026-07, -08, -09 -> targets 2026-08, -09, -10: none published, all features there
    assert list(pr.X.index) == list(pd.to_datetime(["2026-07-01", "2026-08-01", "2026-09-01"]))
    assert pr.y.isna().all() and pr.ctx["y_level"].isna().all()
    assert pr.X.notna().all().all() and pr.ctx["anchor"].notna().all()
    assert len(pr.X.columns) == 22


def test_the_kappa_guard_still_applies_to_prediction_rows(prod_panel):
    panel, meta = prod_panel
    with pytest.raises(ValueError, match="violates kappa"):
        _rows(panel, meta, cols=[*FS3, "y_d1"])            # y_d1 is inadmissible at kappa 2
    with pytest.raises(ValueError, match="violates kappa"):
        dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                      columns=["circ_d0"], keep_unlabelled=True)


def test_prediction_rows_share_the_training_columns_in_every_encoding(prod_panel):
    panel, meta = prod_panel
    for enc, n in (("int", 22), ("onehot", 32)):
        tr = dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                           columns=FS3, encoding=enc)
        pr = _rows(panel, meta, enc)
        assert list(pr.X.columns) == list(tr.X.columns) and pr.X.shape[1] == n
        assert pr.X.index.intersection(tr.X.index).empty        # a row is labelled or pending


# --------------------------------------------------------------------------- #
# The training path is byte-identical
# --------------------------------------------------------------------------- #
def _digest(fr) -> str:
    h = hashlib.sha256()
    for o in (fr.X, fr.y, fr.ctx):
        h.update(pd.util.hash_pandas_object(o, index=True).values.tobytes())
    h.update("|".join(fr.X.columns).encode())
    h.update(str(fr.dropped).encode())
    return h.hexdigest()


def test_training_frames_ignore_the_new_flag_and_trailing_empty_months(prod_panel):
    panel, meta = prod_panel
    for enc in ("int", "onehot"):
        base = dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                             columns=FS3, encoding=enc)
        explicit = dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                                 columns=FS3, encoding=enc, keep_unlabelled=False)
        longer = dataset.build(
            panel.reindex(pd.date_range(panel.index.min(), "2027-03-01", freq="MS")), meta,
            target_id="t3", horizon=3, window="w2019", columns=FS3, encoding=enc)
        assert _digest(base) == _digest(explicit) == _digest(longer)
        # ...and the opt-in frame holds the labelled rows unchanged, plus the pending ones
        wide = dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                             columns=FS3, encoding=enc, keep_unlabelled=True)
        pd.testing.assert_frame_equal(wide.X.loc[base.X.index], base.X)
        assert len(wide.X) == len(base.X) + 3


@pytest.mark.skipif(not REAL.exists(), reason="real snapshot not present")
def test_training_frames_on_the_real_snapshot_equal_the_pre_change_digests():
    """Digests taken from the code as committed BEFORE keep_unlabelled existed."""
    panel, meta = dataset.load_snapshot(REAL)
    want = {"int": "2188d6c5656aefe865f036b5b559111bc43ffcf75eb693a85a9ca73a3957590c",
            "onehot": "6ffbf1784691c0ceffb997ca628735a8ce38b0898e6bc5c7ce8a328f5fa0bdfe"}
    for enc, digest in want.items():
        fr = dataset.build(panel, meta, target_id="t3", horizon=3, window="w2019",
                           columns=FS3, encoding=enc)
        assert _digest(fr) == digest


# --------------------------------------------------------------------------- #
# Band arithmetic and the production fit
# --------------------------------------------------------------------------- #
def test_the_forecast_is_anchor_times_exp_z_and_the_band_is_the_frozen_quantiles(
        prod_panel, member_params, tmp_path):
    panel, meta = prod_panel
    rec = _forecast(panel, meta, member_params, tmp_path)
    point, anchor = rec["point_forecast"], rec["anchor"]["value"]
    z = np.log(np.array(list(rec["members"].values())) / anchor)
    assert point == pytest.approx(anchor * np.exp(z.mean()), rel=1e-3)      # mean of z, not of levels
    assert rec["band_80"] == pytest.approx([point * np.exp(BAND["p10"]), point * np.exp(BAND["p90"])],
                                           rel=1e-4)
    assert rec["band_90"] == pytest.approx([point * np.exp(BAND["p05"]), point * np.exp(BAND["p95"])],
                                           rel=1e-4)
    assert rec["band_90"][0] < rec["band_80"][0] < point < rec["band_80"][1] < rec["band_90"][1]
    assert rec["model_id"] == "t3_ens3_FS3_activity_w2019" and len(rec["config_hash"]) == 12
    assert rec["created_at"].endswith("+00:00") and rec["unit"].startswith("millions")


def test_the_production_fit_uses_every_labelled_row_including_the_holdout_months(
        prod_panel, member_params):
    panel, meta = prod_panel
    frames = ensemble.member_frames(panel, meta, target_id="t3", horizon=3, window="w2019",
                                    columns=FS3)
    pr = {f: _rows(panel, meta, registry.default_encoding(f),
                   registry.model_columns(f, FS3)).X.iloc[[-1]] for f in ensemble.MEMBERS}
    pred = {f: type("F", (), {"X": pr[f]})() for f in pr}
    z0, _ = ensemble.fit_predict(frames, pred, member_params, horizon=3)

    hold_start = ensemble.first_holdout_target(frames, window="w2019", horizon=3, kappa=2)
    wrecked = panel.copy()
    wrecked.loc[wrecked.index >= hold_start, "n_transf_intra_agg"] *= 1.5
    frames2 = ensemble.member_frames(wrecked, meta, target_id="t3", horizon=3, window="w2019",
                                     columns=FS3)
    z1, _ = ensemble.fit_predict(frames2, pred, member_params, horizon=3)
    assert z0 != z1          # the holdout months ARE in the production fit (by design, unscored)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _record(**over):
    rec = {"target_month": "2026-10", "origin": "2026-09", "data_version": "20261005T000000Z",
           "point_forecast": 1273.7, "band_80": [1226.7, 1405.4], "band_90": [1214.7, 1419.5],
           "last_actual": {"month": "2026-07", "value": 1154.1}, "model_id": "m",
           "config_hash": "abc", "git_sha": "deadbee", "created_at": "2026-10-06T00:00:00+00:00"}
    return {**rec, **over}


def test_history_is_created_with_a_header_and_never_duplicates_a_forecast(tmp_path):
    h = tmp_path / "sub" / "history.csv"
    assert predict.append_history(h, _record()) is True
    assert predict.append_history(h, _record(created_at="later")) is False          # idempotent
    assert predict.append_history(h, _record(data_version="20261105T000000Z")) is True   # new vintage
    assert predict.append_history(h, _record(target_month="2026-11")) is True
    with h.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and list(rows[0]) == predict.HISTORY_COLUMNS
    assert rows[0]["created_at"] == "2026-10-06T00:00:00+00:00"                     # not rewritten
    assert h.read_text(encoding="utf-8").count("target_month,origin") == 1          # one header


def test_write_outputs_names_the_file_and_leaves_an_existing_forecast_untouched(tmp_path):
    p1, new1 = predict.write_outputs(_record(), tmp_path)
    assert p1.name == "forecast_2026-10_20261005T000000Z.json" and new1
    before = p1.read_text(encoding="utf-8")
    p2, new2 = predict.write_outputs(_record(created_at="2030-01-01T00:00:00+00:00"), tmp_path)
    assert p2 == p1 and not new2 and p1.read_text(encoding="utf-8") == before
    assert json.loads(before)["point_forecast"] == 1273.7


def test_the_cli_prints_and_writes_only_where_told(prod_snapshot, member_params, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    production.dump(_cfg(member_params), cfg_path)
    out = tmp_path / "out"
    assert predict.main(["--snapshot", prod_snapshot, "--config", str(cfg_path), "--no-write",
                         "--out", str(out)]) == 0
    assert not out.exists() and "forecast for 2026-10" in capsys.readouterr().out
    assert predict.main(["--snapshot", prod_snapshot, "--config", str(cfg_path),
                         "--out", str(out)]) == 0
    assert (out / "forecast_2026-10_20261005T000000Z.json").exists() and (out / "history.csv").exists()


# --------------------------------------------------------------------------- #
# The real snapshot
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL.exists() or not production.DEFAULT_CONFIG.exists(),
                    reason="real snapshot or frozen config not present")
def test_on_snapshot_20261005_the_origin_is_2026_09_and_the_target_2026_10():
    panel, meta = dataset.load_snapshot(REAL)
    cfg = production.load()
    rec = predict.forecast(panel, meta, cfg, data_version="20261005T000000Z")
    assert (rec["origin"], rec["target_month"]) == ("2026-09", "2026-10")
    assert rec["last_actual"]["month"] == "2026-07"
    assert 900 < rec["point_forecast"] < 2000
    assert rec["band_90"][0] < rec["band_80"][0] < rec["point_forecast"] < rec["band_80"][1]
