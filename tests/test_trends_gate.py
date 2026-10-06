"""scripts/trends_gate.py -- G1-G3 exactly as training_plan.md 9.2, CV period only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import trends_gate as gate
from src.data_collection import trends


def _pull(tmp_path, pull, yape, plin=None, n=None):
    n = n or len(yape)
    idx = pd.date_range("2017-01-01", periods=n + 1, freq="MS")      # +1: the incomplete pull month
    plin = plin if plin is not None else np.zeros(n)
    rows = [(d.strftime("%Y-%m-%d"), float(y), float(p))
            for d, y, p in zip(idx, list(yape) + [50], list(plin) + [0], strict=True)]
    csv = tmp_path / f"{pull}_yape_plin.csv"
    csv.write_text('"Time","Yape","Plin"\n' + "\n".join(",".join(map(str, r)) for r in rows) + "\n",
                   encoding="utf-8")
    trends.meta_path(csv).write_text(json.dumps(trends.build_meta(csv)), encoding="utf-8")
    return trends.load_pull(csv)


def _payments(n_months=110, start="2017-01-01"):
    idx = pd.date_range(start, periods=n_months, freq="MS")
    return pd.Series(np.exp(0.02 * np.arange(n_months)), index=idx)


def _growth(n, seed=1):
    rng = np.random.default_rng(seed)
    return np.clip(10 * np.exp(0.03 * np.arange(n) + 0.05 * rng.standard_normal(n)), 6, 100).round()


def test_cv_period_is_everything_before_the_final_twelve_published_months():
    pay = _payments(100)
    pay.iloc[-3:] = np.nan                                  # payments end 3 months before the panel edge
    last = pay.dropna().index.max()
    assert gate.cv_end_from_payments(pay) == last - pd.DateOffset(months=12)
    # plan 9.2: payments through 2026-07 -> CV period ends 2025-07 ("before 2025-08")
    real = pd.Series(1.0, index=pd.date_range("2013-01-01", "2026-07-01", freq="MS"))
    assert gate.cv_end_from_payments(real) == pd.Timestamp("2025-07-01")


def test_g1_passes_with_six_low_months_and_fails_with_seven():
    idx = pd.date_range("2017-01-01", "2025-07-01", freq="MS")
    gt = pd.Series(50.0, index=idx)
    low = gt.index[gt.index >= "2017-09-01"][:6]
    gt[low] = 4.9
    assert gate.g1_coverage(gt, idx[-1]).status == "PASS"
    gt[gt.index[gt.index >= "2017-09-01"][6]] = 0
    g = gate.g1_coverage(gt, idx[-1])
    assert g.status == "FAIL" and "w2021" in g.consequence
    # months BEFORE 2017-09 are not counted
    gt2 = pd.Series(50.0, index=idx)
    gt2[:"2017-08-01"] = 0
    assert gate.g1_coverage(gt2, idx[-1]).status == "PASS"


def test_g1_threshold_is_strictly_below_five():
    idx = pd.date_range("2017-01-01", "2025-07-01", freq="MS")
    gt = pd.Series(5.0, index=idx)                          # exactly 5 is not "< 5"
    assert gate.g1_coverage(gt, idx[-1]).status == "PASS"


def test_g2_is_pending_with_one_pull_and_decides_with_two(tmp_path):
    y = _growth(110)
    a = _pull(tmp_path, "2026-03-01", y)
    cv_end = pd.Timestamp("2025-07-01")
    assert gate.g2_stability([a], cv_end).status == "PENDING"

    same = _pull(tmp_path, "2026-03-02", np.minimum(y + 1, 100))             # shifted by one: dlog nearly identical
    assert gate.g2_stability([a, same], cv_end).status == "PASS"
    noisy = _pull(tmp_path, "2026-03-03", _growth(110, seed=99))
    g = gate.g2_stability([a, noisy], cv_end)
    assert g.status == "FAIL" and "geometric mean" in g.consequence


def test_g3_needs_a_positive_correlation_of_at_least_point_two_at_some_lead():
    idx = pd.date_range("2017-01-01", periods=110, freq="MS")
    rng = np.random.default_rng(3)
    n_growth = 0.02 + 0.03 * rng.standard_normal(110)
    pay = pd.Series(np.exp(np.cumsum(n_growth)), index=idx)
    # gt leads payments by 2 months: gt_m moves with the payments growth of month m+2
    gt_growth = np.roll(n_growth, -2) + 0.01 * rng.standard_normal(110)
    gt = pd.Series(10 * np.exp(np.cumsum(gt_growth)), index=idx)
    cv_end = gate.cv_end_from_payments(pay)
    g = gate.g3_relevance(gt, pay, cv_end)
    assert g.status == "PASS" and "lead 2" in g.result

    flat = pd.Series(10 * np.exp(np.cumsum(0.03 * rng.standard_normal(110))), index=idx)
    assert gate.g3_relevance(flat, pay, cv_end).status == "FAIL"
    # a strong NEGATIVE correlation does not pass: the sign must be positive
    neg = pd.Series(10 * np.exp(np.cumsum(-np.roll(n_growth, -2))), index=idx)
    assert gate.g3_relevance(neg, pay, cv_end).status == "FAIL"


def test_the_gate_never_reads_the_holdout(tmp_path):
    """Perturb everything at or after the holdout start: nothing in the gate may move."""
    n = 110
    y = _growth(n)
    pay = _payments(n)
    pull = _pull(tmp_path, "2026-03-01", y)
    base, cv_end, _, notes = gate.run([pull], pay)

    pay2 = pay.copy()
    pay2[pay2.index > cv_end] *= 5.0                        # wreck the holdout payments
    y2 = y.copy()
    y2[pd.date_range("2017-01-01", periods=n, freq="MS") > cv_end] = 77.0
    pull2 = _pull(tmp_path, "2026-03-02", y2)
    again, _, _, notes2 = gate.run([pull2], pay2)

    for g1, g2 in zip(base, again, strict=True):
        assert (g1.status, g1.result, g1.detail) == (g2.status, g2.result, g2.detail), g1.id
    assert notes[1:] == notes2[1:]


def test_zero_months_are_reported_not_repaired():
    s = pd.Series([0, 0, 0, 4, 8, 9, 10, 12, 11, 13, 14, 15, 16, 17, 18, 19],
                  index=pd.date_range("2017-01-01", periods=16, freq="MS"), dtype=float)
    d = gate.dlog(s)
    assert d.iloc[:4].isna().all() and np.isfinite(d.iloc[4])      # log(0) is never forced to a number
    assert gate.first_full_history(s) == pd.Timestamp("2018-04-01")  # 13 positive levels from 2017-04
    assert gate.first_full_history(s[:8]) is None


def test_render_names_the_consequence_and_the_cv_boundary(tmp_path):
    pull = _pull(tmp_path, "2026-03-01", _growth(110))
    gates, cv_end, chosen, notes = gate.run([pull], _payments(110))
    md = gate.render(gates, cv_end, chosen, notes, pd.Timestamp("2026-10-05").date())
    assert md.startswith("# Google Trends data gate") and f"{cv_end:%Y-%m}" in md
    assert "G2 Stability | **PENDING**" in md
