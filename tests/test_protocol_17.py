"""Protocol 1.7 (training_plan.md 3 Targets 10-11, 4.5, 5.4, 7.4, 9.2): T7, T9, T10 and friends.

Offline, synthetic panels. The one test that reads the real snapshot is skipped when
the (gitignored) snapshot is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model_training import dataset, splits
from src.model_training.train import HOLDOUT_MONTHS, load_feature_sets

REAL_SNAPSHOT = Path("data/processed/panel_20261005T000000Z.parquet")


@pytest.fixture(scope="module")
def proxy_panel():
    """Shaped like the 2026-10-05 snapshot: index to 2026-09 (macro and Trends are ahead of
    payments), payments published through 2026-07 (kappa = 2), gt strictly positive."""
    rng = np.random.default_rng(17)
    idx = pd.date_range("2013-01-01", "2026-09-01", freq="MS")
    n = len(idx)
    month = idx.month.to_numpy() - 1
    p = pd.DataFrame({
        "n_transf_intra_agg": np.exp(np.linspace(np.log(10), np.log(700), n)
                                     + 0.05 * np.sin(2 * np.pi * month / 12)
                                     + 0.01 * rng.standard_normal(n)),
        "circulante": np.exp(np.linspace(np.log(40000), np.log(90000), n)),
        "pbi_idx": 100 + np.linspace(0, 60, n) + rng.standard_normal(n),
        "gt_yape_plin": np.nan,
    }, index=idx)
    k = int((idx >= "2017-01-01").sum())
    p.loc["2017-01-01":, "gt_yape_plin"] = np.clip(
        8 * np.exp(0.03 * np.arange(k) + 0.1 * rng.standard_normal(k)), 6, 190)
    p.loc["2026-08-01":, "n_transf_intra_agg"] = np.nan
    meta = pd.DataFrame({
        "col_name": p.columns, "kappa": [2, 1, 2, 0], "transform": ["log_diff"] * 4,
    }).set_index("col_name")
    return p, meta


def _build(pl, target="t10", h=5, cols=(), window="w2019", encoding="int", *, meta=None):
    p, m = pl
    return dataset.build(p, m if meta is None else meta, target_id=target, horizon=h,
                         window=window, columns=list(cols), encoding=encoding)


GT_COLS = ["gt_d0", "gt_ma3", "gt_ma12"]


# --------------------------------------------------------------------------- #
# Targets 10-11
# --------------------------------------------------------------------------- #
def test_t10_t11_are_the_proxy_at_h5_and_h6():
    assert dataset.TARGETS["t10"] == {"cols": ["n_transf_intra_agg"], "horizon": 5}
    assert dataset.TARGETS["t11"] == {"cols": ["n_transf_intra_agg"], "horizon": 6}
    assert {"t6", "t7", "t8", "t9"}.isdisjoint(dataset.TARGETS)         # reserved IDs stay free


@pytest.mark.parametrize("target,h,months_ahead", [("t10", 5, 3), ("t11", 6, 4)])
def test_t10_t11_target_is_t_plus_3_and_t_plus_4_over_the_t_minus_2_anchor(
        proxy_panel, target, h, months_ahead):
    p, _ = proxy_panel
    f = _build(proxy_panel, target, h)
    n = p["n_transf_intra_agg"]
    t = f.X.index
    expected = np.log(n.reindex(t + pd.DateOffset(months=months_ahead)).to_numpy()) \
        - np.log(n.reindex(t - pd.DateOffset(months=2)).to_numpy())
    np.testing.assert_allclose(f.y.to_numpy(), expected, rtol=1e-12)
    # the window bounds the TARGET month, and the last one is the last published month
    assert f.target_start == pd.Timestamp("2019-01-01")
    assert f.target_end == pd.Timestamp("2026-07-01")


@pytest.mark.parametrize("target,h,purge", [("t10", 5, 4), ("t11", 6, 5)])
def test_t10_t11_purge_is_h_minus_1_and_no_train_target_reaches_a_test_anchor(
        proxy_panel, target, h, purge):
    f = _build(proxy_panel, target, h, load_feature_sets()["FS3_activity"])
    folds = splits.make_folds(len(f.X) - HOLDOUT_MONTHS["w2019"], window="w2019", horizon=h)
    for fold in folds:
        assert fold.train.max() + purge < fold.test.min()
        assert len(fold.train) >= splits.MIN_TRAIN["w2019"]
        # the last training row's target must be known before the test origin's anchor
        last_train_target = f.X.index[fold.train.max()] + pd.DateOffset(months=h - 2)
        test_anchor = f.X.index[fold.test.min()] - pd.DateOffset(months=2)
        assert last_train_target <= test_anchor


@pytest.mark.parametrize("target,h,expected", [("t3", 3, 41), ("t10", 5, 39), ("t11", 6, 38)])
def test_fold_counts_on_a_snapshot_shaped_like_20261005(proxy_panel, target, h, expected):
    """Plan 3 (Targets 10-11): 39 / 38 outer folds on w2019 with FS3, payments through 2026-07."""
    f = _build(proxy_panel, target, h, load_feature_sets()["FS3_activity"])
    assert len(f.X) == 91
    n_cv = len(f.X) - HOLDOUT_MONTHS["w2019"]
    assert len(splits.make_folds(n_cv, window="w2019", horizon=h)) == expected


@pytest.mark.skipif(not REAL_SNAPSHOT.exists(), reason="snapshot 20261005 is gitignored / absent")
@pytest.mark.parametrize("target,h,expected", [("t10", 5, 39), ("t11", 6, 38)])
def test_fold_counts_on_the_real_snapshot(target, h, expected):
    panel, meta = dataset.load_snapshot(REAL_SNAPSHOT)
    f = dataset.build(panel, meta, target_id=target, horizon=h, window="w2019",
                      columns=load_feature_sets()["FS3_activity"], encoding="int")
    n_cv = len(f.X) - HOLDOUT_MONTHS["w2019"]
    assert len(splits.make_folds(n_cv, window="w2019", horizon=h)) == expected


# --------------------------------------------------------------------------- #
# Feature sets: FS3_gt (plan 5.4) and T10
# --------------------------------------------------------------------------- #
def test_t10_fs3_is_a_strict_subset_of_fs3_gt_in_both_encodings(proxy_panel):
    fs = load_feature_sets()
    assert set(fs["FS3_activity"]) < set(fs["FS3_gt"])
    assert set(fs["FS3_gt"]) - set(fs["FS3_activity"]) == set(GT_COLS)
    for enc in ("int", "onehot"):
        a = set(_build(proxy_panel, "t10", 5, fs["FS3_activity"], encoding=enc).X.columns)
        b = set(_build(proxy_panel, "t10", 5, fs["FS3_gt"], encoding=enc).X.columns)
        assert a < b and b - a == set(GT_COLS)


def test_fs3_gt_has_25_tree_and_35_linear_columns(proxy_panel):
    fs = load_feature_sets()
    assert len(fs["FS3_activity"]) == 22
    assert len(fs["FS3_gt"]) == 25
    tree = _build(proxy_panel, "t10", 5, fs["FS3_gt"], encoding="int").X
    linear = _build(proxy_panel, "t10", 5, fs["FS3_gt"], encoding="onehot").X
    assert tree.shape[1] == 25
    assert linear.shape[1] == 35                                          # cal_month -> 11 dummies (+10)


def test_t10_gt_d0_passes_the_kappa_guard_at_zero_and_is_refused_at_one(proxy_panel):
    _, m = proxy_panel
    assert dataset.PREFIX["gt_yape_plin"] == "gt"
    f = _build(proxy_panel, "t10", 5, ["gt_d0"])
    assert list(f.X.columns) == ["gt_d0"]

    m1 = m.copy()
    m1.loc["gt_yape_plin", "kappa"] = 1
    with pytest.raises(ValueError, match="violates kappa"):
        _build(proxy_panel, "t10", 5, ["gt_d0"], meta=m1)
    # ...while gt_d1 is fine at kappa = 1.
    f1 = _build(proxy_panel, "t10", 5, ["gt_d1"], meta=m1)
    assert f1.X["gt_d1"].notna().all()


def test_gt_d1_and_gt_d12_are_lags_not_the_owners_ideas(proxy_panel):
    """gt_d12 in this codebase is the one-month change TWELVE MONTHS AGO, not yoy: the
    year-on-year idea is gt_ma12 (plan 4.5). Pinned so the renaming is not undone."""
    p, _ = proxy_panel
    lg = np.log(p["gt_yape_plin"])
    g = lg.diff()
    f = _build(proxy_panel, "t10", 5, ["gt_d0", "gt_d1", "gt_d12", "gt_ma12"])
    t = f.X.index
    np.testing.assert_allclose(f.X["gt_d1"], g.shift(1).reindex(t))
    np.testing.assert_allclose(f.X["gt_d12"], g.shift(12).reindex(t))
    np.testing.assert_allclose(f.X["gt_ma12"], ((lg - lg.shift(12)) / 12).reindex(t))


def test_gt_feature_definitions_match_plan_4_5(proxy_panel):
    p, _ = proxy_panel
    lg = np.log(p["gt_yape_plin"])
    f = _build(proxy_panel, "t10", 5, GT_COLS)
    t = f.X.index
    np.testing.assert_allclose(f.X["gt_d0"], (lg - lg.shift(1)).reindex(t), rtol=1e-12)
    np.testing.assert_allclose(f.X["gt_ma3"], ((lg - lg.shift(3)) / 3).reindex(t), rtol=1e-12)
    np.testing.assert_allclose(f.X["gt_ma12"], ((lg - lg.shift(12)) / 12).reindex(t), rtol=1e-12)


# --------------------------------------------------------------------------- #
# T7 -- scale invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", [0.01, 0.37, 2.0, 123.456])
def test_t7_rescaling_the_trends_index_leaves_the_features_unchanged(proxy_panel, c):
    p, m = proxy_panel
    base = _build(proxy_panel, "t10", 5, GT_COLS)
    q = p.copy()
    q["gt_yape_plin"] = q["gt_yape_plin"] * c
    scaled = dataset.build(q, m, target_id="t10", horizon=5, window="w2019",
                           columns=GT_COLS, encoding="int")
    pd.testing.assert_frame_equal(base.X, scaled.X, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# No leakage: a feature at origin t uses only months <= t
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("origin", ["2021-06-01", "2023-01-01", "2025-03-01"])
def test_gt_features_at_origin_t_use_only_months_up_to_t(proxy_panel, origin):
    p, m = proxy_panel
    t0 = pd.Timestamp(origin)
    base = _build(proxy_panel, "t10", 5, GT_COLS)

    q = p.copy()
    rng = np.random.default_rng(5)
    future = q.index > t0
    q.loc[future, "gt_yape_plin"] = rng.uniform(1, 190, int(future.sum()))   # rewrite EVERY later month
    pert = dataset.build(q, m, target_id="t10", horizon=5, window="w2019",
                         columns=GT_COLS, encoding="int")

    assert (base.X.index <= t0).any()
    pd.testing.assert_frame_equal(base.X.loc[base.X.index <= t0], pert.X.loc[pert.X.index <= t0])
    # ...and it IS sensitive to the present: rewriting month t0 itself must move the features at t0.
    r = p.copy()
    r.loc[t0, "gt_yape_plin"] = r.loc[t0, "gt_yape_plin"] * 1.7
    moved = dataset.build(r, m, target_id="t10", horizon=5, window="w2019",
                          columns=GT_COLS, encoding="int")
    assert not np.isclose(base.X.loc[t0, "gt_d0"], moved.X.loc[t0, "gt_d0"])


def test_bare_zeros_in_the_trends_index_make_the_build_raise_not_repair(proxy_panel):
    """The shipped pull has bare zeros until 2021: log differences are undefined, so building
    gt features must RAISE (naming the month), not floor, drop or smooth the zeros."""
    p, m = proxy_panel
    q = p.copy()
    q.loc["2017-01-01":"2021-01-01", "gt_yape_plin"] = 0.0
    with pytest.raises(ValueError, match="non-positive") as e:
        dataset.build(q, m, target_id="t10", horizon=5, window="w2019",
                      columns=GT_COLS, encoding="int")
    assert "2017-01" in str(e.value) and "G1" in str(e.value)


def test_a_snapshot_without_the_trends_column_says_what_to_run(proxy_panel):
    p, m = proxy_panel
    with pytest.raises(KeyError, match="snapshot"):
        dataset.build(p.drop(columns="gt_yape_plin"), m, target_id="t10", horizon=5,
                      window="w2019", columns=["gt_d0"], encoding="int")
