"""T1-T6: the six tests that must be green before any baseline runs.

Every one of these guards a failure that is SILENT -- a wrong number in a
feature matrix, not a crash. A leak found at roadmap step 11 invalidates
everything above it; these run in about a second.

See plan 9.0.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import pytest

from src.model_training import dataset, metrics, splits
from src.model_training.train import load_feature_sets

RNG = np.random.default_rng(26)


@pytest.fixture(scope="module")
def synthetic():
    """A panel with known properties. Offline, so no warehouse and no network."""
    idx = pd.date_range("2015-01-01", "2026-06-01", freq="MS")
    n = len(idx)
    trend = np.exp(np.linspace(np.log(10), np.log(700), n))
    season = 1 + 0.06 * np.sin(2 * np.pi * np.arange(n) / 12)
    panel = pd.DataFrame({
        "n_transf_intra_agg": trend * season * (1 + 0.01 * RNG.standard_normal(n)),
        "circulante": np.exp(np.linspace(np.log(40000), np.log(90000), n)),
        "pbi_idx": 100 + np.linspace(0, 60, n) + RNG.standard_normal(n),
        "ipc": np.exp(np.linspace(np.log(100), np.log(150), n)),
        "tipo_cambio": 3.3 + 0.2 * RNG.standard_normal(n),
        "ingreso_formal": np.exp(np.linspace(np.log(1500), np.log(2600), n)),
        "tasa_referencia": np.clip(2.75 + np.cumsum(0.05 * RNG.standard_normal(n)), 0.25, 8),
        "dolarizacion_liquidez": 30 + np.cumsum(0.1 * RNG.standard_normal(n)),
    }, index=idx)

    meta = pd.DataFrame({
        "col_name": panel.columns,
        # target kappa = 2, mirroring production after O-9 closed.
        "kappa": [2, 1, 2, 1, 0, 2, 0, 1],
        "transform": ["log_diff"] * 4 + ["log_diff"] * 2 + ["simple_diff"] * 2,
    }).set_index("col_name")
    return panel, meta


# --------------------------------------------------------------------------- #
# T1 -- calendar features are indexed at the TARGET month, not the origin
# --------------------------------------------------------------------------- #
def test_t1_calendar_indexed_at_target_month(synthetic):
    panel, meta = synthetic
    fs = load_feature_sets()
    kappa = int(meta.loc["n_transf_intra_agg", "kappa"])

    for h in (1, 3):
        frame = dataset.build(
            panel, meta, target_id="t2", horizon=h, window="w2019",
            columns=fs["FS0_calendar"], encoding="int",
        )
        # t - kappa + h, derived here the same way the code derives it. Writing
        # `h - 1` would re-hardcode the assumption O-9 disproved.
        target_months = frame.X.index + pd.DateOffset(months=h - kappa)
        expected = [calendar.monthrange(m.year, m.month)[1] for m in target_months]
        assert frame.X["cal_days"].tolist() == expected, f"cal_days wrong at h={h}"
        assert frame.X["cal_month"].tolist() == [m.month for m in target_months]

    # The specific bug this exists to catch: a February TARGET must not inherit
    # the origin month's length.
    frame = dataset.build(panel, meta, target_id="t2", horizon=3, window="w2019",
                          columns=fs["FS0_calendar"], encoding="int")
    febs = frame.X[(frame.X.index + pd.DateOffset(months=3 - kappa)).month == 2]
    assert set(febs["cal_days"]) <= {28, 29}, "February target month has the wrong length"
    assert not febs.empty


def test_target_month_follows_kappa(synthetic):
    """target month = t - kappa + h, and the anchor is exactly h months before it.

    This is the relationship O-9 broke. `n_transf_intra_agg` moved from kappa=1
    to kappa=2 on 2026-09-05, and a hardcoded `horizon - 1` would have kept the
    old convention silently: z would have spanned h-1 months while NaiveDrift
    still divided by h. Pinned here so the next calendar change fails loudly.
    """
    panel, meta = synthetic
    for kappa in (1, 2):
        m = meta.copy()
        m.loc["n_transf_intra_agg", "kappa"] = kappa
        for h in (1, 3):
            frame = dataset.build(panel, m, target_id="t2", horizon=h,
                                  window="w2019", columns=[], encoding="int")
            origins = frame.X.index
            expected_target = origins + pd.DateOffset(months=h - kappa)
            realised = panel["n_transf_intra_agg"].reindex(expected_target).to_numpy()
            np.testing.assert_allclose(frame.ctx["y_level"].to_numpy(), realised, rtol=1e-12)

            expected_anchor = origins - pd.DateOffset(months=kappa)
            anchored = panel["n_transf_intra_agg"].reindex(expected_anchor).to_numpy()
            np.testing.assert_allclose(frame.ctx["anchor"].to_numpy(), anchored, rtol=1e-12)

            # ...and therefore anchor and target are exactly h months apart.
            gap = (expected_target.to_period("M") - expected_anchor.to_period("M"))
            assert set(gap.map(lambda x: x.n)) == {h}


# --------------------------------------------------------------------------- #
# T2 -- no feature may reference a month later than t - kappa
# --------------------------------------------------------------------------- #
def test_t2_kappa_is_enforced(synthetic):
    panel, meta = synthetic
    # pbi_idx has kappa=2, so pbi_d1 is inadmissible and must RAISE, not warn.
    with pytest.raises(ValueError, match="violates kappa"):
        dataset.build(panel, meta, target_id="t2", horizon=1, window="w2019",
                      columns=["pbi_d1"], encoding="int")
    # kappa=0 series may use the origin month itself.
    frame = dataset.build(panel, meta, target_id="t2", horizon=1, window="w2019",
                          columns=["tc_d0", "tasa_d0"], encoding="int")
    assert frame.X.shape[1] == 2

    # And every shipped feature set must already comply.
    fs = load_feature_sets()
    for name in ("FS1_autoregressive", "FS3_activity", "FS5_macro_full", "FS5b_stance"):
        dataset.build(panel, meta, target_id="t2", horizon=1, window="w2019",
                      columns=fs[name], encoding="int")


# --------------------------------------------------------------------------- #
# T3 -- feature sets are strictly nested, in both encodings
# --------------------------------------------------------------------------- #
def test_t3_feature_sets_strictly_nested():
    fs = load_feature_sets()
    chain = ["FS0_calendar", "FS1_autoregressive", "FS2_cash",
             "FS3_activity", "FS4_prices", "FS5_macro_full", "FS5b_stance"]
    for a, b in zip(chain, chain[1:], strict=False):
        assert set(fs[a]) < set(fs[b]), f"{a} is not a strict subset of {b}"
    for name, cols in fs.items():
        assert len(cols) == len(set(cols)), f"{name} has duplicate columns"

    short = ["FS0_short", "FS1_short", "FS2_short"]
    for a, b in zip(short, short[1:], strict=False):
        assert set(fs[a]) < set(fs[b])
    # The _short variants exist to remove the 12-month terms. If one sneaks back
    # in, w2024 silently loses 43% of its rows.
    for name in short:
        assert not [c for c in fs[name] if c.endswith(("_d12", "_ma12"))]


# --------------------------------------------------------------------------- #
# T4 -- folds never overlap, and the purge gap is real
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window,horizon,expected", [
    ("w2019", 1, 29), ("w2019", 3, 25),
    ("w2021", 1, 17), ("w2021", 3, 13),
])
def test_t4_fold_counts_match_the_plan(window, horizon, expected):
    # Pure splits.py arithmetic on n_cv values copied from the plan. The n_cv
    # values themselves are NOT re-derived here: under target-month windows
    # (protocol 1.6, O-10) they change with the next snapshot, and the plan's
    # table is a marked TODO until then. This test pins the formula, not the data.
    n_cv = {("w2019", 1): 65, ("w2019", 3): 63,
            ("w2021", 1): 41, ("w2021", 3): 39}[(window, horizon)]
    folds = splits.make_folds(n_cv, window=window, horizon=horizon)
    assert len(folds) == expected, "fold arithmetic drifted from plan 2"


def test_t4_purge_gap_and_no_overlap():
    for horizon in (1, 3):
        purge = horizon - 1
        folds = splits.make_folds(65, window="w2019", horizon=horizon)
        for f in folds:
            assert not set(f.train) & set(f.test), "a row is in both train and test"
            assert f.train.max() + purge < f.test.min(), "purge gap violated"
            assert len(f.train) >= splits.MIN_TRAIN["w2019"]
        # Expanding, not rolling: training sets only grow.
        sizes = [len(f.train) for f in folds]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


# --------------------------------------------------------------------------- #
# T5 -- MASE: an exact in-sample identity, plus a loose out-of-sample band
# --------------------------------------------------------------------------- #
def test_t5_mase_identity_in_sample(synthetic):
    """Seasonal naive scored against the SAME rows as the denominator is 1.0.

    This is definitional, so any deviation is a bug in the denominator -- not a
    modelling result. v1.1 of the plan asked for '~1.0' out of sample, which is
    only true in sample; asserting it there would fail correct code.
    """
    panel, _ = synthetic
    levels = panel["n_transf_intra_agg"].to_numpy()
    y = levels[12:]
    yhat_seasonal = levels[:-12]
    assert metrics.mase(y, yhat_seasonal, levels) == pytest.approx(1.0, rel=1e-9)


def test_t5_mase_band_out_of_sample(synthetic):
    """Out of sample, seasonal-naive MASE is NOT pinned to 1 -- only sane.

    Asserted the way it is actually computed: an expanding training set ending
    immediately before each test origin, which is what splits.py produces.
    Taking the denominator from the distant past would test the fixture's
    growth rate, not the metric.
    """
    panel, _ = synthetic
    levels = panel["n_transf_intra_agg"].to_numpy()
    n = len(levels)
    values = [metrics.mase(levels[i:i + 1], levels[i - 12:i - 11], levels[:i])
              for i in range(n - 12, n)]
    assert 0.1 < float(np.mean(values)) < 20.0


def test_seasonal_reference_is_weak_on_a_trending_series(synthetic):
    """Documents a property of THIS series that changes how every result reads.

    The target grows roughly 40x across the sample, so "same month last year"
    is a poor forecast and "last month" is far better -- here by about 8x.
    Two consequences to carry into the write-up:

      * skill_h is defined against the seasonal naive, so it will look
        flattering for almost any model. A large positive skill_h is not on its
        own evidence that a model learned anything.
      * naive_calendar is designated "the hurdle" in plan 3, but on
        a series with this much drift the binding baseline is naive_drift.
        s1_baselines settles it empirically -- which is exactly why the protocol
        runs the floor before any model.
    """
    panel, _ = synthetic
    levels = panel["n_transf_intra_agg"].to_numpy()
    n = len(levels)
    seasonal = np.mean([metrics.mase(levels[i:i + 1], levels[i - 12:i - 11], levels[:i])
                        for i in range(n - 12, n)])
    randomwalk = np.mean([metrics.mase(levels[i:i + 1], levels[i - 1:i], levels[:i])
                          for i in range(n - 12, n)])
    assert randomwalk < seasonal


def test_t5_skill_signs():
    y = np.array([100.0, 110.0, 120.0])
    seasonal = np.array([90.0, 100.0, 110.0])
    assert metrics.skill(y, y, seasonal) == pytest.approx(1.0)          # perfect
    assert metrics.skill(y, seasonal, seasonal) == pytest.approx(0.0)   # tied
    assert metrics.skill(y, seasonal - 50, seasonal) < 0                # worse than naive


def test_reconstruct_inverts_the_target(synthetic):
    panel, meta = synthetic
    frame = dataset.build(panel, meta, target_id="t2", horizon=1, window="w2019",
                          columns=[], encoding="int")
    back = metrics.reconstruct(frame.ctx["anchor"].to_numpy(), frame.y.to_numpy())
    np.testing.assert_allclose(back, frame.ctx["y_level"].to_numpy(), rtol=1e-10)


# --------------------------------------------------------------------------- #
# T6 -- leak canary. Shuffle y; skill must not be positive.
# --------------------------------------------------------------------------- #
def test_t6_shuffled_target_has_no_skill(synthetic):
    """The cheapest insurance in the project.

    No per-module unit test can see information crossing the split end to end.
    This can: if a shuffled target still scores positive skill, something in the
    chain is reading the answer.

    THE REFERENCE MATTERS, and getting it wrong made this test fail on correct
    code first time round. Measured against the SEASONAL naive, a shuffled
    target still scores about 0.93 -- not because of a leak, but because
    reconstruction multiplies by the true anchor n_{t-1}, and on a series that
    grows 40x "last month" beats "same month last year" by roughly 8x. The
    anchor, not the model, was doing the work.

    So the canary compares against the RANDOM WALK (z_hat = 0), which shares
    that same anchor. Any advantage over it must come from the features, and a
    shuffled target has none to give.
    """
    from sklearn.linear_model import Ridge

    panel, meta = synthetic
    fs = load_feature_sets()
    frame = dataset.build(panel, meta, target_id="t2", horizon=1, window="w2019",
                          columns=fs["FS2_cash"], encoding="int")

    y_shuffled = RNG.permutation(frame.y.to_numpy())
    folds = splits.make_folds(len(frame.X) - 12, window="w2019", horizon=1)

    skills = []
    for f in folds:
        model = Ridge(alpha=1.0).fit(frame.X.to_numpy()[f.train], y_shuffled[f.train])
        z_hat = model.predict(frame.X.to_numpy()[f.test])
        ctx_te = frame.ctx.iloc[f.test]
        anchor = ctx_te["anchor"].to_numpy()
        yhat = metrics.reconstruct(anchor, z_hat)
        random_walk = metrics.reconstruct(anchor, np.zeros(len(anchor)))
        skills.append(metrics.skill(ctx_te["y_level"].to_numpy(), yhat, random_walk))

    assert np.nanmean(skills) <= 0.05, (
        f"mean skill {np.nanmean(skills):.3f} over a random walk on a SHUFFLED "
        "target. Information is crossing the train/test split."
    )


# --------------------------------------------------------------------------- #
# v1.4 -- Stage A, Stage C, SARIMAX, sweep expansion, resume key
# --------------------------------------------------------------------------- #
def _frame(synthetic, *, model="ridge", fs="FS1_autoregressive", h=1, window="w2019"):
    from src.model_training import registry

    panel, meta = synthetic
    cols = [] if fs == "none" else registry.model_columns(model, load_feature_sets()[fs])
    enc = "onehot" if model in registry.LINEAR_FAMILIES else "int"
    return dataset.build(panel, meta, target_id="t2" if h == 1 else "t3", horizon=h,
                         window=window, columns=cols, encoding=enc)


def test_tuning_runs_and_never_reaches_the_holdout(synthetic):
    """Regression: Stage A silently tuned NOTHING (n_candidates_evaluated=0 on every
    s0_smoke run) because inner folds were drawn from the first outer fold's 36 rows."""
    from src.model_training.train import HOLDOUT_MONTHS, RunConfig, _tune

    frame = _frame(synthetic)
    n_cv = len(frame.X) - HOLDOUT_MONTHS["w2019"]
    cfg = RunConfig("t2", 1, "w2019", "ridge", "FS1_autoregressive",
                    tune={"alpha": [0.1, 1.0, 10.0]})
    best, table = _tune(frame, cfg, n_cv)
    assert len(table) == 3, "the grid was not evaluated"
    assert table["n_ok_inner_folds"].min() > 0
    assert best["alpha"] in (0.1, 1.0, 10.0)


def test_tuning_preserves_python_types(synthetic):
    """A DataFrame row turns ints into floats and None into NaN: max_depth=3.0 and
    drift_window=nan. The winner must come back exactly as it went in."""
    from src.model_training.train import HOLDOUT_MONTHS, RunConfig, _tune

    frame = _frame(synthetic, model="naive_drift", fs="none")
    n_cv = len(frame.X) - HOLDOUT_MONTHS["w2019"]
    cfg = RunConfig("t2", 1, "w2019", "naive_drift", "none")
    best, _ = _tune(frame, cfg, n_cv)
    assert best["drift_window"] is None or isinstance(best["drift_window"], int)

    frame = _frame(synthetic, model="xgboost")
    cfg = RunConfig("t2", 1, "w2019", "xgboost", "FS1_autoregressive",
                    tune={"max_depth": [2, 3], "n_estimators": [20]})
    best, _ = _tune(frame, cfg, len(frame.X) - HOLDOUT_MONTHS["w2019"])
    assert type(best["max_depth"]) is int and type(best["n_estimators"]) is int


@pytest.mark.parametrize("h", [1, 3])
def test_holdout_folds_are_expanding_and_purged(h):
    n_rows, n_hold = 90, 12
    n_cv = n_rows - n_hold
    folds = splits.holdout_folds(n_rows, n_cv=n_cv, horizon=h)
    assert len(folds) == n_hold
    for f in folds:
        assert f.test[0] >= n_cv                          # every test origin is holdout
        assert f.train.max() + (h - 1) < f.test.min()     # purge respected
        assert len(np.intersect1d(f.train, f.test)) == 0
    assert folds[-1].train.max() > folds[0].train.max()   # expanding, not one fixed fit


def test_s2_spec_expands_to_the_planned_66_parents():
    from pathlib import Path

    import yaml

    from src.model_training import sweep

    spec = yaml.safe_load(Path("configs/sweeps/s2_proxy_grid.yaml").read_text(encoding="utf-8"))
    runs = sweep.expand(spec, "data/processed/panel_x.parquet")
    assert len(runs) == 66                                 # plan 7.2
    sar = [r for r in runs if r.model_family == "sarimax"]
    assert len(sar) == 6
    assert {r.feature_set for r in sar} == {"FS0_calendar", "FS1_autoregressive", "FS2_cash"}
    assert len({(r.target_id, r.model_family, r.feature_set, r.window) for r in runs}) == 66


def test_naive_models_run_once_regardless_of_feature_sets():
    from pathlib import Path

    import yaml

    from src.model_training import sweep

    spec = yaml.safe_load(Path("configs/sweeps/s1_baselines.yaml").read_text(encoding="utf-8"))
    assert len(sweep.expand(spec, "data/processed/panel_x.parquet")) == 20


def test_resume_key_encoding_matches_what_train_tags():
    """Regression: sweep looked up encoding='none' for runs train.py had tagged
    'int'/'onehot', so already_done() never matched a feature model."""
    from src.model_training.train import RunConfig, context_for

    for model, enc in [("ridge", "onehot"), ("elasticnet", "onehot"), ("svr_rbf", "onehot"),
                       ("xgboost", "int"), ("rf", "int"), ("sarimax", "int"),
                       ("naive_calendar", "none")]:
        fs = "none" if model.startswith("naive") else "FS1_autoregressive"
        ctx = context_for(RunConfig("t2", 1, "w2019", model, fs), "data/processed/panel_x.parquet")
        assert ctx.encoding == enc, model


def test_sarimax_exog_rule_is_fixed_and_capped():
    from src.model_training import registry

    fs = load_feature_sets()
    for name in ("FS0_calendar", "FS1_autoregressive", "FS2_cash"):
        cols = registry.model_columns("sarimax", fs[name])
        assert len(cols) <= registry.SARIMAX_MAX_EXOG
        assert not any(c.startswith("y_") or c == "cal_month" for c in cols)
    assert len(registry.model_columns("sarimax", fs["FS2_cash"])) == 8
    assert registry.model_columns("ridge", fs["FS2_cash"]) == fs["FS2_cash"]   # others untouched


def test_sarimax_fs0_is_pinned_to_no_arma_terms():
    from src.model_training import registry

    g0 = registry.grid_for("sarimax", "FS0_calendar")
    g1 = registry.grid_for("sarimax", "FS1_autoregressive")
    assert g0["p"] == [0] and g0["q"] == [0]
    assert g1["p"] == [0, 1, 2] and g1["q"] == [0, 1, 2]


def test_sarimax_forecasts_through_the_purge_gap():
    from src.model_training.registry import SarimaxAdapter

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3))
    y = 0.5 * X[:, 0] + 0.05 * rng.standard_normal(60)
    m = SarimaxAdapter(p=1, q=0, P=0, Q=0, D=0).fit(X[:50], y[:50])
    one = m.predict(X[50:51], None)
    gapped = m.predict(X[52:53], {"X_gap": X[50:52]})      # h=3: two purged rows between
    assert one.shape == (1,) and gapped.shape == (1,)
    assert np.isfinite(one).all() and np.isfinite(gapped).all()


def test_sarimax_end_to_end_one_fold(synthetic):
    """Fit and score through the real _score_fold, with a gap, offline."""
    from src.model_training import registry
    from src.model_training.train import _score_fold

    frame = _frame(synthetic, model="sarimax", fs="FS1_autoregressive", h=3)
    folds = splits.make_folds(len(frame.X) - 12, window="w2019", horizon=3)
    model = registry.build("sarimax", {"p": 1, "q": 0, "P": 0, "Q": 0, "D": 0}, horizon=3)
    m, yhat = _score_fold(model, frame, folds[0].train, folds[0].test)
    assert np.isfinite(m["mase"]) and np.isfinite(yhat).all()


# --------------------------------------------------------------------------- #
# v1.6 / O-10 -- windows bound the TARGET month; origin range derives from kappa
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def production_like(synthetic):
    """Mirrors the real snapshot: panel index runs to 2026-08 (macro is ahead of
    payments), payments are published only through 2026-06 (kappa=2, O-9)."""
    panel, meta = synthetic
    idx = pd.date_range("2015-01-01", "2026-08-01", freq="MS")
    p = panel.reindex(idx)
    p.loc["2026-07-01":, "n_transf_intra_agg"] = np.nan
    # kappa=0/1 series are observed through the panel edge; kappa=2 ones lag it.
    for col, _k in meta["kappa"].items():
        if col != "n_transf_intra_agg":
            p[col] = p[col].ffill()
    return p, meta


def _build(pl, h, window, cols=()):
    p, m = pl
    return dataset.build(p, m, target_id="t2" if h == 1 else "t3", horizon=h,
                         window=window, columns=list(cols), encoding="int")


@pytest.mark.parametrize("window,start", [("w2019", "2019-01-01"), ("w2021", "2021-01-01"),
                                          ("w2024", "2024-01-01")])
@pytest.mark.parametrize("h", [1, 3])
def test_window_bounds_the_target_month(production_like, window, start, h):
    """(i) Every row's TARGET month is inside the window and the first one IS the
    window start. With no lag features nothing forces a later start; with the
    FS1 lags the history lies before the window, so the start is still exact."""
    fs = load_feature_sets()
    for cols in ([], fs["FS1_autoregressive"]):
        f = _build(production_like, h, window, cols)
        tgt = f.X.index + pd.DateOffset(months=h - 2)
        assert tgt.min() == pd.Timestamp(start), "first target month is not the window start"
        assert (tgt >= pd.Timestamp(start)).all()
        assert f.target_start == tgt.min() and f.target_end == tgt.max()


def test_window_does_not_truncate_feature_history(production_like):
    """Lags may reach before the window start: the first row is complete, and its
    y_d12 comes from the panel a year before the window, not from NaN-dropping."""
    p, _ = production_like
    f = _build(production_like, 1, "w2024", ["y_d12"])
    first = f.X.index[0]
    assert first == pd.Timestamp("2024-02-01")              # target 2024-01 -> origin t+1 at h=1
    assert f.X.notna().all().all()
    ly = np.log(p["n_transf_intra_agg"])
    assert f.X["y_d12"].iloc[0] == pytest.approx(
        (ly - ly.shift(1)).loc[first - pd.DateOffset(months=12)])


@pytest.mark.parametrize("window", ["w2019", "w2021", "w2024"])
def test_holdout_is_the_same_target_months_at_both_horizons(production_like, window):
    """(ii) The final HOLDOUT_MONTHS rows are contiguous TARGET months and the
    SAME calendar months at h=1 and h=3. Under origin windows (<= 1.5) they were
    shifted by (h - kappa) and so differed between horizons."""
    from src.model_training.train import HOLDOUT_MONTHS

    n = HOLDOUT_MONTHS[window]
    hold = {}
    for h in (1, 3):
        f = _build(production_like, h, window)
        tgt = f.X.index[-n:] + pd.DateOffset(months=h - 2)
        assert len(tgt) == n
        assert (tgt.to_period("M")[1:] - tgt.to_period("M")[:-1]).map(lambda x: x.n).tolist() == [1] * (n - 1)
        hold[h] = list(tgt)
    assert hold[1] == hold[3]
    assert hold[1][-1] == pd.Timestamp("2026-06-01")        # last published month


@pytest.mark.parametrize("h", [1, 3])
def test_last_published_month_is_a_target_row(production_like, h):
    """(iii) The newest observed payments month (2026-06) is a target at BOTH
    horizons. The old origin bound (<= 2026-06) made h=1 end at 2026-05."""
    p, _ = production_like
    last_obs = p["n_transf_intra_agg"].last_valid_index()
    assert last_obs == pd.Timestamp("2026-06-01")
    f = _build(production_like, h, "w2019")
    assert f.target_end == last_obs
    assert f.ctx["y_level"].iloc[-1] == p.loc[last_obs, "n_transf_intra_agg"]


def test_protocol_version_is_1_7_and_the_legacy_specs_stay_1_6():
    """Protocol 1.7 is the current one; s0-s4 carry no `protocol_version` key and must keep
    producing 1.6 runs (they are not edited). See test_protocol_17.py for the 1.7 sweeps."""
    from pathlib import Path

    import yaml

    from src.model_training import sweep, tracking

    assert tracking.PROTOCOL_VERSION == "1.7"
    assert tracking.LEGACY_PROTOCOL_VERSION == "1.6"
    for name in ("s0_smoke", "s1_baselines", "s2_proxy_grid", "s4_wallet"):
        spec = yaml.safe_load(Path(f"configs/sweeps/{name}.yaml").read_text(encoding="utf-8"))
        assert "protocol_version" not in spec, f"{name} must stay untouched"
        assert {r.protocol_version for r in sweep.expand(spec, "data/processed/panel_x.parquet")} == {"1.6"}


# --------------------------------------------------------------------------- #
# O-12 -- the wallet window has no year-ago level early on, and 12 training
# months is one too few for the seasonal MASE scale
# --------------------------------------------------------------------------- #
WALLET_START = "2024-01-01"
# Month-of-year multiplier of the aggregate, known by construction so the
# seasonal-transfer tests can check that it is recovered.
SEAS_PATTERN = np.array([-0.06, -0.08, 0.00, -0.02, 0.01, 0.00, 0.03, 0.01, -0.01, 0.00, 0.02, 0.10])


@pytest.fixture(scope="module")
def wallet_panel():
    """31 wallet months (2024-01 .. 2026-07, as on the 2026-10-05 pull) on top of an
    aggregate that starts years earlier -- the shape of the real snapshot. The index
    runs two months PAST the last payments month, as in production where macro series
    are published sooner: without that, h=1 could not use 2026-07 as a target."""
    rng = np.random.default_rng(12)
    idx = pd.date_range("2015-01-01", "2026-09-01", freq="MS")
    n = len(idx)
    month = idx.month.to_numpy() - 1
    agg = np.exp(np.linspace(np.log(10), np.log(700), n) + SEAS_PATTERN[month]
                 + 0.005 * rng.standard_normal(n))
    p = pd.DataFrame({
        "n_transf_intra_agg": agg,
        "circulante": np.exp(np.linspace(np.log(40000), np.log(90000), n)),
    }, index=idx)
    w = np.asarray((idx >= pd.Timestamp(WALLET_START)) & (idx <= pd.Timestamp("2026-07-01")))
    k = int(w.sum())
    assert k == 31
    p.loc["2026-08-01":, "n_transf_intra_agg"] = np.nan     # payments end 2026-07
    base = np.exp(np.linspace(np.log(100), np.log(900), k) + SEAS_PATTERN[month[w]]
                  + 0.04 * rng.standard_normal(k))
    for col, share in (("n_transf_intra_yape", 1.0), ("n_transf_intra_plin", 0.3)):
        p[col] = np.nan
        p.loc[w, col] = base * share
    meta = pd.DataFrame({
        "col_name": p.columns, "kappa": [2, 1, 2, 2], "transform": ["log_diff"] * 4,
    }).set_index("col_name")
    return p, meta


def _wallet(wallet_panel, target, cols=()):
    p, m = wallet_panel
    h = dataset.TARGETS[target]["horizon"]
    return dataset.build(p, m, target_id=target, horizon=h, window="w2024",
                         columns=list(cols), encoding="int")


@pytest.mark.parametrize("target", ["t4", "t5"])
@pytest.mark.parametrize("fs", ["none", "FS0_short", "FS1_short", "FS2_short"])
def test_wallet_seasonal_reference_never_reaches_a_test_origin(wallet_panel, target, fs):
    """The early wallet rows DO have no year-ago level (the series starts 2024-01),
    but they are all training rows: every row with a missing seas_level sits below
    the first test index, for every fold of every feature set. This is the property
    that makes a NaN-handling fallback unnecessary -- and the reason _score_fold
    raises, rather than skips, if it ever stops holding."""
    cols = [] if fs == "none" else load_feature_sets()[fs]
    f = _wallet(wallet_panel, target, cols)
    h = dataset.TARGETS[target]["horizon"]
    n_cv = len(f.X) - 6
    missing = np.flatnonzero(f.ctx["seas_level"].isna().to_numpy())
    assert 1 <= len(missing) <= 11, "the wallet frame should start without a year-ago level"
    tgt = f.X.index + pd.DateOffset(months=h - 2)
    assert (tgt[missing] < pd.Timestamp("2025-01-01")).all()
    assert (tgt[~f.ctx["seas_level"].isna().to_numpy()] >= pd.Timestamp("2025-01-01")).all()
    for fold in splits.make_folds(n_cv, window="w2024", horizon=h):
        assert missing.max() < fold.test.min()


def test_seasonal_mase_scale_needs_more_than_twelve_levels():
    """Documents the defect O-12 found: with min_train = 12 the first outer fold has
    exactly 12 training levels and the m=12 scale is undefined -- ValueError, so
    every w2024 configuration used to fail on fold 0 before logging anything."""
    levels = np.linspace(100, 200, 13)
    with pytest.raises(ValueError, match="more than 12"):
        metrics.seasonal_naive_scale(levels[:12])
    assert metrics.seasonal_naive_scale(levels) == pytest.approx(100.0 * 12 / 12)  # one pair


def test_mase_period_falls_back_only_where_the_seasonal_scale_is_undefined():
    assert {w: metrics.mase_period(n) for w, n in splits.MIN_TRAIN.items()} == {
        "w2019": 12, "w2021": 12, "w2024": 1}


def test_mase_scale_is_unchanged_on_the_proxy_windows(synthetic):
    """The fallback must not move t2/t3: on w2019 the fold's MASE is the old m=12 one."""
    from src.model_training import registry
    from src.model_training.train import _score_fold

    frame = _frame(synthetic, model="naive_last", fs="none")
    fold = splits.make_folds(len(frame.X) - 12, window="w2019", horizon=1)[5]
    m, yhat = _score_fold(registry.build("naive_last", {}, horizon=1), frame, fold.train, fold.test)
    y = frame.ctx["y_level"].to_numpy()[fold.test]
    expected = metrics.mase(y, yhat, frame.ctx["y_level"].to_numpy()[fold.train], 12)
    assert m["mase"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("target", ["t4", "t5"])
def test_every_wallet_fold_is_scored_with_the_random_walk_scale(wallet_panel, target):
    """Every outer fold of every naive family and of ridge is finite, and the scale
    is the in-sample MAE of the random walk on that fold's training levels."""
    from src.model_training import registry
    from src.model_training.train import _score_fold

    h = dataset.TARGETS[target]["horizon"]
    for family, params, fs in [("naive_last", {}, "none"), ("naive_drift", {}, "none"),
                               ("naive_seasonal", {}, "none"), ("naive_seasdrift", {}, "none"),
                               ("naive_calendar", {}, "none"),
                               ("ridge", {"alpha": 10.0}, "FS1_short")]:
        cols = [] if fs == "none" else registry.model_columns(family, load_feature_sets()[fs])
        enc = "onehot" if family in registry.LINEAR_FAMILIES else "int"
        p, mt = wallet_panel
        f = dataset.build(p, mt, target_id=target, horizon=h, window="w2024",
                          columns=cols, encoding=enc)
        folds = splits.make_folds(len(f.X) - 6, window="w2024", horizon=h)
        assert folds, (family, fs)
        for fold in folds:
            m, yhat = _score_fold(registry.build(family, params, horizon=h), f,
                                  fold.train, fold.test)
            assert all(np.isfinite(v) for v in m.values()), (family, fold.index, m)
            lv = f.ctx["y_level"].to_numpy()[fold.train]
            scale = float(np.mean(np.abs(np.diff(lv))))
            y = f.ctx["y_level"].to_numpy()[fold.test]
            assert m["mase"] == pytest.approx(np.mean(np.abs(y - yhat)) / scale, rel=1e-12)


def test_score_fold_refuses_a_test_origin_without_a_seasonal_reference(wallet_panel):
    """If min_train ever drops below the number of NaN rows, fail loudly instead of
    letting nanmean in log_config skip the fold."""
    from src.model_training import registry
    from src.model_training.train import _score_fold

    f = _wallet(wallet_panel, "t4")
    assert np.isnan(f.ctx["seas_level"].iloc[3])
    with pytest.raises(ValueError, match="seasonal reference missing"):
        _score_fold(registry.build("naive_last", {}, horizon=1), f,
                    np.arange(0, 3), np.array([3]))


def test_naive_seasdrift_refuses_to_forecast_without_a_single_year_ago_pair(wallet_panel):
    from src.model_training import registry

    f = _wallet(wallet_panel, "t4")
    rows = np.arange(0, 5)                                   # all NaN seas_level
    assert f.ctx["seas_level"].iloc[rows].isna().all()
    with pytest.raises(ValueError, match="no training row"):
        registry.build("naive_seasdrift", {}, horizon=1).fit(None, None, f.ctx.iloc[rows])
    # 12 rows is enough: the wallet window's first fold has exactly one pair.
    m = registry.build("naive_seasdrift", {}, horizon=1).fit(None, None, f.ctx.iloc[np.arange(12)])
    assert np.isfinite(m.growth_)


def test_evaluation_status_follows_the_fold_count():
    from src.model_training import tracking

    assert [tracking.evaluation_status(n) for n in (0, 4, 7, 8, 9, 42)] == [
        "demonstration", "demonstration", "demonstration", "evaluation", "evaluation", "evaluation"]


@pytest.fixture
def local_mlflow(tmp_path):
    """A throwaway SQLite tracking store. NEVER the server in .env: these tests log."""
    import mlflow

    old = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    yield tmp_path
    mlflow.set_tracking_uri(old)


def _snapshot_files(tmp_path, wallet_panel):
    p, m = wallet_panel
    path = tmp_path / "panel_20261005T000000Z.parquet"
    p.to_parquet(path)
    m.reset_index().to_csv(tmp_path / "series_meta_20261005T000000Z.csv", index=False)
    return str(path)


def _log(cfg, tmp_path):
    import mlflow

    from src.model_training import tracking
    from src.model_training.train import context_for, fit_config, log_config

    exp = context_for(cfg, cfg.snapshot).experiment
    if mlflow.get_experiment_by_name(exp) is None:
        mlflow.create_experiment(exp, artifact_location=(tmp_path / "artifacts").as_uri())
    return tracking, mlflow, log_config(fit_config(cfg))


def test_wallet_run_logs_status_tag_and_fallback_params(local_mlflow, wallet_panel):
    """End to end on the wallet window, offline: the runs complete (they raised on
    fold 0 before O-12), carry evaluation_status, and say which scale they used."""
    from src.model_training.train import RunConfig

    snap = _snapshot_files(local_mlflow, wallet_panel)
    # t5, no features: targets 2024-04 .. 2026-07 = 28 rows, 22 CV, 22 - 12 - 2 = 8
    # folds -- exactly the threshold, so this one is an evaluation.
    cfg = RunConfig("t5", 3, "w2024", "naive_drift", "none", snapshot=snap)
    _, mlflow, res = _log(cfg, local_mlflow)
    run = mlflow.get_run(res.run_id)
    assert res.n_folds == 8
    assert run.data.tags["evaluation_status"] == "evaluation"
    assert "mlflow.note.content" not in run.data.tags
    assert run.data.params["mase_scale_period"] == "1"
    assert run.data.params["n_folds_nan_metric"] == "0"
    assert int(run.data.params["n_rows_seasonal_ref_missing"]) >= 1

    # t5 with FS1_short: first target 2024-07, 25 rows, 19 CV, 19 - 12 - 2 = 5 folds.
    cfg = RunConfig("t5", 3, "w2024", "ridge", "FS1_short", snapshot=snap,
                    tune={"alpha": [1.0, 10.0]})
    _, mlflow, res = _log(cfg, local_mlflow)
    run = mlflow.get_run(res.run_id)
    assert res.n_folds == 5
    assert run.data.tags["evaluation_status"] == "demonstration"
    assert "DEMONSTRATION" in run.data.tags["mlflow.note.content"]


def test_report_shows_the_evaluation_status_column(local_mlflow, wallet_panel):
    from src.model_training import report
    from src.model_training.train import RunConfig

    snap = _snapshot_files(local_mlflow, wallet_panel)
    _log(RunConfig("t5", 3, "w2024", "naive_drift", "none", snapshot=snap), local_mlflow)
    _log(RunConfig("t5", 3, "w2024", "ridge", "FS1_short", snapshot=snap,
                   tune={"alpha": [1.0, 10.0]}), local_mlflow)
    df = report.fetch("t5", 3)
    assert "evaluation_status" in df.columns
    got = dict(zip(df["model_family"], zip(df["n_folds"], df["evaluation_status"], strict=False), strict=False))
    assert got == {"naive_drift": (8, "evaluation"), "ridge": (5, "demonstration")}


# --------------------------------------------------------------------------- #
# Seasonal-transfer test (plan 5.2) -- seas_transfer, FS1s_short, FS2s_short
# --------------------------------------------------------------------------- #
def test_seas_transfer_sets_nest_strictly_over_their_parents():
    """T3, extended: FS_k_short is a strict subset of FS_ks_short, and the pair differs
    by exactly the one column -- that is what lets the delta be attributed to it."""
    fs = load_feature_sets()
    for parent, child in [("FS1_short", "FS1s_short"), ("FS2_short", "FS2s_short")]:
        assert set(fs[parent]) < set(fs[child])
        assert set(fs[child]) - set(fs[parent]) == {"seas_transfer"}
        assert len(fs[child]) == len(set(fs[child])) == len(fs[parent]) + 1
    assert set(fs["FS1s_short"]) < set(fs["FS2s_short"])
    assert set(fs["FS0_short"]) < set(fs["FS1s_short"])
    # It belongs to the transfer sets only: never smuggled into a control.
    others = [k for k in fs if k not in ("FS1s_short", "FS2s_short")]
    assert not [k for k in others if "seas_transfer" in fs[k]]
    # ...and, like the other short sets, carries no 12-month term.
    assert not [c for c in fs["FS2s_short"] if c.endswith(("_d12", "_ma12"))]


def test_seas_transfer_factors_sum_to_zero_and_recover_the_known_pattern(wallet_panel):
    p, _ = wallet_panel
    f = dataset.seasonal_transfer_factors(p)
    assert list(f.index) == list(range(1, 13))
    assert f.sum() == pytest.approx(0.0, abs=1e-12)
    # The fixture's level has a known month shape; the factor is the demeaned
    # month-on-month DIFFERENCE of it (it is a Delta-log), up to sampling noise.
    expected = SEAS_PATTERN - np.roll(SEAS_PATTERN, 1)           # month m minus month m-1
    assert np.allclose(f.to_numpy(), expected - expected.mean(), atol=0.02)


def test_seas_transfer_factors_ignore_everything_from_the_cutoff(wallet_panel):
    """THE leak test. Scramble the aggregate from 2024-01 on (the months every wallet
    fold lives in), before 2019, and inside the COVID pulse: the factors do not move.
    Scramble one month inside the sample and they do -- so the test can fail."""
    p, _ = wallet_panel
    base = dataset.seasonal_transfer_factors(p)
    rng = np.random.default_rng(99)

    def scrambled(mask):
        q = p.copy()
        q.loc[mask, dataset.SEAS_TRANSFER_SOURCE] *= np.exp(rng.standard_normal(int(mask.sum())))
        return dataset.seasonal_transfer_factors(q)

    idx = p.index
    for mask in (idx >= dataset.SEAS_TRANSFER_CUTOFF,
                 idx < dataset.SEAS_TRANSFER_FIRST_TARGET - pd.DateOffset(months=1),
                 (idx >= "2020-03-01") & (idx <= "2020-08-01")):
        assert scrambled(np.asarray(mask)).equals(base)
    assert not scrambled(np.asarray(idx == "2022-05-01")).equals(base)


@pytest.mark.parametrize("target", ["t4", "t5"])
def test_seas_transfer_is_indexed_at_the_target_month(wallet_panel, target):
    h = dataset.TARGETS[target]["horizon"]
    p, _ = wallet_panel
    f = _wallet(wallet_panel, target, ["cal_month", "seas_transfer"])
    factors = dataset.seasonal_transfer_factors(p)
    tgt_month = (f.X.index + pd.DateOffset(months=h - 2)).month
    assert (f.X["cal_month"].to_numpy() == tgt_month.to_numpy()).all()
    assert np.allclose(f.X["seas_transfer"].to_numpy(), factors.loc[tgt_month].to_numpy())
    # The h=3 trap, as in T1: a February target gets February's factor even though the
    # origin is January.
    feb = np.flatnonzero(tgt_month == 2)
    assert len(feb) and (f.X["seas_transfer"].to_numpy()[feb] == factors.loc[2]).all()


def test_seas_transfer_uses_the_aggregate_and_records_its_provenance(wallet_panel):
    p, _ = wallet_panel
    for target in ("t4", "t5"):
        f = _wallet(wallet_panel, target, load_feature_sets()["FS2s_short"])
        assert f.seas_transfer["source"] == "n_transf_intra_agg"
        assert f.seas_transfer["first_target"] == "2019-01"
        assert f.seas_transfer["cutoff"] == "2024-01"
        assert f.seas_transfer["excludes"] == "2020-03..2020-09"
        want = dataset.seasonal_transfer_factors(p)
        assert np.allclose([f.seas_transfer["factors"][m] for m in range(1, 13)], want.to_numpy())
    assert _wallet(wallet_panel, "t4", load_feature_sets()["FS1_short"]).seas_transfer is None


def test_seas_transfer_changes_columns_not_rows(wallet_panel):
    """Same rows with and without it, so a with/without comparison runs on the same folds."""
    fs = load_feature_sets()
    for target in ("t4", "t5"):
        a = _wallet(wallet_panel, target, fs["FS1_short"])
        b = _wallet(wallet_panel, target, fs["FS1s_short"])
        assert a.X.index.equals(b.X.index) and a.y.equals(b.y)
        assert b.X.shape[1] == a.X.shape[1] + 1


@pytest.mark.parametrize("window", ["w2019", "w2021"])
def test_seas_transfer_is_refused_where_it_would_leak(synthetic, window):
    panel, meta = synthetic
    with pytest.raises(ValueError, match="only admissible"):
        dataset.build(panel, meta, target_id="t2", horizon=1, window=window,
                      columns=["cal_days", "seas_transfer"], encoding="int")


def test_seas_transfer_runs_log_their_factor_source_and_cutoff(local_mlflow, wallet_panel):
    """Every run that uses the feature carries its source and cutoff as params; a run
    that does not use it carries none."""
    import json
    from pathlib import Path

    import mlflow

    from src.model_training.train import RunConfig

    snap = _snapshot_files(local_mlflow, wallet_panel)
    with_t = RunConfig("t4", 1, "w2024", "ridge", "FS1s_short", snapshot=snap,
                       tune={"alpha": [1.0, 10.0]})
    without = RunConfig("t4", 1, "w2024", "ridge", "FS1_short", snapshot=snap,
                        tune={"alpha": [1.0, 10.0]})
    _, _, r1 = _log(with_t, local_mlflow)
    _, _, r0 = _log(without, local_mlflow)

    pr = mlflow.get_run(r1.run_id).data.params
    assert pr["seas_transfer_source"] == "n_transf_intra_agg"
    assert pr["seas_transfer_cutoff"] == "2024-01"
    assert pr["seas_transfer_first_target"] == "2019-01"
    assert pr["seas_transfer_excludes"] == "2020-03..2020-09"
    assert int(pr["n_features"]) == int(mlflow.get_run(r0.run_id).data.params["n_features"]) + 1
    assert not [k for k in mlflow.get_run(r0.run_id).data.params if k.startswith("seas_transfer")]

    path = mlflow.artifacts.download_artifacts(run_id=r1.run_id, artifact_path="features.json")
    feats = json.loads(Path(path).read_text())
    assert "seas_transfer" in feats["columns"]
    assert len(feats["seas_transfer"]["factors"]) == 12
    assert sum(feats["seas_transfer"]["factors"].values()) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# s4_wallet -- spec, reduced w2024 grids, feasibility of every configuration
# --------------------------------------------------------------------------- #
def _s4_runs():
    from pathlib import Path

    import yaml

    from src.model_training import sweep

    spec = yaml.safe_load(Path("configs/sweeps/s4_wallet.yaml").read_text(encoding="utf-8"))
    return sweep.expand(spec, "data/processed/panel_x.parquet")


def test_s4_spec_expands_to_the_reconciled_30_parents():
    """Plan 7.2: 5 naive x 2 targets + (ridge, xgboost) x 5 feature sets x 2 targets.
    The plan's old '18' counted one naive per feature set (3 x 3 x 2) and no transfer sets."""
    from src.model_training import registry

    runs = _s4_runs()
    assert len(runs) == 30
    assert len({(r.target_id, r.model_family, r.feature_set, r.window) for r in runs}) == 30
    assert {r.window for r in runs} == {"w2024"}
    assert {(r.target_id, r.horizon) for r in runs} == {("t4", 1), ("t5", 3)}
    naive = [r for r in runs if r.model_family in registry.NAIVE_FAMILIES]
    assert len(naive) == 10 and {r.feature_set for r in naive} == {"none"}
    assert {r.model_family for r in naive} == {"naive_last", "naive_drift", "naive_seasonal",
                                                "naive_seasdrift", "naive_calendar"}
    for fam in ("ridge", "xgboost"):
        sets = {r.feature_set for r in runs if r.model_family == fam}
        assert sets == {"FS0_short", "FS1_short", "FS2_short", "FS1s_short", "FS2s_short"}
    # naive_drift must be present for BOTH targets: report.py derives the hurdle from it.
    assert {r.target_id for r in runs if r.model_family == "naive_drift"} == {"t4", "t5"}
    # Every feature set a w2024 run asks for is a _short one (never a 12-month term).
    assert all(r.feature_set == "none" or "short" in r.feature_set for r in runs)


def test_w2024_grids_are_reduced_and_the_proxy_grids_are_untouched():
    import math

    from src.model_training import registry

    size = lambda g: math.prod(len(v) for v in g.values())          # noqa: E731
    assert size(registry.grid_for("ridge", "FS1_short", None, "w2024")) == 5
    assert size(registry.grid_for("xgboost", "FS1_short", None, "w2024")) == 16
    # Not a single value may fall outside the full menu, except where the plan reduced it.
    assert set(registry.W2024_GRIDS["xgboost"]["max_depth"]) <= {2, 3, 4}
    # t2/t3 are byte-for-byte what they were, with or without a window argument.
    for fam, n in (("ridge", 13), ("elasticnet", 66), ("svr_rbf", 36), ("rf", 54),
                   ("xgboost", 288), ("sarimax", 72)):
        for w in (None, "w2019", "w2021"):
            assert size(registry.grid_for(fam, "FS3_activity", None, w)) == n, (fam, w)
    assert registry.grid_for("xgboost", "FS3_activity") == registry.default_grid("xgboost")
    # A sweep spec's `tune:` still wins; naive models keep their own tiny grid.
    assert registry.grid_for("ridge", "FS1_short", {"alpha": [1.0]}, "w2024") == {"alpha": [1.0]}
    assert registry.grid_for("naive_drift", "none", None, "w2024") == registry.default_grid("naive_drift")
    # A family with no reduced grid is refused on w2024 rather than given a full one.
    for fam in ("elasticnet", "svr_rbf", "rf", "sarimax"):
        with pytest.raises(ValueError, match="No reduced w2024 grid"):
            registry.grid_for(fam, "FS1_short", None, "w2024")


@pytest.mark.parametrize("run", _s4_runs(), ids=lambda r: f"{r.target_id}-{r.model_family}-{r.feature_set}")
def test_every_s4_configuration_has_folds_to_tune_and_to_score(wallet_panel, run):
    """No s4 config may die in fit_config on a fold-count error three hours in: each has
    CV rows above min_train, at least one inner fold for Stage A, and at least one outer
    fold. Offline, on the synthetic wallet panel (31 months, as on the 2026-10-05 pull)."""
    from src.model_training import registry
    from src.model_training.train import HOLDOUT_MONTHS

    p, m = wallet_panel
    naive = run.model_family in registry.NAIVE_FAMILIES
    cols = [] if naive else registry.model_columns(run.model_family, load_feature_sets()[run.feature_set])
    f = dataset.build(p, m, target_id=run.target_id, horizon=run.horizon, window="w2024",
                      columns=cols, encoding="int")
    n_cv = len(f.X) - HOLDOUT_MONTHS["w2024"]
    mt = splits.MIN_TRAIN["w2024"]
    assert n_cv > mt
    assert splits.inner_folds(n_cv, horizon=run.horizon, min_train=mt)
    outer = splits.make_folds(n_cv, window="w2024", horizon=run.horizon)
    assert len(outer) == n_cv - mt - (run.horizon - 1)
    assert f.target_end == pd.Timestamp("2026-07-01")


@pytest.mark.parametrize("model,n_cands", [("ridge", 5), ("xgboost", 16)])
def test_stage_a_uses_the_reduced_grid_on_the_wallet_window(wallet_panel, model, n_cands):
    """Through the real _tune, which must hand cfg.window to grid_for."""
    from src.model_training import registry
    from src.model_training.train import HOLDOUT_MONTHS, RunConfig, _tune

    p, m = wallet_panel
    cols = registry.model_columns(model, load_feature_sets()["FS1s_short"])
    enc = "onehot" if model in registry.LINEAR_FAMILIES else "int"
    f = dataset.build(p, m, target_id="t5", horizon=3, window="w2024", columns=cols, encoding=enc)
    cfg = RunConfig("t5", 3, "w2024", model, "FS1s_short")
    best, table = _tune(f, cfg, len(f.X) - HOLDOUT_MONTHS["w2024"])
    assert len(table) == n_cands
    assert table["n_ok_inner_folds"].min() == table["n_inner_folds"].max() >= 4
