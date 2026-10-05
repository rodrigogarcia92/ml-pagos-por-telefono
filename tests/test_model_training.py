"""T1-T6: the six tests that must be green before any baseline runs.

Every one of these guards a failure that is SILENT -- a wrong number in a
feature matrix, not a crash. A leak found at roadmap step 11 invalidates
everything above it; these run in about a second.

See docs/training_plan.md 9.0.
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
    for a, b in zip(chain, chain[1:]):
        assert set(fs[a]) < set(fs[b]), f"{a} is not a strict subset of {b}"
    for name, cols in fs.items():
        assert len(cols) == len(set(cols)), f"{name} has duplicate columns"

    short = ["FS0_short", "FS1_short", "FS2_short"]
    for a, b in zip(short, short[1:]):
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
    n_cv = {("w2019", 1): 65, ("w2019", 3): 63,
            ("w2021", 1): 41, ("w2021", 3): 39}[(window, horizon)]
    folds = splits.make_folds(n_cv, window=window, horizon=horizon)
    assert len(folds) == expected, "fold arithmetic drifted from training_plan.md 2"


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
      * naive_calendar is designated "the hurdle" in training_plan.md 3, but on
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
