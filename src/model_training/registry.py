"""What a model IS. One entry per family; train.py never names a model.

Two kinds of estimator, behind one interface:

  * CONTEXT models (the naive family) predict from ctx -- the anchor level, the
    seasonal reference, the month lengths. They have no features and nothing
    to tune.
  * FEATURE models (everything else) are sklearn estimators fitted on X.

train.py checks `uses_context` and passes the right thing. Because both sides
produce z on the same rows and are scored by the same metrics.py, the
comparison between "last month plus trend" and a tuned XGBoost is genuinely
like for like -- which is the entire point of having a floor.

Adding a seventh family should be one entry here plus one YAML file, and zero
lines of train.py. If it is not, the abstraction is wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --------------------------------------------------------------------------- #
# Naive family -- the floor (plan 3, Target 1)
# --------------------------------------------------------------------------- #


class _Naive:
    uses_context = True

    def fit(self, X, y, ctx):  # noqa: D102
        return self

    def predict(self, X, ctx) -> np.ndarray:  # noqa: D102
        raise NotImplementedError


class NaiveLast(_Naive):
    """Random walk: next month equals this month. z = 0."""

    def predict(self, X, ctx):
        return np.zeros(len(ctx))


class NaiveDrift(_Naive):
    """Persistence plus average trend. z = h * mean(dlog n) over training."""

    def __init__(self, horizon: int = 1, drift_window: int | None = None):
        self.horizon, self.drift_window = horizon, drift_window

    def fit(self, X, y, ctx):
        d = np.log(ctx["y_level"].to_numpy()) - np.log(ctx["anchor"].to_numpy())
        if self.drift_window:
            d = d[-self.drift_window:]
        # y_level/anchor already spans h steps, so this is drift PER STEP.
        self.drift_ = float(np.nanmean(d)) / self.horizon
        return self

    def predict(self, X, ctx):
        return np.full(len(ctx), self.horizon * self.drift_)


class NaiveSeasonal(_Naive):
    """Same month last year. z = log n_{m-12} - log n_{t-1}."""

    def predict(self, X, ctx):
        return np.log(ctx["seas_level"].to_numpy()) - np.log(ctx["anchor"].to_numpy())


class NaiveSeasonalDrift(_Naive):
    """Seasonal naive plus the average 12-month log growth seen in training."""

    def fit(self, X, y, ctx):
        g = np.log(ctx["y_level"].to_numpy()) - np.log(ctx["seas_level"].to_numpy())
        # Wallet frames carry NaN seas_level on their first rows (O-12): those
        # rows contribute no pair and are skipped. Skipping is the point; having
        # NO pair at all is not something to turn into a NaN forecast.
        g = g[np.isfinite(g)]
        if not len(g):
            raise ValueError("naive_seasdrift: no training row has a year-ago level")
        self.growth_ = float(np.mean(g))
        return self

    def predict(self, X, ctx):
        base = np.log(ctx["seas_level"].to_numpy()) - np.log(ctx["anchor"].to_numpy())
        return base + self.growth_


class NaiveCalendar(_Naive):
    """THE HURDLE. Seasonal naive scaled by month length.

    February has ~10.7% fewer days than March. In a transaction count that is a
    mechanical swing with zero economic content, and adjusting for it is free.
    If a tuned gradient-boosted model cannot beat this, that IS the headline
    finding and it gets reported as such (plan 6.4 rule 2).
    """

    def predict(self, X, ctx):
        ratio = ctx["cal_days_m"].to_numpy() / ctx["cal_days_m12"].to_numpy()
        return (
            np.log(ctx["seas_level"].to_numpy() * ratio)
            - np.log(ctx["anchor"].to_numpy())
        )


# --------------------------------------------------------------------------- #
# Feature models
# --------------------------------------------------------------------------- #


class SklearnAdapter:
    """Wraps an sklearn estimator so train.py can treat every model the same."""

    uses_context = False

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, ctx=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, ctx=None):
        return self.estimator.predict(X)


class EnsembleAdapter:
    """Equal-weight mean of the members' z-forecasts (plan 7.4, `ens3`).

    A model FAMILY, not post-processing: it is tagged, resumed and ranked like any
    other. Each member keeps its OWN encoding of the same feature set (the SVR gets
    the one-hot calendar, the trees the integer month) and its own scaler, so unlike
    every other estimator this one is fed a dict {member family: matrix} rather than
    one matrix -- train._score_fold prepares each member's matrix from the frame in
    that member's encoding.

    Equal weights are fixed by the plan, not estimated: ~40 folds would fit noise.
    The mean is taken on z (the differenced target); train reconstructs to levels
    ONCE, from the mean (plan 1.2). `member_z_` keeps each member's own forecast of
    the last predict() call so the ensemble can be scored against its members.
    """

    uses_context = False
    is_ensemble = True

    def __init__(self, members: list[tuple[str, SklearnAdapter]]):
        self.members = members
        self.member_z_: dict[str, np.ndarray] = {}

    def fit(self, X_by_member: dict, y, ctx=None):
        for family, est in self.members:
            est.fit(X_by_member[family], y)
        return self

    def predict(self, X_by_member: dict, ctx=None) -> np.ndarray:
        self.member_z_ = {
            family: np.asarray(est.predict(X_by_member[family]), dtype=float)
            for family, est in self.members
        }
        return np.mean(np.vstack(list(self.member_z_.values())), axis=0)


class SarimaxAdapter:
    """SARIMAX on the differenced target z, with the feature set as exogenous input.

    The econometrician's model (plan 7.1). z is ALREADY a log
    difference, so d = 0 (7.2 grid) and the seasonal difference D in {0, 1} is the
    only differencing left to choose.

    WHY `wants_gap`. At h=3 the purge removes two origins between the end of
    training and the test origin, so a one-step-ahead forecast would silently
    treat the test origin as immediately following the last training row. The
    model is instead asked for (purge + 1) steps and the LAST is used. The
    intermediate exogenous rows are the features of the gap origins -- known at
    forecast time (they are built from information at those origins' closes), so
    using them leaks nothing; only their TARGETS are withheld. At h=1 the gap is
    empty and this is an ordinary one-step forecast.
    """

    uses_context = False
    wants_gap = True

    def __init__(self, p=0, q=0, P=0, Q=0, D=0):
        self.p, self.q, self.P, self.Q, self.D = int(p), int(q), int(P), int(Q), int(D)

    def fit(self, X, y, ctx=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # A constant under D=1 is a seasonal-drift term that the data cannot
        # separate from the seasonal differencing; leave it out there.
        trend = "c" if self.D == 0 else "n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                np.asarray(y, dtype=float), exog=np.asarray(X, dtype=float),
                order=(self.p, 0, self.q), seasonal_order=(self.P, self.D, self.Q, 12),
                trend=trend, enforce_stationarity=False, enforce_invertibility=False,
            )
            self.res_ = mod.fit(disp=False, maxiter=100)
        return self

    def predict(self, X, ctx=None):
        X = np.asarray(X, dtype=float)
        gap = None if ctx is None else ctx.get("X_gap")
        exog = X if gap is None or len(gap) == 0 else np.vstack([gap, X])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = np.asarray(self.res_.forecast(steps=len(exog), exog=exog))
        return fc[-len(X):]


# SARIMAX exogenous columns, fixed before any SARIMAX run (plan 11).
# The plan says "exog capped at 8 columns" without saying which 8. Rule:
#   * drop y_*       -- own history is the ARMA terms' job, and tuning p, q is how
#                       the persistence ablation (FS0 -> FS1) is expressed here;
#   * drop cal_month -- the seasonal terms (P, D, Q at lag 12) own the month effect,
#                       and an integer month is meaningless to a linear model;
#   * keep the first 8 of what remains, in the order feature_sets.yaml declares.
SARIMAX_MAX_EXOG = 8


def model_columns(model_family: str, columns: list[str]) -> list[str]:
    """The columns a model actually receives from a feature set."""
    if model_family != "sarimax":
        return columns
    keep = [c for c in columns if not c.startswith("y_") and c != "cal_month"]
    return keep[:SARIMAX_MAX_EXOG]


def _sarimax(**kw):
    return SarimaxAdapter(**kw)


def _ridge(**kw):
    return SklearnAdapter(Ridge(**kw))


def _elasticnet(**kw):
    return SklearnAdapter(ElasticNet(max_iter=20000, **kw))


def _svr(**kw):
    return SklearnAdapter(SVR(kernel="rbf", **kw))


def _rf(**kw):
    return SklearnAdapter(RandomForestRegressor(random_state=26, n_jobs=1, **kw))


def _xgb(**kw):
    # CPU 'hist', deliberately. On a 65 x 30 matrix CUDA kernel-launch overhead
    # exceeds the fit time, so device="cuda" is measurably SLOWER here.
    # n_jobs=1 because sweep.py parallelises across CONFIGURATIONS; nested
    # parallelism oversubscribes 8 physical cores and slows everything down.
    return SklearnAdapter(
        XGBRegressor(tree_method="hist", device="cpu", n_jobs=1, random_state=26, **kw)
    )


# name -> (factory, default search space). A sweep spec's `tune:` block narrows
# or overrides the space per model (sweep.py); this is the full menu.
MODELS: dict[str, tuple] = {
    "naive_last": (lambda **kw: NaiveLast(), {}),
    "naive_drift": (lambda **kw: NaiveDrift(**kw), {"drift_window": [None, 12, 24]}),
    "naive_seasonal": (lambda **kw: NaiveSeasonal(), {}),
    "naive_seasdrift": (lambda **kw: NaiveSeasonalDrift(), {}),
    "naive_calendar": (lambda **kw: NaiveCalendar(), {}),
    "ridge": (_ridge, {"alpha": list(np.logspace(-3, 3, 13))}),
    "elasticnet": (
        _elasticnet,
        {"alpha": list(np.logspace(-3, 2, 11)), "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]},
    ),
    "svr_rbf": (
        _svr,
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.001, 0.01, 0.05]},
    ),
    "rf": (
        _rf,
        {
            "n_estimators": [300, 800], "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 3, 5], "max_features": [0.3, 0.6, 1.0],
        },
    ),
    "sarimax": (
        _sarimax,
        {"p": [0, 1, 2], "q": [0, 1, 2], "P": [0, 1], "Q": [0, 1], "D": [0, 1]},
    ),
    "xgboost": (
        _xgb,
        {
            "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 600], "subsample": [0.7, 1.0],
            "colsample_bytree": [0.5, 0.8], "min_child_weight": [1, 5],
            "reg_lambda": [1, 10],
        },
    ),
}

# `ens3` (plan 7.4): the members, in the order they are listed everywhere
# (tuning table, member_* params, member_z_). Each member is tuned in its own Stage A.
ENSEMBLES: dict[str, tuple[str, ...]] = {"ens3": ("svr_rbf", "rf", "xgboost")}


def _ensemble(name: str):
    def factory(**member_params):
        """member_params = {member family: that member's frozen hyperparameters}."""
        missing = [m for m in ENSEMBLES[name] if m not in member_params]
        if missing:
            raise ValueError(f"{name}: no hyperparameters for member(s) {missing}")
        return EnsembleAdapter([(m, MODELS[m][0](**member_params[m])) for m in ENSEMBLES[name]])
    return factory


# Default grid is empty on purpose: an ensemble has no search space of its own, its
# members do (grid_for / train._tune_ensemble tune each one separately).
for _name in ENSEMBLES:
    MODELS[_name] = (_ensemble(_name), {})

NAIVE_FAMILIES = {k for k in MODELS if k.startswith("naive")}


def build(model_family: str, params: dict, *, horizon: int):
    factory, _ = MODELS[model_family]
    if model_family == "naive_drift":
        params = {**params, "horizon": horizon}
    return factory(**params)


def default_grid(model_family: str) -> dict:
    return MODELS[model_family][1]


# w2024 search spaces (plan 7.2 row 13). The wallet window has 18-22
# CV rows and its Stage A has 4-8 inner folds, so the full grids (xgboost: 288
# candidates, ridge: 13 alphas) would pick a winner from noise and spend most of
# their evaluations differentiating configurations the data cannot tell apart.
# Fixed here, before any wallet run. Only the families s4_wallet uses have one:
# asking for another on w2024 raises rather than silently using a full grid.
#   ridge   5 alphas, 0.1 .. 1000: weak shrinkage up to "almost the intercept",
#           which is where a 12-row fit may well belong.
#   xgboost 16 candidates: depth 2 only (a depth-3 tree has more leaves than a
#           12-row fold has rows to fill), min_child_weight 1 / 3 (5 forbids
#           nearly every split at n = 12), fewer rounds than the full grid.
W2024_GRIDS: dict[str, dict] = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "xgboost": {
        "max_depth": [2], "learning_rate": [0.05, 0.1], "n_estimators": [100, 300],
        "subsample": [1.0], "colsample_bytree": [0.8], "min_child_weight": [1, 3],
        "reg_lambda": [1, 10],
    },
}


def grid_for(
    model_family: str, feature_set: str, override: dict | None = None,
    window: str | None = None,
) -> dict:
    """The search space for one configuration.

    Precedence: a sweep spec's `tune:` override, then the reduced w2024 grid for a
    feature model on that window, then the full default grid. Naive models keep
    their (tiny) default grid everywhere -- it is the hurdle's own tuning and is
    the same on every window.

    SARIMAX on FS0_calendar is pinned to p = q = 0. FS0 is "how much is pure
    calendar" and FS1 is "how much is persistence"; if the ARMA terms were free
    on FS0, the two sets would differ only in exogenous columns that SARIMAX drops
    anyway (model_columns), and the ablation would compare a model with itself.
    """
    if override:
        grid = dict(override)
    elif window == "w2024" and model_family not in NAIVE_FAMILIES:
        if model_family not in W2024_GRIDS:
            raise ValueError(
                f"No reduced w2024 grid is registered for {model_family!r} "
                f"(have {sorted(W2024_GRIDS)}): its full grid cannot be supported by "
                "~20 CV rows. Register one in registry.W2024_GRIDS, in the plan, first."
            )
        grid = dict(W2024_GRIDS[model_family])
    else:
        grid = dict(default_grid(model_family))
    if model_family == "sarimax" and feature_set.startswith("FS0"):
        grid["p"], grid["q"] = [0], [0]
    return grid


# What the encoding tag says. Defined once so that train.py (which writes the tag)
# and sweep.py (which looks it up to decide whether a run already exists) cannot
# disagree. They did, once: sweep looked for "none" on models that were tagged
# "int" or "onehot", so resumability never matched a feature model.
LINEAR_FAMILIES = {"ridge", "elasticnet", "svr_rbf"}


# An ensemble mixes encodings (SVR one-hot, trees integer), so its own tag says so; the
# members' encodings are what default_encoding gives them individually.
MIXED_ENCODING = "mixed"


def default_encoding(model_family: str) -> str:
    if model_family in NAIVE_FAMILIES:
        return "none"
    if model_family in ENSEMBLES:
        return MIXED_ENCODING
    return "onehot" if model_family in LINEAR_FAMILIES else "int"


def member_encodings(model_family: str) -> dict[str, str]:
    """{member: encoding} for an ensemble."""
    return {m: default_encoding(m) for m in ENSEMBLES[model_family]}
