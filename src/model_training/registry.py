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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

# --------------------------------------------------------------------------- #
# Naive family -- the floor (training_plan.md 3, Target 1)
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
        self.growth_ = float(np.nanmean(g))
        return self

    def predict(self, X, ctx):
        base = np.log(ctx["seas_level"].to_numpy()) - np.log(ctx["anchor"].to_numpy())
        return base + self.growth_


class NaiveCalendar(_Naive):
    """THE HURDLE. Seasonal naive scaled by month length.

    February has ~10.7% fewer days than March. In a transaction count that is a
    mechanical swing with zero economic content, and adjusting for it is free.
    If a tuned gradient-boosted model cannot beat this, that IS the headline
    finding and it gets reported as such (training_plan.md 6.4 rule 2).
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


# name -> (factory, default search space). The YAML in configs/models/ narrows
# or overrides the space; this is the full menu.
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

NAIVE_FAMILIES = {k for k in MODELS if k.startswith("naive")}


def build(model_family: str, params: dict, *, horizon: int):
    factory, _ = MODELS[model_family]
    if model_family == "naive_drift":
        params = {**params, "horizon": horizon}
    return factory(**params)


def default_grid(model_family: str) -> dict:
    return MODELS[model_family][1]
