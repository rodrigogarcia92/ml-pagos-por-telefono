"""Metrics, all computed on RECONSTRUCTED LEVELS.

Models are fitted on z = log n_{t+h-1} - log n_{t-1}. Every metric here first
undoes that. A metric on the differenced scale is not comparable to the naive
baselines and means nothing to a business reader (docs/training_plan.md 1.2).

Why the two unfamiliar ones exist: the target grows about 75x across the sample,
so an MAE of two million operations is a catastrophe in 2019 and a rounding
error in 2026. Averaging that across folds produces a number with no meaning.
Both MASE and skill_h divide by an error measured on the same data, which
removes the scale.
"""

from __future__ import annotations

import numpy as np

SEASONAL_PERIOD = 12


def mase_period(min_train: int) -> int:
    """Period m of the MASE scale for a window with this `min_train`.

    The seasonal scale needs MORE THAN m training levels (one pair |x_i - x_{i-m}|
    at the very least). The first outer fold of a window has exactly `min_train`
    of them, so a window whose min_train is <= 12 cannot be scaled seasonally at
    all: every one of its configurations would raise on fold 0. That is w2024
    (min_train = 12); training_plan.md 6.2 had called 12 "the floor at which the
    denominator is computable" -- it is the floor minus one (O-12, section 11).

    Such windows fall back to m = 1, the in-sample MAE of the random walk: defined
    from 2 levels, stable at 12, and the SAME for every fold of the window. A
    per-fold fallback was rejected because it would mix two scales inside one
    run's mean (a 12-month difference on a fast-growing series is ~12x a 1-month
    one). The choice depends only on min_train, so it is deterministic, and
    w2019 / w2021 (36 / 24) are unchanged.
    """
    return SEASONAL_PERIOD if min_train > SEASONAL_PERIOD else 1


def reconstruct(anchor: np.ndarray, z_hat: np.ndarray) -> np.ndarray:
    """n_hat_{t+h-1} = n_{t-1} * exp(z_hat).  training_plan.md 1.2."""
    return np.asarray(anchor, dtype=float) * np.exp(np.asarray(z_hat, dtype=float))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    return float(np.mean(np.abs((y - np.asarray(yhat, float)) / y)) * 100.0)


def seasonal_naive_scale(train_levels: np.ndarray, m: int = SEASONAL_PERIOD) -> float:
    """The MASE denominator: in-sample MAE of the seasonal naive on TRAINING rows.

    Training rows only, per fold. Computing it on the full series would leak the
    test period's volatility into the scale and make MASE incomparable between
    folds.
    """
    x = np.asarray(train_levels, float)
    if len(x) <= m:
        raise ValueError(f"Need more than {m} training levels to scale MASE, got {len(x)}")
    d = np.abs(x[m:] - x[:-m])
    scale = float(np.mean(d))
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("Degenerate MASE scale (zero or non-finite)")
    return scale


def mase(
    y: np.ndarray, yhat: np.ndarray, train_levels: np.ndarray, m: int = SEASONAL_PERIOD
) -> float:
    """MAE divided by the seasonal naive's in-sample MAE.

    `m` is 12 except on windows too short for it (see mase_period).

    0.7 = 30% better than "same month last year". 1.0 = no better. 1.4 = worse.

    NOTE the deliberate compromise (training_plan.md 6.3): the denominator is
    fixed at m=12 for BOTH horizons, so h=1 and h=3 MASE are directly
    comparable -- at the cost that MASE < 1 no longer literally means "beats
    naive at this horizon". skill_h below is the metric that does mean that.
    """
    return mae(y, yhat) / seasonal_naive_scale(train_levels, m)


def skill(y: np.ndarray, yhat: np.ndarray, y_seasonal: np.ndarray) -> float:
    """1 - MAE_model / MAE_seasonal_naive, same rows, same horizon.

    0.25 = cut the naive forecast's error by a quarter. 0 = tied.
    NEGATIVE = worse than doing nothing, which gets reported as a finding
    rather than quietly dropped (training_plan.md 6.4 rule 2).
    """
    denom = mae(y, y_seasonal)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return 1.0 - mae(y, yhat) / denom


def all_metrics(
    y_level: np.ndarray,
    yhat_level: np.ndarray,
    *,
    train_levels: np.ndarray,
    seasonal_level: np.ndarray,
    scale_m: int = SEASONAL_PERIOD,
) -> dict[str, float]:
    return {
        "mase": mase(y_level, yhat_level, train_levels, scale_m),
        "skill_h": skill(y_level, yhat_level, seasonal_level),
        "mae": mae(y_level, yhat_level),
        "rmse": rmse(y_level, yhat_level),
        "mape": mape(y_level, yhat_level),
    }
