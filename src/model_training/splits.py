"""Expanding-window backtest folds, with the purge gap.

Never a random split. The rows are consecutive months and the target is
autocorrelated, so a shuffled split lets the model see the future -- that is a
guaranteed fake result, not a subtle bug (plan 6.2).

    origins:  0 1 2 ... 35 | 36 | 37 ...
    fold 0:   [--- train ---] test
    fold 1:   [--- train ----] test
    ...

THE PURGE. At h=3 a row's target spans [t-1, t+2]. Two consecutive origins
therefore overlap by two months. Without a gap, the last training row's target
window overlaps the test row's target window and leaks. purge = h - 1 rows are
dropped from the END of every training set -- which also costs h-1 folds,
because the first admissible test index moves right by the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per window, because a flat 36 leaves 5 folds on w2021 and none at all on
# w2024 at h=3 (plan 6.2). Logged as a param on every run, so the
# choice is visible in the MLflow table and runs are only compared within a window.
MIN_TRAIN = {"w2019": 36, "w2021": 24, "w2024": 12}


@dataclass(frozen=True)
class Fold:
    index: int
    train: np.ndarray
    test: np.ndarray


def expanding_window(n_rows: int, *, min_train: int, purge: int) -> list[Fold]:
    """One test origin per fold, training on everything before it minus the purge."""
    if min_train < 1:
        raise ValueError("min_train must be >= 1")
    folds = []
    for i in range(min_train + purge, n_rows):
        train = np.arange(0, i - purge)
        if len(train) < min_train:
            continue
        folds.append(Fold(index=len(folds), train=train, test=np.array([i])))
    return folds


def make_folds(n_rows: int, *, window: str, horizon: int) -> list[Fold]:
    return expanding_window(n_rows, min_train=MIN_TRAIN[window], purge=horizon - 1)


def inner_folds(n_train: int, *, horizon: int, min_train: int, max_folds: int = 10) -> list[Fold]:
    """Folds for hyperparameter selection, INSIDE the training portion only.

    Stage A never touches an outer test row. Capped at max_folds because the
    tuning grid is large and inner folds multiply it -- the cap trades a little
    selection precision for an order of magnitude of wall clock, and the choice
    is logged as a param.
    """
    all_folds = expanding_window(n_train, min_train=min_train, purge=horizon - 1)
    if len(all_folds) <= max_folds:
        return all_folds
    # Keep the LAST max_folds: the most recent regime is the one being forecast.
    kept = all_folds[-max_folds:]
    return [Fold(index=i, train=f.train, test=f.test) for i, f in enumerate(kept)]


def holdout_folds(n_rows: int, *, n_cv: int, horizon: int) -> list[Fold]:
    """Stage C: one expanding-window fold per holdout origin, purge included.

    Same construction as Stage B, so the B -> C gap (selection optimism,
    plan 7.3) compares like with like. Rows `n_cv` .. `n_rows - 1` are
    the holdout; the first test origin is `n_cv`, and every fold trains on
    everything up to `origin - purge`.

    v1.3 evaluated the holdout as a SINGLE fit on all CV rows against all
    holdout rows at once. That had no purge -- a leak at h=3, where the last
    training target overlaps the first test target -- and scored a model that was
    never refitted as the holdout unfolded, so it was not comparable with Stage B.
    Nothing had been run against it when it was replaced (plan 11).
    """
    purge = horizon - 1
    folds = []
    for i in range(n_cv, n_rows):
        folds.append(Fold(index=len(folds), train=np.arange(0, i - purge), test=np.array([i])))
    return folds
