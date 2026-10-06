"""The ens3 ensemble outside the sweep: fold-by-fold refits, error quantiles, the production fit.

Shared by scripts/make_readme_figures.py (figures), src/forecasting/freeze.py (the error band
stored in the production config) and src/forecasting/predict.py (the live forecast), so that the
three can never disagree about what "the ensemble" is.

Nothing here creates an MLflow run, tunes anything or selects anything. Every function takes the
hyperparameters as an argument; they come from a logged Stage-A result (frozen config), never
from a search done here.

The mechanics are exactly train._score_fold's ensemble branch: each member gets ITS OWN encoding
of the frame and its own imputer + scaler, fitted on the training rows only; the forecast is the
mean of the members' z-forecasts, reconstructed to levels once (plan 1.2, 7.4).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model_training import dataset, registry, splits, train

ENSEMBLE = "ens3"
MEMBERS: tuple[str, ...] = registry.ENSEMBLES[ENSEMBLE]
QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)


# --------------------------------------------------------------------------- #
# Hyperparameters as MLflow stores them (strings) -> Python values
# --------------------------------------------------------------------------- #
def cast_param(v: str):
    """'None' -> None, 'scale' stays a string, '600' -> 600, '0.05' -> 0.05."""
    if v == "None":
        return None
    if v in ("scale", "auto"):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def parse_member_params(rows: Iterable[tuple[str, str]]) -> dict[str, dict]:
    """`member_{family}__{param}` rows -> {family: {param: value}} (plan 7.4 logging scheme)."""
    out: dict[str, dict] = {m: {} for m in MEMBERS}
    for k, v in rows:
        if k.startswith("member_") and "__" in k:
            fam, p = k.removeprefix("member_").split("__", 1)
            if fam in out:
                out[fam][p] = cast_param(v)
    missing = [m for m, p in out.items() if not p]
    if missing:
        raise ValueError(f"no logged hyperparameters for member(s) {missing}")
    return out


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def kappa_of(meta: pd.DataFrame, target_id: str) -> int:
    """Publication lag of the target's series: target month = origin - kappa + h."""
    return int(meta.loc[dataset.TARGETS[target_id]["cols"][0], "kappa"])


def member_frames(panel, meta, *, target_id: str, horizon: int, window: str,
                  columns: list[str]) -> dict[str, dataset.Frame]:
    """One frame per member, each in the member's own encoding."""
    return {
        fam: dataset.build(panel, meta, target_id=target_id, horizon=horizon, window=window,
                           columns=registry.model_columns(fam, columns),
                           encoding=registry.default_encoding(fam))
        for fam in MEMBERS
    }


def n_cv_rows(frames: dict[str, dataset.Frame], window: str) -> int:
    """CV rows = all rows minus the final HOLDOUT_MONTHS target months."""
    return len(next(iter(frames.values())).X) - train.HOLDOUT_MONTHS[window]


# --------------------------------------------------------------------------- #
# Fold-by-fold refit
# --------------------------------------------------------------------------- #
@dataclass
class FoldFit:
    fold: splits.Fold
    origin: pd.Timestamp
    target: pd.Timestamp
    anchor: float
    actual: float
    z: dict[str, float]                 # each member's z-forecast
    z_ens: float
    trend: float                        # naive_drift level forecast, same fold
    mats: dict[str, tuple[np.ndarray, np.ndarray]]   # member -> (scaled train, scaled test)
    models: dict[str, object]           # member -> fitted adapter


def iter_fold_fits(frames: dict[str, dataset.Frame], params: dict, folds: list[splits.Fold], *,
                   horizon: int, kappa: int) -> Iterator[FoldFit]:
    """Refit the three members on each fold's training rows and forecast its test origin."""
    ref = frames[MEMBERS[0]]
    shift = pd.DateOffset(months=horizon - kappa)
    anchor_all = ref.ctx["anchor"].to_numpy()
    level_all = ref.ctx["y_level"].to_numpy()
    for f in folds:
        tr, te = f.train, int(f.test[0])
        Xtr_by, Xte_by = {}, {}
        for fam in MEMBERS:
            Xa = frames[fam].X.to_numpy()
            Xtr_by[fam], Xte_by[fam] = train._prepare(Xa[tr], Xa[[te]])
        model = registry.build(ENSEMBLE, params, horizon=horizon)
        model.fit(Xtr_by, frames[MEMBERS[0]].y.to_numpy()[tr], None)
        z_ens = float(model.predict(Xte_by, None)[0])
        drift = np.nanmean(np.log(level_all[tr]) - np.log(anchor_all[tr]))
        origin = ref.X.index[te]
        yield FoldFit(
            fold=f, origin=origin, target=origin + shift,
            anchor=float(anchor_all[te]), actual=float(level_all[te]),
            z={fam: float(v[0]) for fam, v in model.member_z_.items()}, z_ens=z_ens,
            trend=float(anchor_all[te] * np.exp(drift)),
            mats={fam: (Xtr_by[fam], Xte_by[fam]) for fam in MEMBERS},
            models=dict(model.members),
        )


def cv_folds(frames: dict[str, dataset.Frame], *, window: str, horizon: int) -> list[splits.Fold]:
    """The sweep's expanding folds on the CV rows only -- the holdout is never indexed."""
    n_cv = n_cv_rows(frames, window)
    folds = splits.make_folds(n_cv, window=window, horizon=horizon)
    assert max(int(f.test[0]) for f in folds) < n_cv, "a CV fold reached the holdout"
    return folds


def cv_backtest(frames: dict[str, dataset.Frame], params: dict, *, window: str, horizon: int,
                kappa: int) -> pd.DataFrame:
    """Forecasts (levels) of the ensemble, its members and naive_drift on every CV fold."""
    rows = []
    for ff in iter_fold_fits(frames, params, cv_folds(frames, window=window, horizon=horizon),
                             horizon=horizon, kappa=kappa):
        rows.append({
            "target": ff.target, "actual": ff.actual,
            "ensemble": ff.anchor * np.exp(ff.z_ens), "trend": ff.trend,
            **{fam: ff.anchor * np.exp(v) for fam, v in ff.z.items()},
        })
    return pd.DataFrame(rows).set_index("target")


def first_holdout_target(frames: dict[str, dataset.Frame], *, window: str, horizon: int,
                         kappa: int) -> pd.Timestamp:
    """The first target month of the holdout: nothing in the error band may reach it."""
    ref = frames[MEMBERS[0]]
    return ref.X.index[n_cv_rows(frames, window)] + pd.DateOffset(months=horizon - kappa)


# --------------------------------------------------------------------------- #
# Error band
# --------------------------------------------------------------------------- #
def log_errors(fc: pd.DataFrame) -> np.ndarray:
    """log(actual / forecast): positive when the model under-forecast."""
    return np.log(fc["actual"].to_numpy() / fc["ensemble"].to_numpy())


def error_band(fc: pd.DataFrame) -> dict:
    """Empirical quantiles of the log error, the fold count and the CV MAPE (percent)."""
    e = log_errors(fc)
    q = np.quantile(e, QUANTILES)
    return {
        **{f"p{int(round(p * 100)):02d}": float(v) for p, v in zip(QUANTILES, q, strict=True)},
        "n_folds": int(len(e)),
        "cv_mape_pct": float(100 * np.mean(np.abs(fc["ensemble"] - fc["actual"]) / fc["actual"])),
    }


def apply_band(point: float, band: dict) -> dict:
    """80% (p10..p90) and 90% (p05..p95) intervals around a point forecast, in levels.

    actual = forecast * exp(e), so a quantile q of e maps to forecast * exp(q).
    """
    return {
        "80": [point * float(np.exp(band["p10"])), point * float(np.exp(band["p90"]))],
        "90": [point * float(np.exp(band["p05"])), point * float(np.exp(band["p95"]))],
    }


# --------------------------------------------------------------------------- #
# Production fit: all labelled rows -> the unlabelled prediction row
# --------------------------------------------------------------------------- #
def fit_predict(frames: dict[str, dataset.Frame], pred_frames: dict[str, dataset.Frame],
                params: dict, *, horizon: int) -> tuple[float, dict[str, float]]:
    """Fit the three members on every labelled row, forecast the (single) prediction row.

    Returns (z_ens, {member: z}). This is the PRODUCTION fit, not an evaluation: it scores
    nothing and is never compared with an actual.
    """
    ref = frames[MEMBERS[0]]
    if len(next(iter(pred_frames.values())).X) != 1:
        raise ValueError("fit_predict expects exactly one prediction row")
    Xtr_by, Xp_by = {}, {}
    for fam in MEMBERS:
        Xtr_by[fam], Xp_by[fam] = train._prepare(
            frames[fam].X.to_numpy(), pred_frames[fam].X[frames[fam].X.columns].to_numpy())
    model = registry.build(ENSEMBLE, params, horizon=horizon)
    model.fit(Xtr_by, ref.y.to_numpy(), None)
    z_ens = float(model.predict(Xp_by, None)[0])
    return z_ens, {fam: float(v[0]) for fam, v in model.member_z_.items()}
