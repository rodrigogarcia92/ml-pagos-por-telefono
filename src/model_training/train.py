"""One configuration, three stages. The procedure, written once.

    python -m src.model_training.train --target t2 --horizon 1 --window w2019 \
        --model xgboost --feature-set FS3_activity --stage cv

Two halves, so that sweeps can parallelise the expensive one safely:

  fit_config(cfg) -> FitResult     ALL the computation; touches no MLflow at all.
  log_config(fit) -> RunResult     ALL the logging; does no modelling.
  run(cfg)                         the two in sequence.

The split exists because the tracking server is a single uvicorn worker over
SQLite. Joblib workers call fit_config and RETURN results; the parent process
alone calls log_config. One writer, always (sweep.py). A worker that crashes
cannot leave a half-written run behind, because it never held a handle to one.

`run(cfg)` is the real entry point and the CLI a thin wrapper, so sweep.py calls
functions rather than spawning subprocesses -- which is also what makes the
Vertex demo at roadmap step 14 a config change rather than a refactor.

The three stages (docs/training_plan.md 7.3):

  A  tune     inner expanding CV INSIDE THE CV PERIOD (never the holdout).
              Creates NO MLflow runs; the whole search is logged as
              tuning_results.csv on the Stage B parent (8.0).
  B  cv       outer expanding CV with hyperparameters frozen. Parent + one
              child per fold. Mildly optimistic by construction -- Stage A saw the
              last inner folds' targets -- and flagged as such (7.3).
  C  holdout  the reserved final months, evaluated ONCE, for the selected
              configuration only. One expanding-window fold per holdout origin.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless sweep
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.model_training import dataset, metrics, registry, splits, tracking
from src.model_training.snapshot import latest_snapshot

CONFIG_DIR = Path("configs")
HOLDOUT_MONTHS = {"w2019": 12, "w2021": 12, "w2024": 6}
SEED = 26

# Fold counts below this are demonstrations, not evaluations
# (training_plan.md 6.4 rule 4). Defined in tracking.py, which owns the tag.
MIN_EVALUATION_FOLDS = tracking.MIN_EVALUATION_FOLDS


@dataclass
class RunConfig:
    target_id: str
    horizon: int
    window: str
    model_family: str
    feature_set: str
    stage: str = "cv"
    encoding: str | None = None
    snapshot: str | None = None
    tune: dict = field(default_factory=dict)
    # 1.6 only for the s0-s4 specs (sweep.py gives a spec with no key the legacy value);
    # everything new is 1.7. Part of the resume key.
    protocol_version: str = tracking.PROTOCOL_VERSION
    # Stage C only: hyperparameters frozen BEFORE the holdout (configs/production/*.yaml, plan 11),
    # {member: params} for an ensemble. When set, the holdout run uses them as they are and runs
    # no Stage A at all; absent, Stage A re-runs on the CV rows (never the holdout) as before.
    frozen_params: dict | None = None
    frozen_source: str = ""             # label logged as param hp_source, e.g. "frozen:ab12cd34ef56"


# What exists only from protocol 1.7 on (training_plan.md 3, 5.4, 7.4). A 1.6 run asking
# for one of these is a mislabelled run -- refused rather than tagged "1.6".
PROTOCOL_17_ONLY = {
    "target_id": {"t10", "t11"},
    "model_family": {"ens3"},
    "feature_set": {"FS3_gt"},
}


def check_protocol(cfg: "RunConfig") -> None:
    """Refuse a configuration that uses a 1.7 feature under a 1.6 tag."""
    if cfg.protocol_version == tracking.PROTOCOL_VERSION:
        return
    used = {k: getattr(cfg, k) for k, only in PROTOCOL_17_ONLY.items() if getattr(cfg, k) in only}
    if used:
        raise ValueError(
            f"{used} exist only from protocol {tracking.PROTOCOL_VERSION}, but this run is "
            f"tagged {cfg.protocol_version}. Set protocol_version: '{tracking.PROTOCOL_VERSION}' "
            "in the sweep spec.")


@dataclass
class FitResult:
    """Everything computed for one configuration. Picklable, MLflow-free."""

    cfg: RunConfig
    snapshot_path: str
    frame: dataset.Frame
    columns: list[str]
    encoding_tag: str
    n_cv: int
    n_hold: int
    eval_folds: list
    best_params: dict
    tuning_table: pd.DataFrame
    per_fold: list[dict]
    preds: list
    actuals: list
    dates: list


@dataclass
class RunResult:
    run_id: str
    run_name: str
    metrics: dict
    n_folds: int
    best_params: dict


def _flatten(v) -> list[str]:
    """feature_sets.yaml composes sets with YAML anchors, which nests lists."""
    out: list[str] = []
    for item in v:
        out.extend(_flatten(item) if isinstance(item, list) else [item])
    return out


def load_feature_sets() -> dict[str, list[str]]:
    raw = yaml.safe_load((CONFIG_DIR / "feature_sets.yaml").read_text(encoding="utf-8"))
    return {k: _flatten(v) for k, v in raw.items()}


def context_for(cfg: RunConfig, snapshot_path: str) -> tracking.RunContext:
    """The ONE place a RunConfig becomes a RunContext.

    sweep.py uses this to decide whether a run already exists; train uses it to
    tag the run it creates. Two hand-built copies disagreed on `encoding` once,
    and resumability silently never matched a feature model.
    """
    naive = cfg.model_family in registry.NAIVE_FAMILIES
    return tracking.RunContext(
        target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
        feature_set="none" if naive else cfg.feature_set,
        model_family=cfg.model_family,
        encoding=cfg.encoding or registry.default_encoding(cfg.model_family),
        stage=cfg.stage, snapshot_path=snapshot_path,
        protocol_version=cfg.protocol_version,
    )


def _prepare(X_tr, *others):
    """Impute then scale, FITTED ON TRAINING ROWS ONLY; apply to everything else.

    Fitting StandardScaler on the full series before splitting is the most
    common silent leak in this kind of project: the scaler carries the test
    period's mean and variance into training. Test T4 asserts the fitted mean
    differs between the first and last fold.
    """
    imp = SimpleImputer(strategy="median").fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))
    # An empty array (the h=1 purge gap) is passed through: sklearn's transform
    # raises on zero samples, and an empty gap needs no scaling.
    return tuple(
        sc.transform(imp.transform(a)) if len(a) else a for a in (X_tr, *others)
    )


def _score_fold(model, frame, tr, te) -> tuple[dict, np.ndarray]:
    X_all = frame.X.to_numpy()
    Xtr, Xte = X_all[tr], X_all[te]
    ytr = frame.y.to_numpy()[tr]
    ctx_tr, ctx_te = frame.ctx.iloc[tr], frame.ctx.iloc[te]

    # A test origin with no year-ago level has no seasonal reference, so neither
    # skill_h nor the seasonal naives can be scored there. NaN would be skipped by
    # nanmean in log_config -- a silent drop. w2024 frames DO contain such rows
    # (the wallets start 2024-01, so seas_level is NaN for the first <= 11 rows),
    # but min_train = 12 keeps them all in TRAINING, which is why nothing reached
    # this guard in the wallet window (O-12). If min_train ever drops, fail here.
    if ctx_te["seas_level"].isna().any():
        raise ValueError(
            f"{frame.window}: seasonal reference missing at test origin "
            f"{[f'{d:%Y-%m}' for d in ctx_te.index[ctx_te['seas_level'].isna()]]}. "
            "Refusing to score it as NaN (training_plan.md 6.4 rule 2, O-12)."
        )
    scale_m = metrics.mase_period(splits.MIN_TRAIN[frame.window])

    if getattr(model, "is_ensemble", False):
        # ens3: every member gets ITS OWN encoding of the frame and its own imputer and
        # scaler, fitted on the training rows only; the forecast is the mean of the
        # members' z-forecasts, reconstructed to levels once below (plan 7.4, 1.2).
        Xtr_by, Xte_by = {}, {}
        for family, _ in model.members:
            Xa = frame.for_encoding(registry.default_encoding(family)).X.to_numpy()
            Xtr_by[family], Xte_by[family], _ = _prepare(
                Xa[tr], Xa[te], Xa[tr[-1] + 1: te[0]])
        model.fit(Xtr_by, ytr, None)
        z_hat = model.predict(Xte_by, None)
    elif model.uses_context:
        model.fit(None, ytr, ctx_tr)
        z_hat = model.predict(None, ctx_te)
    else:
        # Rows strictly between the end of training and the test origin: the
        # purge. Their FEATURES are known when the test origin is forecast; only
        # their targets are withheld. Models that forecast through the gap
        # (SARIMAX) get them; everyone else ignores them.
        gap = X_all[tr[-1] + 1: te[0]]
        Xtr_s, Xte_s, gap_s = _prepare(Xtr, Xte, gap)
        model.fit(Xtr_s, ytr, None)
        z_hat = model.predict(Xte_s, {"X_gap": gap_s} if getattr(model, "wants_gap", False) else None)

    # Everything below is on LEVELS. Never score on z (training_plan.md 1.2).
    yhat = metrics.reconstruct(ctx_te["anchor"].to_numpy(), z_hat)
    m = metrics.all_metrics(
        ctx_te["y_level"].to_numpy(),
        yhat,
        train_levels=ctx_tr["y_level"].to_numpy(),
        seasonal_level=ctx_te["seas_level"].to_numpy(),
        scale_m=scale_m,
    )
    if getattr(model, "is_ensemble", False):
        # Each member's own fold MASE, so "is the ensemble better than its best member"
        # can be answered from the run itself (plan 7.4).
        for family, z in model.member_z_.items():
            m[f"mase_{family}"] = metrics.mase(
                ctx_te["y_level"].to_numpy(),
                metrics.reconstruct(ctx_te["anchor"].to_numpy(), z),
                ctx_tr["y_level"].to_numpy(), scale_m,
            )
    return m, yhat


def hyperparameter_params(cfg: RunConfig, best_params: dict) -> dict:
    """MLflow params for the tuned hyperparameters: `hp_{param}`, or for an ensemble
    `member_{family}__{param}` (plan 7.4) -- the parents' hyperparameters, logged."""
    if cfg.model_family in registry.ENSEMBLES:
        return {f"member_{family}__{k}": v
                for family, params in best_params.items() for k, v in params.items()}
    return {f"hp_{k}": v for k, v in best_params.items()}


def _pyval(v):
    """numpy scalar -> plain Python, so params survive YAML/JSON and model ctors."""
    return v.item() if hasattr(v, "item") else v


def _tune(frame, cfg, n_cv) -> tuple[dict, pd.DataFrame]:
    """Stage A. Grid search on INNER folds drawn from the CV period only.

    The inner folds are the last `max_inner_folds` origins of the CV period, each
    trained on everything before it. The holdout is never indexed: every index
    below `n_cv`, asserted rather than assumed.

    Returns the winning parameters and the complete search as a table. The table
    is logged as one artifact; the ~35k individual fits are not logged at all,
    because MLflow run creation would cost one to two orders of magnitude more
    than the modelling itself (training_plan.md 8.0).

    HISTORY. The first version took the training portion of the FIRST outer fold,
    which has exactly `min_train` rows, so the inner-fold generator returned
    nothing and tuning was silently skipped: every s0_smoke run logged
    n_candidates_evaluated=0 and used library defaults. The winner was also read
    from a DataFrame row, which would have turned ints into floats and None into
    NaN. Both fixed; test_tuning_* pins them.
    """
    if cfg.model_family in registry.ENSEMBLES:
        return _tune_ensemble(frame, cfg, n_cv)
    grid = registry.grid_for(cfg.model_family, cfg.feature_set, cfg.tune, cfg.window)
    if not grid:
        return {}, pd.DataFrame()

    inner = splits.inner_folds(
        n_cv, horizon=cfg.horizon, min_train=splits.MIN_TRAIN[cfg.window]
    )
    if not inner:
        raise ValueError(
            f"{cfg.window} h={cfg.horizon}: {n_cv} CV rows leave no inner folds for "
            f"min_train={splits.MIN_TRAIN[cfg.window]}. Refusing to tune nothing and "
            "report library defaults as tuned."
        )
    assert max(int(f.test.max()) for f in inner) < n_cv, "Stage A reached the holdout"

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {k: _pyval(v) for k, v in zip(keys, combo, strict=True)}
        scores = []
        for f in inner:
            model = registry.build(cfg.model_family, params, horizon=cfg.horizon)
            try:
                m, _ = _score_fold(model, frame, f.train, f.test)
                scores.append(m["mase"])
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                # An infeasible combination (e.g. a seasonal order that does not
                # fit the training length). Scored NaN, never chosen.
                scores.append(np.nan)
        ok = [x for x in scores if np.isfinite(x)]
        rows.append({
            **params,
            "mase_inner_mean": float(np.mean(ok)) if ok else float("nan"),
            "mase_inner_std": float(np.std(ok)) if ok else float("nan"),
            "n_ok_inner_folds": len(ok), "n_inner_folds": len(inner),
        })

    finite = [r for r in rows if np.isfinite(r["mase_inner_mean"])]
    if not finite:
        raise RuntimeError(f"Every candidate failed in Stage A for {cfg}")
    # Python values straight from `rows`: a DataFrame row would coerce int -> float
    # and None -> NaN, which would break max_depth and drift_window.
    winner = min(finite, key=lambda r: r["mase_inner_mean"])
    best = {k: winner[k] for k in keys}
    table = pd.DataFrame(rows).sort_values("mase_inner_mean", na_position="last")
    return best, table


def _tune_ensemble(frame, cfg, n_cv) -> tuple[dict, pd.DataFrame]:
    """Stage A for ens3: every member tuned INDEPENDENTLY (plan 7.4).

    Each member runs the ordinary single-model Stage A -- its own full grid, its own
    encoding of the frame, the same inner folds (they depend only on n_cv, h and the
    window) -- and sees nothing of the other members or of the ensemble's score.
    Returns {member: its best hyperparameters} and one table with a `member` column.
    """
    best, tables = {}, []
    for family in registry.ENSEMBLES[cfg.model_family]:
        sub = replace(cfg, model_family=family, tune=(cfg.tune or {}).get(family, {}))
        member_frame = frame.for_encoding(registry.default_encoding(family))
        best[family], table = _tune(member_frame, sub, n_cv)
        tables.append(table.assign(member=family))
    out = pd.concat(tables, ignore_index=True)
    return best, out[["member"] + [c for c in out.columns if c != "member"]]


def _forecast_plot(dates, actual, predicted, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, actual, label="actual", linewidth=1.6)
    ax.plot(dates, predicted, label="forecast", linewidth=1.6, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("operations")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Half one: compute
# --------------------------------------------------------------------------- #
def fit_config(cfg: RunConfig) -> FitResult:
    """Stages A and B (or A and C) for one configuration. No MLflow."""
    check_protocol(cfg)
    snapshot_path = cfg.snapshot or str(latest_snapshot())
    panel, meta = dataset.load_snapshot(snapshot_path)

    naive = cfg.model_family in registry.NAIVE_FAMILIES
    fs = load_feature_sets()
    columns = [] if naive else registry.model_columns(cfg.model_family, fs[cfg.feature_set])
    ctx = context_for(cfg, snapshot_path)
    ensemble = cfg.model_family in registry.ENSEMBLES
    # "none" and "mixed" are tags, not build modes.
    encoding = "int" if (not columns or ensemble) else ctx.encoding

    def build_frame(enc: str) -> dataset.Frame:
        return dataset.build(
            panel, meta,
            target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
            columns=columns, encoding=enc,
        )

    frame = build_frame(encoding)
    if ensemble:
        # The members use different encodings of the same rows; build the others and
        # prove they ARE the same rows -- a frame that differed in more than the
        # calendar columns would make the members forecast different targets.
        wanted = set(registry.member_encodings(cfg.model_family).values()) - {encoding}
        others = {e: build_frame(e) for e in wanted}
        for e, f in others.items():
            if not (f.X.index.equals(frame.X.index) and f.y.equals(frame.y)
                    and f.ctx.equals(frame.ctx)):
                raise RuntimeError(f"{cfg.model_family}: the {e!r} frame differs from the "
                                   f"{encoding!r} frame in rows, target or context")
        frame.alt_encodings = others

    # The holdout is carved off the END and is invisible until Stage C. Rows are
    # one per target month and the window bounds the target month (protocol 1.6,
    # O-10), so this is the FINAL n_hold TARGET MONTHS -- the same calendar months
    # at h=1 and h=3, whatever kappa is.
    n_hold = HOLDOUT_MONTHS[cfg.window]
    n_rows = len(frame.X)
    n_cv = n_rows - n_hold
    if n_cv <= splits.MIN_TRAIN[cfg.window]:
        raise ValueError(f"{cfg.window}: {n_cv} CV rows is not enough for min_train")

    if cfg.frozen_params is not None:
        if cfg.stage != "holdout":
            raise ValueError("frozen_params are a Stage C input; a CV run tunes its own (plan 7.3)")
        best_params, tuning_table = cfg.frozen_params, pd.DataFrame()
    else:
        best_params, tuning_table = _tune(frame, cfg, n_cv)

    if cfg.stage == "holdout":
        eval_folds = splits.holdout_folds(n_rows, n_cv=n_cv, horizon=cfg.horizon)
    else:
        eval_folds = splits.make_folds(n_cv, window=cfg.window, horizon=cfg.horizon)
    if not eval_folds:
        raise ValueError(f"No folds for {cfg.window} h={cfg.horizon} stage={cfg.stage}")

    per_fold, preds, actuals, dates = [], [], [], []
    for f in eval_folds:
        model = registry.build(cfg.model_family, best_params, horizon=cfg.horizon)
        m, yhat = _score_fold(model, frame, f.train, f.test)
        per_fold.append(m)
        preds.extend(yhat)
        actuals.extend(frame.ctx["y_level"].to_numpy()[f.test])
        dates.extend(frame.X.index[f.test])

    return FitResult(
        cfg=cfg, snapshot_path=snapshot_path, frame=frame, columns=columns,
        encoding_tag=ctx.encoding if columns else "none",
        n_cv=n_cv, n_hold=n_hold, eval_folds=eval_folds,
        best_params=best_params, tuning_table=tuning_table,
        per_fold=per_fold, preds=preds, actuals=actuals, dates=dates,
    )


# --------------------------------------------------------------------------- #
# Half two: record
# --------------------------------------------------------------------------- #
def log_config(fit: FitResult) -> RunResult:
    """Write one FitResult to MLflow. Called from ONE process only."""
    cfg, frame = fit.cfg, fit.frame
    ctx = context_for(cfg, fit.snapshot_path)
    n_folds = len(fit.eval_folds)

    note = None
    if n_folds < MIN_EVALUATION_FOLDS:
        note = (f"DEMONSTRATION, NOT AN EVALUATION: {n_folds} outer folds "
                f"(< {MIN_EVALUATION_FOLDS}). training_plan.md 6.4 rule 4.")

    with tracking.parent_run(ctx, description=note, n_folds=n_folds) as parent:
        mlflow.log_params({
            **hyperparameter_params(cfg, fit.best_params),
            "min_train": splits.MIN_TRAIN[cfg.window],
            "purge": cfg.horizon - 1,
            "n_folds": n_folds,
            "n_features": frame.X.shape[1],
            "n_candidates_evaluated": len(fit.tuning_table),
            # O-12. Which MASE scale this window uses (12 normally; 1 where
            # min_train <= 12 makes the seasonal one undefined on fold 0), how
            # many frame rows have no year-ago level (they sit in training only --
            # _score_fold refuses them at a test origin), and how many folds came
            # out NaN on any metric (log_config's nanmean would skip them silently).
            "mase_scale_period": metrics.mase_period(splits.MIN_TRAIN[cfg.window]),
            "n_rows_seasonal_ref_missing": int(frame.ctx["seas_level"].isna().sum()),
            "n_folds_nan_metric": sum(
                any(np.isnan(v) for v in d.values()) for d in fit.per_fold),
            "holdout_months": fit.n_hold,   # the final N target months (h-independent)
            **({"hp_source": cfg.frozen_source or "frozen"} if cfg.frozen_params is not None else {}),
            **({"member_encodings": ",".join(
                f"{m}={e}" for m, e in registry.member_encodings(cfg.model_family).items())}
               if cfg.model_family in registry.ENSEMBLES else {}),
            "window_target_start": f"{frame.target_start:%Y-%m}",
            "window_target_end": f"{frame.target_end:%Y-%m}",
            "seed": SEED,
        })
        if frame.seas_transfer:
            # Plan 5.2: where the factors came from and the cutoff, on every run
            # that uses them. The twelve values go into features.json below.
            st = frame.seas_transfer
            mlflow.log_params({
                "seas_transfer_source": st["source"],
                "seas_transfer_first_target": st["first_target"],
                "seas_transfer_cutoff": st["cutoff"],          # exclusive
                "seas_transfer_excludes": st["excludes"],
            })

        # Populates MLflow's Datasets tab, which is empty otherwise. It records
        # WHICH rows this run saw -- name, digest, and the snapshot path as
        # source -- so two runs claiming the same data_version can be shown to
        # have used the same frame rather than merely asserting it.
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(
                    frame.X.assign(y=frame.y),
                    source=fit.snapshot_path,
                    name=f"{cfg.target_id}_h{cfg.horizon}_{cfg.window}_{cfg.feature_set}",
                    targets="y",
                ),
                context="cv" if cfg.stage == "cv" else "holdout",
            )
        except Exception:  # noqa: BLE001 -- provenance nicety, never worth failing a run
            pass

        for f, m in zip(fit.eval_folds, fit.per_fold, strict=True):
            with tracking.fold_run(f.index):
                mlflow.log_metrics(m)          # children: metrics only (8.5)

        agg = {}
        for key in fit.per_fold[0]:
            vals = [d[key] for d in fit.per_fold]
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            # The sd matters as much as the mean: a model that wins on average
            # by being wildly variable is not a better model.
            agg[f"{key}_std"] = float(np.nanstd(vals))
        # Each fold has ONE test point, so a fold's `rmse` equals its `mae` and `rmse_mean` above
        # is really a MAE. The pooled root-mean-square error over all test points is
        # sqrt(mean(fold_mae^2)) (plan 11, 2026-10-06). Additive; report.py derives it for old runs.
        agg["rmse_pooled"] = metrics.rmse_pooled([d["mae"] for d in fit.per_fold])
        mlflow.log_metrics(agg)

        # ---- artifacts, parent only (8.5) --------------------------------- #
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"obs_date": fit.dates, "actual": fit.actuals,
                          "predicted": fit.preds}).to_csv(d / "predictions.csv", index=False)
            if len(fit.tuning_table):
                fit.tuning_table.to_csv(d / "tuning_results.csv", index=False)
            (d / "features.json").write_text(json.dumps({
                "columns": list(frame.X.columns), "dropped": frame.dropped,
                **({"seas_transfer": frame.seas_transfer} if frame.seas_transfer else {}),
            }, indent=2))
            (d / "folds.json").write_text(json.dumps(
                [{"fold": f.index, "train": f.train.tolist(), "test": f.test.tolist()}
                 for f in fit.eval_folds], indent=2))
            frame.X.assign(y=frame.y).to_parquet(d / "training_frame.parquet")
            fig = _forecast_plot(fit.dates, fit.actuals, fit.preds, ctx.run_name)
            fig.savefig(d / "forecast_vs_actual.png", dpi=120)
            plt.close(fig)
            mlflow.log_artifacts(str(d))

        return RunResult(
            run_id=parent.info.run_id, run_name=ctx.run_name,
            metrics=agg, n_folds=n_folds, best_params=fit.best_params,
        )


def run(cfg: RunConfig) -> RunResult:
    return log_config(fit_config(cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--window", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--feature-set", default="none")
    ap.add_argument("--stage", default="cv", choices=["cv", "holdout"])
    ap.add_argument("--snapshot", default=None)
    a = ap.parse_args()

    res = run(RunConfig(
        target_id=a.target, horizon=a.horizon, window=a.window,
        model_family=a.model, feature_set=a.feature_set,
        stage=a.stage, snapshot=a.snapshot,
    ))
    print(f"{res.run_name}  folds={res.n_folds}  "
          f"MASE={res.metrics['mase_mean']:.3f} (sd {res.metrics['mase_std']:.3f})  "
          f"skill={res.metrics['skill_h_mean']:+.3f}")


if __name__ == "__main__":
    main()
