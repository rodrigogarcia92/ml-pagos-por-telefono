"""One configuration, three stages. The procedure, written once.

    python -m src.model_training.train --target t2 --horizon 1 --window w2019 \
        --model xgboost --feature-set FS3_activity --stage cv

`run(cfg)` is the real entry point; the CLI is a thin wrapper, so sweep.py calls
the FUNCTION rather than spawning subprocesses -- which is also what makes the
Vertex demo at roadmap step 14 a config change rather than a refactor.

The three stages (docs/training_plan.md 7.3):

  A  tune     inner expanding CV inside the training portion. Creates NO MLflow
              runs; the whole search is logged as tuning_results.csv on the
              Stage B parent (8.0).
  B  cv       outer expanding CV with hyperparameters frozen. Parent + one
              child per fold.
  C  holdout  the reserved final months, evaluated ONCE, for the selected
              configuration only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from dataclasses import dataclass, field
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


def _prepare(X_tr, X_te):
    """Impute then scale, FITTED ON TRAINING ROWS ONLY.

    Fitting StandardScaler on the full series before splitting is the most
    common silent leak in this kind of project: the scaler carries the test
    period's mean and variance into training. Test T4 asserts the fitted mean
    differs between the first and last fold.
    """
    imp = SimpleImputer(strategy="median").fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))
    return sc.transform(imp.transform(X_tr)), sc.transform(imp.transform(X_te))


def _score_fold(model, frame, tr, te) -> tuple[dict, np.ndarray]:
    Xtr, Xte = frame.X.to_numpy()[tr], frame.X.to_numpy()[te]
    ytr = frame.y.to_numpy()[tr]
    ctx_tr, ctx_te = frame.ctx.iloc[tr], frame.ctx.iloc[te]

    if model.uses_context:
        model.fit(None, ytr, ctx_tr)
        z_hat = model.predict(None, ctx_te)
    else:
        Xtr_s, Xte_s = _prepare(Xtr, Xte)
        model.fit(Xtr_s, ytr, None)
        z_hat = model.predict(Xte_s, None)

    # Everything below is on LEVELS. Never score on z (training_plan.md 1.2).
    yhat = metrics.reconstruct(ctx_te["anchor"].to_numpy(), z_hat)
    m = metrics.all_metrics(
        ctx_te["y_level"].to_numpy(),
        yhat,
        train_levels=ctx_tr["y_level"].to_numpy(),
        seasonal_level=ctx_te["seas_level"].to_numpy(),
    )
    return m, yhat


def _tune(frame, cfg, folds) -> tuple[dict, pd.DataFrame]:
    """Stage A. Grid search on INNER folds inside the training portion only.

    Returns the winning parameters and the complete search as a table. The table
    is logged as one artifact; the ~35k individual fits are not logged at all,
    because MLflow run creation would cost one to two orders of magnitude more
    than the modelling itself (training_plan.md 8.0).
    """
    grid = cfg.tune or registry.default_grid(cfg.model_family)
    if not grid:
        return {}, pd.DataFrame()

    n_train = int(folds[0].train[-1]) + 1
    inner = splits.inner_folds(
        n_train, horizon=cfg.horizon, min_train=splits.MIN_TRAIN[cfg.window]
    )
    if not inner:
        return {}, pd.DataFrame()

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        scores = []
        for f in inner:
            model = registry.build(cfg.model_family, params, horizon=cfg.horizon)
            try:
                m, _ = _score_fold(model, frame, f.train, f.test)
                scores.append(m["mase"])
            except (ValueError, FloatingPointError):
                scores.append(np.nan)
        rows.append({**params, "mase_inner_mean": np.nanmean(scores),
                     "mase_inner_std": np.nanstd(scores), "n_inner_folds": len(inner)})

    table = pd.DataFrame(rows).sort_values("mase_inner_mean")
    best = {k: table.iloc[0][k] for k in keys}
    # numpy scalars do not survive a YAML/JSON round trip cleanly
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, table


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


def run(cfg: RunConfig) -> RunResult:
    snapshot_path = cfg.snapshot or str(latest_snapshot())
    panel, meta = dataset.load_snapshot(snapshot_path)

    fs = load_feature_sets()
    columns = [] if cfg.model_family in registry.NAIVE_FAMILIES else fs[cfg.feature_set]
    encoding = cfg.encoding or ("onehot" if cfg.model_family in
                                {"ridge", "elasticnet", "svr_rbf"} else "int")

    frame = dataset.build(
        panel, meta,
        target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
        columns=columns, encoding=encoding,
    )

    # The holdout is carved off the END and is invisible until Stage C.
    n_hold = HOLDOUT_MONTHS[cfg.window]
    n_cv = len(frame.X) - n_hold
    if n_cv <= splits.MIN_TRAIN[cfg.window]:
        raise ValueError(f"{cfg.window}: {n_cv} CV rows is not enough for min_train")

    folds = splits.make_folds(n_cv, window=cfg.window, horizon=cfg.horizon)
    if not folds:
        raise ValueError(f"No folds for {cfg.window} h={cfg.horizon}")

    best_params, tuning_table = _tune(frame, cfg, folds)

    ctx = tracking.RunContext(
        target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
        feature_set=cfg.feature_set if columns else "none",
        model_family=cfg.model_family,
        encoding=encoding if columns else "none",
        stage=cfg.stage, snapshot_path=snapshot_path,
    )

    if cfg.stage == "holdout":
        eval_folds = [splits.Fold(0, np.arange(0, n_cv), np.arange(n_cv, len(frame.X)))]
    else:
        eval_folds = folds

    per_fold, preds, actuals, dates = [], [], [], []
    with tracking.parent_run(ctx) as parent:
        mlflow.log_params({
            **{f"hp_{k}": v for k, v in best_params.items()},
            "min_train": splits.MIN_TRAIN[cfg.window],
            "purge": cfg.horizon - 1,
            "n_folds": len(eval_folds),
            "n_features": frame.X.shape[1],
            "n_candidates_evaluated": len(tuning_table),
            "holdout_months": n_hold,
            "seed": SEED,
        })

        # Populates MLflow's Datasets tab, which is empty otherwise. It records
        # WHICH rows this run saw -- name, digest, and the snapshot path as
        # source -- so two runs claiming the same data_version can be shown to
        # have used the same frame rather than merely asserting it.
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(
                    frame.X.assign(y=frame.y),
                    source=snapshot_path,
                    name=f"{cfg.target_id}_h{cfg.horizon}_{cfg.window}_{cfg.feature_set}",
                    targets="y",
                ),
                context="cv" if cfg.stage == "cv" else "holdout",
            )
        except Exception:  # noqa: BLE001 -- provenance nicety, never worth failing a run
            pass

        for f in eval_folds:
            model = registry.build(cfg.model_family, best_params, horizon=cfg.horizon)
            m, yhat = _score_fold(model, frame, f.train, f.test)
            per_fold.append(m)
            preds.extend(yhat)
            actuals.extend(frame.ctx["y_level"].to_numpy()[f.test])
            dates.extend(frame.X.index[f.test])
            with tracking.fold_run(f.index):
                mlflow.log_metrics(m)          # children: metrics only (8.5)

        agg = {}
        for key in per_fold[0]:
            vals = [d[key] for d in per_fold]
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            # The sd matters as much as the mean: a model that wins on average
            # by being wildly variable is not a better model.
            agg[f"{key}_std"] = float(np.nanstd(vals))
        mlflow.log_metrics(agg)

        # ---- artifacts, parent only (8.5) --------------------------------- #
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"obs_date": dates, "actual": actuals, "predicted": preds}) \
                .to_csv(d / "predictions.csv", index=False)
            if len(tuning_table):
                tuning_table.to_csv(d / "tuning_results.csv", index=False)
            (d / "features.json").write_text(json.dumps({
                "columns": list(frame.X.columns), "dropped": frame.dropped,
            }, indent=2))
            (d / "folds.json").write_text(json.dumps(
                [{"fold": f.index, "train": f.train.tolist(), "test": f.test.tolist()}
                 for f in eval_folds], indent=2))
            frame.X.assign(y=frame.y).to_parquet(d / "training_frame.parquet")
            fig = _forecast_plot(dates, actuals, preds, ctx.run_name)
            fig.savefig(d / "forecast_vs_actual.png", dpi=120)
            plt.close(fig)
            mlflow.log_artifacts(str(d))

        return RunResult(
            run_id=parent.info.run_id, run_name=ctx.run_name,
            metrics=agg, n_folds=len(eval_folds), best_params=best_params,
        )


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
