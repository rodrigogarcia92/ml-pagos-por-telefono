"""Rank the parent runs in an experiment. The 'read the results' step.

    python -m src.model_training.report --target t2 --horizon 1
    python -m src.model_training.report --target t2 --horizon 1 --window w2019 --csv out.csv

Invoked as a MODULE, like every other entry point in this package, because
`python -m` puts the project root on sys.path while `python scripts/foo.py`
puts scripts/ there instead -- and this file imports `src.model_training`.

Why a script and not the UI: the MLflow Compare view loads every run it is
given, and the write-up needs tables that can be regenerated rather than
screenshotted. This is also the honest way to read the protocol, because it
prints the two columns that matter TOGETHER (docs/training_plan.md 6.3):

    mase       ranks configurations, comparable across horizons
    skill_h    1 - MAE_model / MAE_seasonal_naive, at THIS horizon

and adds one the plan does not define but this series demands:

    skill_rw   1 - MAE_model / MAE_random_walk, derived from the naive_last
               run in the same experiment/window

WHY skill_rw. n_transf_intra_agg grows about 40x across the sample, so "same
month last year" is a weak reference and "last month" is a strong one. Measured
against the seasonal naive, simply predicting no change already scores about
+0.89. A model reporting skill_h = +0.92 has therefore added almost nothing
beyond the anchor, and only a comparison against the random walk shows that.
Reported alongside, never instead of -- skill_h is what the protocol
pre-registered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from src.model_training import tracking  # noqa: F401 -- loads .env, sets tracking URI

COLS = ["model_family", "feature_set", "window", "n_folds",
        "mase_mean", "mase_std", "skill_h_mean", "mae_mean", "mape_mean"]


def fetch(target: str, horizon: int, stage: str = "cv") -> pd.DataFrame:
    exp = f"{tracking.EXPERIMENT_PREFIX}{target}_h{horizon}"
    runs = mlflow.search_runs(
        experiment_names=[exp],
        filter_string=f"tags.stage = '{stage}' and attributes.status = 'FINISHED'",
    )
    if runs.empty:
        raise SystemExit(f"No finished '{stage}' runs in {exp}.")

    # Parents only. Children are folds and carry no aggregate metrics.
    runs = runs[runs["metrics.mase_mean"].notna()].copy()

    out = pd.DataFrame({
        "model_family": runs["tags.model_family"],
        "feature_set": runs["tags.feature_set"],
        "window": runs["tags.window"],
        "n_folds": runs["params.n_folds"].astype(int),
        "mase_mean": runs["metrics.mase_mean"],
        "mase_std": runs["metrics.mase_std"],
        "skill_h_mean": runs["metrics.skill_h_mean"],
        "mae_mean": runs["metrics.mae_mean"],
        "mape_mean": runs["metrics.mape_mean"],
    })
    return out.sort_values(["window", "mase_mean"]).reset_index(drop=True)


def add_skill_rw(df: pd.DataFrame) -> pd.DataFrame:
    """Skill against the random walk, per window, using naive_last's MAE."""
    df = df.copy()
    df["skill_rw"] = pd.NA
    for window, block in df.groupby("window"):
        rw = block.loc[block["model_family"] == "naive_last", "mae_mean"]
        if rw.empty:
            continue  # naive_last not run for this window; leave blank, don't guess
        base = float(rw.iloc[0])
        df.loc[block.index, "skill_rw"] = 1.0 - block["mae_mean"] / base
    return df


def detail(target: str, horizon: int, top: pd.DataFrame, stage: str) -> None:
    """Everything needed to reproduce one run, printed per run.

    The same facts are all in the MLflow UI -- this exists because a terminal
    table can be pasted into a write-up and a screenshot cannot, and because it
    puts tags, params and the feature list side by side instead of on three
    different tabs.
    """
    exp = f"{tracking.EXPERIMENT_PREFIX}{target}_h{horizon}"
    client = mlflow.MlflowClient()
    exp_id = mlflow.get_experiment_by_name(exp).experiment_id

    for _, row in top.iterrows():
        hits = client.search_runs(
            [exp_id],
            filter_string=(
                f"tags.model_family = '{row.model_family}' and "
                f"tags.feature_set = '{row.feature_set}' and "
                f"tags.window = '{row.window}' and tags.stage = '{stage}'"
            ),
            max_results=1,
        )
        if not hits:
            continue
        r = hits[0]
        print("\n" + "=" * 78)
        print(f"{r.info.run_name}    run_id={r.info.run_id}")
        print("=" * 78)

        tg = r.data.tags
        print("  PROVENANCE   git_sha      ", tg.get("git_sha"))
        print("               data_version ", tg.get("data_version"))
        print("               protocol     ", tg.get("protocol_version"))
        print(f"  TARGET       {tg.get('target_id')}  h={tg.get('horizon')}  "
              f"window={tg.get('window')}  stage={tg.get('stage')}")
        print(f"  FEATURES     {tg.get('feature_set')}  encoding={tg.get('encoding')}  "
              f"n={r.data.params.get('n_features')}")

        hp = {k[3:]: v for k, v in r.data.params.items() if k.startswith("hp_")}
        print(f"  TUNED        {hp if hp else '(nothing to tune)'}")
        print(f"               searched {r.data.params.get('n_candidates_evaluated')} candidates")
        print(f"  PROTOCOL     min_train={r.data.params.get('min_train')}  "
              f"purge={r.data.params.get('purge')}  folds={r.data.params.get('n_folds')}  "
              f"holdout={r.data.params.get('holdout_months')}mo  "
              f"seed={r.data.params.get('seed')}")

        m = r.data.metrics
        print(f"  RESULT       MASE {m.get('mase_mean', float('nan')):.3f} "
              f"+/- {m.get('mase_std', float('nan')):.3f}   "
              f"skill_h {m.get('skill_h_mean', float('nan')):+.3f}   "
              f"MAPE {m.get('mape_mean', float('nan')):.2f}%")

        try:
            import json
            path = mlflow.artifacts.download_artifacts(
                run_id=r.info.run_id, artifact_path="features.json")
            feats = json.loads(Path(path).read_text())
            cols, dropped = feats["columns"], feats["dropped"]
            print(f"  COLUMNS ({len(cols)})")
            for i in range(0, len(cols), 4):
                print("               " + "  ".join(f"{c:<22}" for c in cols[i:i + 4]))
            if dropped:
                print(f"  DROPPED      {dropped}   (zero-variance in this window)")
        except Exception as e:  # noqa: BLE001
            print(f"  COLUMNS      unavailable ({type(e).__name__})")

        print(f"  ARTIFACTS    mlflow ui -> {r.info.run_id} -> Artifacts")
        print("               forecast_vs_actual.png | predictions.csv | "
              "tuning_results.csv")
        print("               training_frame.parquet | features.json | folds.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--window", default=None, help="Filter to one window.")
    ap.add_argument("--stage", default="cv", choices=["cv", "holdout"])
    ap.add_argument("--csv", default=None, help="Also write the table here.")
    ap.add_argument("--detail", type=int, default=0, metavar="N",
                    help="Also print full provenance for the top N runs.")
    a = ap.parse_args()

    df = add_skill_rw(fetch(a.target, a.horizon, a.stage))
    if a.window:
        df = df[df["window"] == a.window]

    show = df[["model_family", "feature_set", "window", "n_folds",
               "mase_mean", "mase_std", "skill_h_mean", "skill_rw", "mape_mean"]]
    with pd.option_context("display.float_format", lambda v: f"{v:8.3f}"):
        print(f"\n{a.target} h={a.horizon}  stage={a.stage}  "
              f"(ranked by MASE, lower is better)\n")
        print(show.to_string(index=False))

    print("\n  mase_std matters as much as mase_mean: a model that wins on average")
    print("  by being wildly variable is not a better model.")
    print("  skill_h is measured against the SEASONAL naive; skill_rw against the")
    print("  RANDOM WALK. On this series the second is the demanding one.\n")

    if a.detail:
        detail(a.target, a.horizon, show.head(a.detail), a.stage)

    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
