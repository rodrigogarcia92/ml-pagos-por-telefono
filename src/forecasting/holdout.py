"""The one-time holdout (Stage C), behind guards.

    python scripts/run_holdout.py              # prints what it would run, exits 0
    python scripts/run_holdout.py --confirm    # the real run

Stage C is already implemented by train.fit_config (stage="holdout": one expanding, purged fold
per holdout origin, plan 7.3) and driven by the sweep machinery; this module only decides
WHETHER it may run, runs it, and prints the result in the form the pre-registered rule asks for.

It refuses unless ALL of:
  (a) docs/methodology.md (decision log) holds the marker `PRODUCTION-FREEZE t3_ens3 v1` -- the
      holdout rule must exist in writing before the holdout is evaluated;
  (b) `--confirm` was passed;
  (c) MLflow has no finished holdout run for these two parents on this data_version -- the
      holdout is evaluated once.

The two parents are exactly `ens3 x FS3_activity` and `naive_drift x none` (s6_holdout.yaml).
Nothing is tuned: ens3 takes the frozen members from configs/production/t3_ens3.yaml, and
naive_drift runs its (3-candidate) Stage A on the CV rows, as in every CV run.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.forecasting import production
from src.model_training import sweep, tracking
from src.model_training.snapshot import latest_snapshot
from src.model_training.train import RunConfig, context_for, log_config

SPEC = Path("configs/sweeps/s6_holdout.yaml")
PLAN = Path("docs/methodology.md")
EXPECTED = {("ens3", "FS3_activity"), ("naive_drift", "none")}
CHALLENGER, BASELINE = "ens3", "naive_drift"


@dataclass
class Guard:
    name: str
    ok: bool | None          # None = could not be checked
    detail: str


# --------------------------------------------------------------------------- #
# Plan and guards
# --------------------------------------------------------------------------- #
def load_runs(spec_path: str | Path = SPEC, snapshot: str | None = None
              ) -> tuple[dict, str, list[RunConfig]]:
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    snap = snapshot or spec.get("snapshot") or str(latest_snapshot())
    configs = sweep.expand(spec, snap)
    got = {(c.model_family, c.feature_set) for c in configs}
    bad = [c for c in configs if (c.target_id, c.horizon, c.window, c.stage, c.protocol_version)
           != ("t3", 3, "w2019", "holdout", tracking.PROTOCOL_VERSION)]
    if len(configs) != 2 or got != EXPECTED or bad:
        raise ValueError(
            f"{spec_path} must expand to exactly the two parents {sorted(EXPECTED)} on "
            f"t3 / w2019 / holdout / protocol {tracking.PROTOCOL_VERSION}; got {len(configs)}: "
            f"{[context_for(c, snap).run_name for c in configs]}")
    return spec, snap, configs


def marker_present(plan_path: str | Path = PLAN) -> bool:
    p = Path(plan_path)
    return p.exists() and production.FREEZE_MARKER in p.read_text(encoding="utf-8")


def _probe_tracking_server() -> None:
    """Fail in two seconds, not after MLflow's default retries, when the server is down."""
    import mlflow
    import requests

    uri = mlflow.get_tracking_uri()
    if uri.startswith("http"):
        requests.get(uri.rstrip("/") + "/health", timeout=2).raise_for_status()


def finished_holdouts(configs: list[RunConfig], snapshot: str) -> list[str]:
    """Run names that already have a FINISHED holdout run on this data_version."""
    return [context_for(c, snapshot).run_name for c in configs
            if tracking.already_done(context_for(c, snapshot))]


def guards(configs: list[RunConfig], snapshot: str, *, confirm: bool,
           plan_path: str | Path = PLAN) -> list[Guard]:
    out = [Guard(
        f"(a) marker {production.FREEZE_MARKER!r} in {plan_path}", marker_present(plan_path),
        "present" if marker_present(plan_path) else
        "MISSING: write the section 11 freeze entry (with the holdout rule) first")]
    out.append(Guard("(b) --confirm", confirm,
                     "given" if confirm else "not given: nothing will be run"))
    try:
        _probe_tracking_server()
        done = finished_holdouts(configs, snapshot)
        out.append(Guard(
            f"(c) no finished holdout run on data_version {tracking.data_version(snapshot)}",
            not done, "none" if not done else f"ALREADY EVALUATED: {done}"))
    except Exception as e:  # noqa: BLE001 -- unreachable store is a guard result, not a crash
        out.append(Guard("(c) no finished holdout run on this data_version", None,
                         f"could not check MLflow ({type(e).__name__}: {str(e)[:100]})"))
    return out


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
def _parent(client, exp_id: str, cfg: RunConfig, snapshot: str):
    ctx = context_for(cfg, snapshot)
    tg = ctx.tags()
    clauses = [f"tags.{k} = '{tg[k]}'" for k in
               ("target_id", "horizon", "window", "feature_set", "model_family", "encoding",
                "stage", "data_version", "protocol_version")]
    hits = client.search_runs([exp_id], " and ".join(clauses + ["attributes.status = 'FINISHED'"]),
                              max_results=1, order_by=["attributes.start_time DESC"])
    if not hits:
        raise LookupError(f"no finished holdout run for {ctx.run_name}")
    return hits[0]


def _folds(client, exp_id: str, run_id: str) -> pd.DataFrame:
    kids = client.search_runs([exp_id], f"tags.mlflow.parentRunId = '{run_id}'", max_results=500)
    rows = [{"fold": k.info.run_name, "mase": k.data.metrics["mase"], "mape": k.data.metrics["mape"]}
            for k in kids]
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Mean of (a - b) and its standard error, sd / sqrt(n) -- the plan's SE (7.4)."""
    d = (a - b).to_numpy(dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def comparison(configs: list[RunConfig], snapshot: str) -> dict:
    """Read the two finished holdout parents back from MLflow and pair them month by month."""
    import mlflow

    client = mlflow.MlflowClient()
    by = {c.model_family: c for c in configs}
    exp = mlflow.get_experiment_by_name(context_for(configs[0], snapshot).experiment)
    runs = {fam: _parent(client, exp.experiment_id, by[fam], snapshot) for fam in (CHALLENGER, BASELINE)}
    folds = {fam: _folds(client, exp.experiment_id, r.info.run_id) for fam, r in runs.items()}
    n = len(folds[CHALLENGER])
    if len(folds[BASELINE]) != n:
        raise RuntimeError("the two holdout runs have different numbers of folds")
    end = pd.Period(runs[CHALLENGER].data.params["window_target_end"], freq="M")
    months = [str(m) for m in pd.period_range(end=end, periods=n, freq="M")]
    out = {"n_months": n, "first_month": months[0], "last_month": months[-1],
           "data_version": runs[CHALLENGER].data.tags["data_version"],
           "run_ids": {fam: r.info.run_id for fam, r in runs.items()}, "months": months,
           "folds": folds}
    for fam, r in runs.items():
        out[fam] = {"mase": r.data.metrics["mase_mean"], "mape": r.data.metrics["mape_mean"]}
    out["diff_mase"] = paired(folds[CHALLENGER]["mase"], folds[BASELINE]["mase"])
    out["diff_mape"] = paired(folds[CHALLENGER]["mape"], folds[BASELINE]["mape"])
    out["challenger_wins"] = bool(out[CHALLENGER]["mase"] < out[BASELINE]["mase"])
    return out


def render(res: dict) -> str:
    c, b = res[CHALLENGER], res[BASELINE]
    (dm, sm), (dp, sp) = res["diff_mase"], res["diff_mape"]
    span = f"{res['first_month']} .. {res['last_month']}"
    lines = [
        "", f"HOLDOUT RESULT  t3 (h=3)  w2019  {res['n_months']} target months {span}  "
            f"data_version {res['data_version']}", "",
        f"  {'forecast':<22}{'MASE':>9}{'MAPE %':>9}",
        f"  {'ens3 / FS3_activity':<22}{c['mase']:>9.3f}{c['mape']:>9.2f}",
        f"  {'naive_drift':<22}{b['mase']:>9.3f}{b['mape']:>9.2f}", "",
        "  paired per-month difference, ens3 - naive_drift (negative = ens3 better), +/- SE:",
        f"    MASE   {dm:+.3f} +/- {sm:.3f}   ({dm / sm:+.1f} SE)",
        f"    MAPE   {dp:+.2f} +/- {sp:.2f} pp  ({dp / sp:+.1f} SE)", "",
    ]
    if res["challenger_wins"]:
        verdict = "ens3 beats naive_drift on the holdout (lower MASE)."
        readme = ("On the held-out final 12 months, evaluated once after the model was frozen, the "
                  f"ensemble's MASE was {c['mase']:.3f} against {b['mase']:.3f} for the trend-line baseline "
                  f"(MAPE {c['mape']:.1f}% vs {b['mape']:.1f}%).")
    else:
        verdict = ("ens3 does NOT beat naive_drift on the holdout. Per the pre-registered rule, "
                   "naive_drift becomes the production fallback and the README says so.")
        readme = ("On the held-out final 12 months, evaluated once after the model was frozen, the "
                  f"ensemble did NOT beat the trend-line baseline (MASE {c['mase']:.3f} vs {b['mase']:.3f}; "
                  f"MAPE {c['mape']:.1f}% vs {b['mape']:.1f}%). The trend line is the production fallback.")
    lines += [f"  VERDICT (pre-registered rule, point MASE): {verdict}", "",
              "Paste into docs/methodology.md (decision log):", "",
              f"| {pd.Timestamp.today():%Y-%m-%d} | 6.1, 7.3 Stage C | **PRODUCTION-FREEZE t3_ens3 v1 -- holdout result.** "
              f"ens3 FS3 vs naive_drift on {res['n_months']} target months {span} (data_version {res['data_version']}, "
              f"MLflow runs {res['run_ids'][CHALLENGER]} / {res['run_ids'][BASELINE]}): MASE {c['mase']:.3f} vs {b['mase']:.3f}, "
              f"MAPE {c['mape']:.2f}% vs {b['mape']:.2f}%; paired per-month difference (ens3 - drift) MASE {dm:+.3f} +/- {sm:.3f}, "
              f"MAPE {dp:+.2f} +/- {sp:.2f} pp. {verdict} Published whatever it is; nothing re-tuned, re-selected or re-run. | "
              "The one-time test, under the rule pre-registered in the freeze entry. |", "",
              "Paste into README.md:", "", f"> {readme}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(SPEC))
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: the spec's / newest)")
    ap.add_argument("--plan", default=str(PLAN), help="training plan holding the freeze marker")
    ap.add_argument("--confirm", action="store_true", help="actually run the holdout (once, ever)")
    a = ap.parse_args(argv)

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

    spec, snapshot, configs = load_runs(a.config, a.snapshot)
    print(f"sweep     : {spec.get('name', Path(a.config).stem)}   stage: holdout   "
          f"protocol: {sweep.protocol_of(spec)}")
    print(f"snapshot  : {Path(snapshot).name}")
    print(f"runs      : {len(configs)}")
    for c in configs:
        src = f"   hyperparameters {c.frozen_source}" if c.frozen_params else "   Stage A on the CV rows"
        print(f"  t3_h3  {context_for(c, snapshot).run_name}{src}")
    print()
    gs = guards(configs, snapshot, confirm=a.confirm, plan_path=a.plan)
    for g in gs:
        mark = {True: "ok ", False: "NO ", None: "?  "}[g.ok]
        print(f"  [{mark}] {g.name}: {g.detail}")

    if not a.confirm:
        print("\nWithout --confirm nothing is run. This evaluates the final 12 target months, "
              "once; add --confirm only when you mean it.")
        return 0
    blocked = [g for g in gs if g.ok is not True]
    if blocked:
        print("\nREFUSED: " + "; ".join(g.name for g in blocked))
        return 2

    print("\nrunning Stage C ...")
    for c in configs:
        cfg, fit, err = sweep._safe_fit(c)
        if err:
            print(f"  FAILED {context_for(cfg, snapshot).run_name}: {err}")
            print("A failed holdout run is not recorded. Fix the cause; the guards will allow a re-run "
                  "because nothing finished.")
            return 1
        res = log_config(fit)
        print(f"  {res.run_name}  MASE={res.metrics['mase_mean']:.3f}  folds={res.n_folds}")
    print(render(comparison(configs, snapshot)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
