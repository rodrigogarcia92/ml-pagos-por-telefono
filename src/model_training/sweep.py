"""Run many configurations from one spec.

    python -m src.model_training.sweep --config configs/sweeps/s2_proxy_grid.yaml --dry-run
    python -m src.model_training.sweep --config configs/sweeps/s2_proxy_grid.yaml --jobs 6

Two properties that matter more than speed:

  RESUMABILITY. Before fitting anything, ask MLflow whether a FINISHED run with
  this exact tag set already exists, and skip if so. A multi-hour sweep will be
  interrupted, and restarting from scratch is how data_version ends up
  inconsistent within one experiment.

  --dry-run. Prints run names and counts without fitting. Catches a
  mis-specified grid before it costs an evening. Needs no MLflow server.

ON PARALLELISM (--jobs N). Workers call train.fit_config, which does all the
modelling and touches no MLflow, and RETURN the result; this parent process alone
calls train.log_config. One writer, always: the tracking server is a single
uvicorn worker over SQLite, and concurrent writers produce lock contention that
surfaces three hours into an overnight sweep. A crashed worker also cannot leave
a half-written run behind, because it never held a handle to one. Results are
logged as they complete, in completion order, so an interruption loses at most
the configurations still in flight.

SPEC KEYS
  targets, windows, stage, models, tune       as before
  feature_sets                                default feature sets for every model
  feature_sets_by_model                       per-model override, e.g. SARIMAX on
                                              FS0-FS2 only (training_plan.md 7.2)
  protocol_version                            the tag every run of the sweep carries.
                                              A spec WITHOUT the key is 1.6 -- s0 to s4
                                              predate it and must keep producing 1.6
                                              runs -- so protocol 1.7 specs (s1b, s5)
                                              say "1.7" explicitly.
  Naive models run once, with feature set "none".
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from src.model_training import dataset, registry, tracking
from src.model_training.snapshot import latest_snapshot
from src.model_training.train import RunConfig, check_protocol, context_for, fit_config, log_config


def _pairs(spec: dict) -> list[tuple[str, int]]:
    """(target, horizon) pairs, with the horizon taken FROM the target.

    t2 and t3 are the same series at h=1 and h=3; t4 and t5 likewise (t10 / t11: h=5 / h=6,
    protocol 1.7). Crossing
    targets with horizons independently would happily produce a run tagged
    `target_id=t2, horizon=3` -- which is t3 wearing the wrong name, and the
    MLflow table would carry that lie forever.

    So `horizons` is optional. Omit it and each target contributes its own
    canonical horizon. Supply it and it is treated as a FILTER, and any
    contradictory pair is rejected loudly rather than silently renamed.
    """
    wanted = spec.get("horizons")
    pairs = []
    for target in spec["targets"]:
        h = dataset.TARGETS[target]["horizon"]
        if wanted is not None and h not in wanted:
            raise ValueError(
                f"target {target} has horizon {h}, which is not in horizons={wanted}. "
                f"Drop the horizons key, or use the target whose horizon you mean "
                f"(t2/t4 are h=1, t3/t5 are h=3, t10 is h=5, t11 is h=6)."
            )
        pairs.append((target, h))
    return pairs


def protocol_of(spec: dict) -> str:
    """The sweep's protocol tag. No key -> 1.6, because s0-s4 predate the key."""
    p = str(spec.get("protocol_version", tracking.LEGACY_PROTOCOL_VERSION))
    if p not in tracking.KNOWN_PROTOCOLS:
        raise ValueError(f"protocol_version {p!r} is not one of {tracking.KNOWN_PROTOCOLS}")
    return p


def _feature_sets(spec: dict, model: str) -> list[str]:
    if model in registry.NAIVE_FAMILIES:
        return ["none"]      # no features: one run, not one per feature set
    return spec.get("feature_sets_by_model", {}).get(model) or spec.get("feature_sets", ["none"])


def expand(spec: dict, snapshot: str) -> list[RunConfig]:
    """Cartesian product of the spec, minus combinations that make no sense."""
    unknown = [m for m in spec["models"] if m not in registry.MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; known: {sorted(registry.MODELS)}")

    protocol = protocol_of(spec)
    out = []
    for target, horizon in _pairs(spec):
        for window, model in itertools.product(spec["windows"], spec["models"]):
            for fs in _feature_sets(spec, model):
                out.append(RunConfig(
                    target_id=target, horizon=horizon, window=window,
                    model_family=model, feature_set=fs,
                    stage=spec.get("stage", "cv"), snapshot=snapshot,
                    tune=spec.get("tune", {}).get(model, {}),
                    protocol_version=protocol,
                ))
                check_protocol(out[-1])      # a 1.7-only target / model / set under a 1.6 spec raises
    return out


def _safe_fit(cfg: RunConfig):
    """Worker entry point. Module-level so loky can pickle it; never raises."""
    try:
        return cfg, fit_config(cfg), None
    except Exception as e:  # noqa: BLE001 -- one bad config must not kill the sweep
        return cfg, None, f"{type(e).__name__}: {e}"


def _label(i: int, n: int, cfg: RunConfig, snapshot: str) -> str:
    ctx = context_for(cfg, snapshot)
    return f"[{i}/{n}] {cfg.target_id}_h{cfg.horizon} {ctx.run_name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if finished.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Worker processes for fitting. MLflow logging stays in this process.")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    snapshot = spec.get("snapshot") or str(latest_snapshot())
    configs = expand(spec, snapshot)

    print(f"sweep     : {spec.get('name', Path(a.config).stem)}")
    print(f"snapshot  : {Path(snapshot).name}")
    print(f"protocol  : {protocol_of(spec)}")
    print(f"runs      : {len(configs)}\n")

    for cfg in configs:
        print(f"  {cfg.target_id}_h{cfg.horizon}  {context_for(cfg, snapshot).run_name}")

    if a.dry_run:
        print("\n--dry-run: nothing fitted.")
        return

    t0 = time.time()
    todo, skipped = [], 0
    for i, cfg in enumerate(configs, 1):
        if not a.force and tracking.already_done(context_for(cfg, snapshot)):
            print(f"{_label(i, len(configs), cfg, snapshot)}  SKIP (already finished)")
            skipped += 1
        else:
            todo.append(cfg)

    done = failed = 0

    def record(n: int, cfg, fit, err) -> None:
        nonlocal done, failed
        label = _label(n, len(todo), cfg, snapshot)
        if err:
            print(f"{label}  FAILED: {err}")
            failed += 1
            return
        res = log_config(fit)
        print(f"{label}  MASE={res.metrics['mase_mean']:.3f} "
              f"(sd {res.metrics['mase_std']:.3f})  "
              f"skill={res.metrics['skill_h_mean']:+.3f}  folds={res.n_folds}")
        done += 1

    if a.jobs > 1 and len(todo) > 1:
        from joblib import Parallel, delayed

        stream = Parallel(n_jobs=a.jobs, return_as="generator_unordered")(
            delayed(_safe_fit)(c) for c in todo
        )
        for n, (cfg, fit, err) in enumerate(stream, 1):
            record(n, cfg, fit, err)
    else:
        for n, cfg in enumerate(todo, 1):
            record(n, *_safe_fit(cfg))

    print(f"\n{done} run, {skipped} skipped, {failed} failed "
          f"in {time.time() - t0:.0f}s")
    if failed:
        print("Failed configurations are NOT recorded in MLflow. Fix and re-run; "
              "finished ones will be skipped.")


if __name__ == "__main__":
    main()
