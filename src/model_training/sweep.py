"""Run many configurations from one spec.

    python -m src.model_training.sweep --config configs/sweeps/s0_smoke.yaml --dry-run
    python -m src.model_training.sweep --config configs/sweeps/s0_smoke.yaml

Two properties that matter more than speed:

  RESUMABILITY. Before fitting anything, ask MLflow whether a FINISHED run with
  this exact tag set already exists, and skip if so. A multi-hour sweep will be
  interrupted, and restarting from scratch is how data_version ends up
  inconsistent within one experiment.

  --dry-run. Prints run names and counts without fitting. Catches a
  mis-specified grid before it costs an evening.

ON PARALLELISM. Runs are executed SEQUENTIALLY here, and when joblib is added
the workers will fit models and RETURN results while this parent process does
all the MLflow logging. One writer, always: the tracking server is a single
uvicorn worker over SQLite, and concurrent writers produce lock contention that
surfaces three hours into an overnight sweep. A crashed worker also cannot leave
a half-written run behind if it never had a handle to one.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import yaml

from src.model_training import dataset, registry, tracking
from src.model_training.snapshot import latest_snapshot
from src.model_training.train import RunConfig, run


def _pairs(spec: dict) -> list[tuple[str, int]]:
    """(target, horizon) pairs, with the horizon taken FROM the target.

    t2 and t3 are the same series at h=1 and h=3; t4 and t5 likewise. Crossing
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
                f"(t2/t4 are h=1, t3/t5 are h=3)."
            )
        pairs.append((target, h))
    return pairs


def expand(spec: dict, snapshot: str) -> list[RunConfig]:
    """Cartesian product of the spec, minus combinations that make no sense."""
    out = []
    for (target, horizon), window, model, fs in itertools.product(
        _pairs(spec), spec["windows"],
        spec["models"], spec.get("feature_sets", ["none"]),
    ):
        is_naive = model in registry.NAIVE_FAMILIES
        # A naive model has no features, so running it once per feature set
        # would produce identical duplicate runs.
        if is_naive and fs != spec.get("feature_sets", ["none"])[0]:
            continue
        out.append(RunConfig(
            target_id=target, horizon=horizon, window=window,
            model_family=model, feature_set="none" if is_naive else fs,
            stage=spec.get("stage", "cv"), snapshot=snapshot,
            tune=spec.get("tune", {}).get(model, {}),
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-run even if finished.")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    snapshot = spec.get("snapshot") or str(latest_snapshot())
    configs = expand(spec, snapshot)

    print(f"sweep     : {spec.get('name', Path(a.config).stem)}")
    print(f"snapshot  : {Path(snapshot).name}")
    print(f"runs      : {len(configs)}\n")

    for cfg in configs:
        ctx = tracking.RunContext(
            target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
            feature_set=cfg.feature_set, model_family=cfg.model_family,
            encoding="none", stage=cfg.stage, snapshot_path=snapshot,
        )
        print(f"  {cfg.target_id}_h{cfg.horizon}  {ctx.run_name}")

    if a.dry_run:
        print("\n--dry-run: nothing fitted.")
        return

    done = skipped = failed = 0
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        ctx = tracking.RunContext(
            target_id=cfg.target_id, horizon=cfg.horizon, window=cfg.window,
            feature_set=cfg.feature_set, model_family=cfg.model_family,
            encoding="none", stage=cfg.stage, snapshot_path=snapshot,
        )
        label = f"[{i}/{len(configs)}] {cfg.target_id}_h{cfg.horizon} {ctx.run_name}"

        if not a.force and tracking.already_done(ctx):
            print(f"{label}  SKIP (already finished)")
            skipped += 1
            continue
        try:
            res = run(cfg)
            print(f"{label}  MASE={res.metrics['mase_mean']:.3f} "
                  f"(sd {res.metrics['mase_std']:.3f})  "
                  f"skill={res.metrics['skill_h_mean']:+.3f}  folds={res.n_folds}")
            done += 1
        except Exception as e:  # noqa: BLE001 -- one bad config must not kill the sweep
            print(f"{label}  FAILED: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{done} run, {skipped} skipped, {failed} failed "
          f"in {time.time() - t0:.0f}s")
    if failed:
        print("Failed configurations are NOT recorded in MLflow. Fix and re-run; "
              "finished ones will be skipped.")


if __name__ == "__main__":
    main()
