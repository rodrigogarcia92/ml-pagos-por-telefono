"""Moving-block bootstrap of the model ranking. Read-only on data and MLflow.

    .venv\\Scripts\\python.exe scripts/selection_bootstrap.py

Takes the per-fold MASE that MLflow logged for the candidates of the selection chart
(target t3, window w2019, 41 expanding CV folds, newest protocol) and resamples the fold
sequence in overlapping blocks of 6 consecutive folds (5,000 replicates, fixed seed).
Because every candidate is evaluated on the same resampled folds, comparisons stay paired.

Reports (a) the share of replicates where the ensemble's mean MASE is below naive_drift's,
(b) the 90% percentile interval of the mean paired difference (ensemble - naive_drift),
(c) how often each candidate has the lowest mean MASE.

This is a stability check that supports the pre-registered decision rule (paired SE); it
did not drive the selection. No MLflow run is created and nothing is written.
"""

from __future__ import annotations

import argparse
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BLOCK, REPLICATES, SEED = 6, 5000, 20261005


def block_indices(n: int, block: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, n) fold indices: ceil(n / block) overlapping blocks of consecutive folds, cut to n."""
    if not 1 <= block <= n:
        raise ValueError("block must be between 1 and n")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)
    return idx[:, :n]


def bootstrap_ranking(M: pd.DataFrame, leader: str, baseline: str, *, block: int = BLOCK,
                      reps: int = REPLICATES, seed: int = SEED) -> dict:
    """M: folds x models of MASE. Resample folds in blocks; return the three summaries."""
    idx = block_indices(len(M), block, reps, np.random.default_rng(seed))
    vals = M.to_numpy()[idx]                                  # (reps, n, models)
    means = vals.mean(axis=1)                                 # (reps, models)
    cols = list(M.columns)
    diff = means[:, cols.index(leader)] - means[:, cols.index(baseline)]
    best = pd.Series(np.bincount(means.argmin(axis=1), minlength=len(cols)) / reps, index=cols)
    return {
        "p_leader_beats_baseline": float((diff < 0).mean()),
        "diff_mean": float(diff.mean()),
        "diff_ci90": (float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))),
        "share_best": best,
    }


def load_fold_mase(db: str, snapshot: str | None) -> tuple[pd.DataFrame, str, str]:
    """Reuse the figure script's read-only MLflow queries (same runs as fig2_selection.png)."""
    spec = importlib.util.spec_from_file_location("make_readme_figures", ROOT / "scripts" / "make_readme_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    snap = Path(snapshot) if snapshot else sorted((ROOT / "data" / "processed").glob("panel_*.parquet"))[-1]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        M = mod.fold_mase(con, mod._parents(con, snap.stem.removeprefix("panel_")))
    finally:
        con.close()
    return M, mod.CANDIDATES[mod.LEADER], mod.CANDIDATES[mod.BASELINE]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--snapshot", default=None, help="panel_*.parquet (default: newest in data/processed)")
    ap.add_argument("--db", default=str(ROOT / "mlflow" / "mlflow.db"), help="MLflow SQLite backend")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--reps", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    M, leader, baseline = load_fold_mase(a.db, a.snapshot)
    if M.isna().any().any():
        raise SystemExit("Some candidate is missing folds; refusing to bootstrap unpaired data.")
    r = bootstrap_ranking(M, leader, baseline, block=a.block, reps=a.reps, seed=a.seed)

    print(f"{len(M)} folds x {M.shape[1]} candidates | block {a.block} | {a.reps} replicates | seed {a.seed}")
    print(f"observed mean paired MASE difference ({leader} - {baseline}): {(M[leader] - M[baseline]).mean():+.3f}")
    print(f"(a) ensemble mean MASE below trend line in {100 * r['p_leader_beats_baseline']:.1f}% of replicates")
    lo, hi = r["diff_ci90"]
    print(f"(b) 90% interval of mean paired difference: [{lo:+.3f}, {hi:+.3f}]")
    print("(c) share of replicates with the lowest mean MASE:")
    tab = pd.DataFrame({"observed MASE": M.mean(), "best in % of replicates": 100 * r["share_best"]})
    print(tab.sort_values("best in % of replicates", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
