import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "selection_bootstrap", Path(__file__).resolve().parents[1] / "scripts" / "selection_bootstrap.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def test_indices_shape_range_and_block_structure():
    idx = sb.block_indices(41, 6, 200, np.random.default_rng(0))
    assert idx.shape == (200, 41)
    assert idx.min() >= 0 and idx.max() <= 40
    # inside each block of 6 the folds are consecutive
    first = idx[:, :36].reshape(200, 6, 6)
    assert (np.diff(first, axis=2) == 1).all()


def test_block_one_is_iid_and_bad_block_rejected():
    idx = sb.block_indices(10, 1, 50, np.random.default_rng(0))
    assert idx.shape == (50, 10)
    for bad in (0, 11):
        try:
            sb.block_indices(10, bad, 5, np.random.default_rng(0))
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def _synthetic():
    rng = np.random.default_rng(1)
    base = rng.normal(0.4, 0.05, 41)
    return pd.DataFrame({"lead": base - 0.15, "trend": base, "other": base - 0.05})


def test_seed_reproducible_and_summaries():
    M = _synthetic()
    a = sb.bootstrap_ranking(M, "lead", "trend", reps=500, seed=3)
    b = sb.bootstrap_ranking(M, "lead", "trend", reps=500, seed=3)
    assert a["p_leader_beats_baseline"] == b["p_leader_beats_baseline"] == 1.0
    assert a["diff_ci90"] == b["diff_ci90"]
    lo, hi = a["diff_ci90"]
    assert lo <= hi < 0 and abs(a["diff_mean"] + 0.15) < 1e-9
    assert abs(a["share_best"].sum() - 1) < 1e-9
    assert a["share_best"]["lead"] == 1.0 and a["share_best"]["trend"] == 0.0
