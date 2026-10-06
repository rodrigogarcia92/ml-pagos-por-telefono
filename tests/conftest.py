"""Shared offline fixtures for the production-forecasting tests (src/forecasting/).

`prod_panel` mirrors the real snapshot's release calendar on synthetic numbers: the panel index
runs to 2026-09, payments and pbi are published through 2026-07 (kappa = 2), circulante through
2026-08 (kappa = 1). No network, no credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Small, fast, but the real shape of the production members' parameters.
MEMBER_PARAMS = {
    "svr_rbf": {"C": 1, "gamma": "scale", "epsilon": 0.01},
    "rf": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 3, "max_features": 0.6},
    "xgboost": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 30, "subsample": 1.0,
                "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1},
}


def make_prod_panel(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-01", "2026-09-01", freq="MS")
    n = len(idx)
    month = idx.month.to_numpy() - 1
    panel = pd.DataFrame({
        "n_transf_intra_agg": np.exp(np.linspace(np.log(10), np.log(1150), n)
                                     + 0.05 * np.sin(2 * np.pi * month / 12)
                                     + 0.015 * rng.standard_normal(n)),
        "circulante": np.exp(np.linspace(np.log(40000), np.log(100000), n)
                             + 0.01 * rng.standard_normal(n)),
        "pbi_idx": 100 + np.linspace(0, 95, n) + rng.standard_normal(n),
    }, index=idx)
    panel.loc["2026-08-01":, "n_transf_intra_agg"] = np.nan   # published through 2026-07
    panel.loc["2026-08-01":, "pbi_idx"] = np.nan
    panel.loc["2026-09-01":, "circulante"] = np.nan           # published through 2026-08
    meta = pd.DataFrame({"col_name": panel.columns, "kappa": [2, 1, 2],
                         "transform": ["log_diff"] * 3}).set_index("col_name")
    return panel, meta


def write_snapshot(directory, panel, meta, version: str = "20261005T000000Z") -> str:
    path = directory / f"panel_{version}.parquet"
    panel.to_parquet(path)
    meta.reset_index().to_csv(directory / f"series_meta_{version}.csv", index=False)
    return str(path)


@pytest.fixture(scope="session")
def prod_panel():
    return make_prod_panel()


@pytest.fixture(scope="session")
def prod_snapshot(tmp_path_factory, prod_panel):
    panel, meta = prod_panel
    return write_snapshot(tmp_path_factory.mktemp("prod_snap"), panel, meta)


@pytest.fixture(scope="session")
def member_params():
    return {k: dict(v) for k, v in MEMBER_PARAMS.items()}
