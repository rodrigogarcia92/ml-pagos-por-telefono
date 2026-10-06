"""The frozen production configuration: where it lives, how it is read, what it is called."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

PRODUCTION_DIR = Path("configs/production")
DEFAULT_CONFIG = PRODUCTION_DIR / "t3_ens3.yaml"

# The literal marker written into docs/training_plan.md section 11 when the model is frozen.
# scripts/run_holdout.py refuses to run unless it is there: the holdout rule has to exist in
# writing BEFORE the holdout is evaluated.
FREEZE_MARKER = "PRODUCTION-FREEZE t3_ens3 v1"

REQUIRED = ("target_id", "horizon", "kappa", "window", "feature_set", "columns", "model_family",
            "members", "mlflow_run_id", "data_version", "protocol_version", "selected_on",
            "error_band")
BAND_KEYS = ("p05", "p10", "p50", "p90", "p95", "n_folds", "cv_mape_pct")


def load(path: str | Path = DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    bad = [k for k in BAND_KEYS if k not in cfg["error_band"]]
    if bad:
        raise ValueError(f"{path}: error_band is missing {bad}")
    return cfg


def member_params(cfg: dict) -> dict[str, dict]:
    """{member family: its frozen hyperparameters}, the shape registry.build('ens3') takes."""
    return {fam: dict(m["params"]) for fam, m in cfg["members"].items()}


def config_hash(path: str | Path = DEFAULT_CONFIG) -> str:
    """12 hex chars of the file's content (line endings normalised)."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()[:12]


def model_id(cfg: dict) -> str:
    return f"{cfg['target_id']}_{cfg['model_family']}_{cfg['feature_set']}_{cfg['window']}"


def dump(cfg: dict, path: str | Path, header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8", newline="\n")
