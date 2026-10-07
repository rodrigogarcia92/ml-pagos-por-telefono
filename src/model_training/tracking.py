"""MLflow tag and experiment vocabulary — the ONLY module that talks to MLflow's
configuration or sets tags.

Why one module: `training_plan.md` §8 mandates ten tags on every run. Ten tags set
by hand across six model modules will drift within a week, and a sweep with
inconsistent tags cannot be queried — which makes the whole experiment record
worth less than the runs that produced it.

Nothing else in `src/model_training/` may call `mlflow.set_tag`,
`mlflow.set_tracking_uri`, or `mlflow.set_experiment`.

MLflow version note: this project runs MLflow 3.x, where model *stages*
(Staging/Production) are gone and the Registry uses **aliases**. Promotion is
`set_registered_model_alias(name, "production", version)` and loading is
`models:/{name}@production`. See docs/mlflow_setup.md §7.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import mlflow
from dotenv import load_dotenv

# MLFLOW_TRACKING_URI comes from .env — never hardcoded, so the same code runs
# against a local server now and a remote one later without an edit.
load_dotenv()

# Against an HTTP tracking server MLflow writes two lines to stdout for EVERY
# run it creates. At 42 folds per configuration that is 84 lines per parent and
# ~750 for one nine-run sweep, which buries the single line per run that
# actually carries the result. This is MLflow's own supported switch, set here
# rather than in .env so it holds however the code is invoked. Nothing is lost:
# run() returns the run_id and every run is one click away in the UI.
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

# docs/training_plan.md. Part of the resume key (already_done), so bumping it
# makes every earlier run invisible to a sweep -- on purpose: runs made under a
# different protocol are not comparable and must not satisfy "already done".
PROTOCOL_VERSION = "1.7"
# s0-s4 were run under 1.6 and their sweep specs carry no `protocol_version` key, so a
# spec without one means THIS (sweep.py). Protocol 1.7 sweeps say so explicitly.
LEGACY_PROTOCOL_VERSION = "1.6"
KNOWN_PROTOCOLS = (LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION)
EXPERIMENT_PREFIX = os.getenv("MLFLOW_EXPERIMENT_PREFIX", "26.1__")

_SNAPSHOT_RE = re.compile(r"panel_(?P<version>.+)\.parquet$")

# Fewer outer folds than this is a demonstration, not an evaluation
# (training_plan.md 6.4 rule 4). Lives here, not in train.py, because the tag
# vocabulary lives here and train.py imports this module (not the other way round).
MIN_EVALUATION_FOLDS = 8

# Horizons a run may carry: 1 and 3 (t2-t5) plus 5 and 6 (t10, t11; protocol 1.7).
HORIZONS = (1, 3, 5, 6)


def evaluation_status(n_folds: int) -> str:
    """'evaluation' or 'demonstration', from the number of outer folds."""
    return "evaluation" if n_folds >= MIN_EVALUATION_FOLDS else "demonstration"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_sha() -> str:
    """Short commit SHA, suffixed '-dirty' when the tree has uncommitted changes.

    A run tagged with a dirty SHA is not reproducible, and saying so in the tag
    is more useful than silently recording a commit that does not describe the
    code that ran.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_tree_state(cwd: str | Path | None = None, max_paths: int = 20) -> dict:
    """The commit a run started from and what, if anything, differed from it.

    `git_sha` is the bare short SHA (never suffixed). `tree_dirty` says whether the working tree
    differed from that commit when this was called, and `tree_changes` lists up to `max_paths`
    of the changed paths (untracked files included, ignored ones not). A monthly refresh that
    fetched new raw JSON before predicting is legitimately dirty; the list shows it is only that.
    """
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", "core.quotepath=off", *args], text=True, encoding="utf-8",
            stderr=subprocess.DEVNULL, cwd=cwd).strip("\n")
    try:
        sha = git("rev-parse", "--short", "HEAD").strip()
        lines = [ln for ln in git("status", "--porcelain", "--untracked-files=all").splitlines() if ln]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "unknown", "tree_dirty": None, "tree_changes": []}
    paths = sorted(ln[3:].split(" -> ")[-1] for ln in lines)
    return {"git_sha": sha, "tree_dirty": bool(paths), "tree_changes": paths[:max_paths]}


def data_version(snapshot_path: str | Path) -> str:
    """Read the data version out of the snapshot filename (training_plan.md §4.0).

    The version is not typed by hand anywhere: it *is* the filename, so a run
    cannot claim a data version it did not load. The warehouse is append-only and
    BCRP revises figures, so a run without this is not reproducible however
    carefully the seed was logged.
    """
    m = _SNAPSHOT_RE.search(str(snapshot_path))
    if not m:
        raise ValueError(
            f"Cannot read data_version from {snapshot_path!r}. "
            "Expected a path ending in 'panel_{version}.parquet'."
        )
    return m.group("version")


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunContext:
    """Everything that identifies one configuration in the experiment record."""

    target_id: str          # t1 | t2 | t3 | t4 | t5
    horizon: int            # 1 | 3
    window: str             # w2019 | w2021 | w2024
    feature_set: str        # FS0_calendar ... FS5b_stance | FS1_short | none
    model_family: str       # naive_calendar | sarimax | ridge | ... | xgboost
    encoding: str           # int | onehot | none
    stage: str              # cv | holdout        (never 'tune' — see §8.0)
    snapshot_path: str
    protocol_version: str = PROTOCOL_VERSION    # 1.6 for s0-s4, 1.7 from s1b on

    def __post_init__(self) -> None:
        if self.protocol_version not in KNOWN_PROTOCOLS:
            raise ValueError(
                f"protocol_version must be one of {KNOWN_PROTOCOLS}, got {self.protocol_version!r}")
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}, got {self.horizon}")
        if self.stage not in ("cv", "holdout"):
            # Stage A (tuning) creates no runs at all (training_plan.md §8.0):
            # ~35k inner-CV fits at 50-200 ms of run-creation overhead each would
            # cost more than the modelling. It is logged as tuning_results.csv.
            raise ValueError(
                f"stage must be 'cv' or 'holdout', got {self.stage!r}. "
                "Stage A is logged as an artifact, not as runs (§8.0)."
            )

    @property
    def experiment(self) -> str:
        """One experiment per target x horizon: '26.1__t2_proxy_h1'."""
        return f"{EXPERIMENT_PREFIX}{self.target_id}_h{self.horizon}"

    @property
    def run_name(self) -> str:
        """'{model}__{feature_set}__{window}' — training_plan.md §8.1."""
        return f"{self.model_family}__{self.feature_set}__{self.window}"

    def tags(self) -> dict[str, str]:
        """The ten mandatory tags. Set here and nowhere else."""
        return {
            "git_sha": git_sha(),
            "data_version": data_version(self.snapshot_path),
            "target_id": self.target_id,
            "horizon": str(self.horizon),
            "window": self.window,
            "feature_set": self.feature_set,
            "model_family": self.model_family,
            "encoding": self.encoding,
            "protocol_version": self.protocol_version,
            "stage": self.stage,
        }


# --------------------------------------------------------------------------- #
# Run helpers
# --------------------------------------------------------------------------- #
@contextmanager
def parent_run(ctx: RunContext, description: str | None = None, n_folds: int | None = None):
    """Open the parent run for one configuration, with all ten tags attached.

    `n_folds`, when known, adds the eleventh tag `evaluation_status`. It is NOT in
    RunContext.tags() because it is a property of the result, not of the
    configuration, and so must not enter the resume key (already_done).
    """
    mlflow.set_experiment(ctx.experiment)
    tags = ctx.tags()
    if n_folds is not None:
        tags["evaluation_status"] = evaluation_status(n_folds)
    with mlflow.start_run(run_name=ctx.run_name, tags=tags) as run:
        if description:
            mlflow.set_tag("mlflow.note.content", description)
        yield run


@contextmanager
def fold_run(fold: int):
    """Open a child run for one backtest fold.

    Children log metrics and params ONLY — artifacts attach to the parent
    (training_plan.md §8.5). Cloud Storage's always-free tier allows 5,000
    Class A operations per month; per-child artifacts would be ~24,000.
    """
    with mlflow.start_run(run_name=f"fold_{fold:02d}", nested=True) as run:
        yield run


def already_done(ctx: RunContext) -> bool:
    """True when a FINISHED run with this exact tag set already exists.

    Called by sweep.py before fitting anything, so an interrupted multi-hour
    sweep resumes instead of restarting — which is also how `data_version` stays
    consistent within one experiment (training_plan.md §9.1).
    """
    exp = mlflow.get_experiment_by_name(ctx.experiment)
    if exp is None:
        return False
    keys = ("target_id", "horizon", "window", "feature_set", "model_family",
            "encoding", "stage", "data_version", "protocol_version")
    tags = ctx.tags()
    clauses = [f"tags.{k} = '{tags[k]}'" for k in keys]
    clauses.append("attributes.status = 'FINISHED'")
    hits = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=" and ".join(clauses),
        max_results=1,
    )
    return len(hits) > 0
