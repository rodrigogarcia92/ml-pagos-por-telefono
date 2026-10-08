"""Monthly refresh: new BCRP data -> warehouse -> snapshot -> forecast -> drift check.

    python scripts/monthly_refresh.py --dry-run --skip-etl      # print the plan, write nothing
    python scripts/monthly_refresh.py --skip-etl                # forecast from the newest local snapshot
    python scripts/monthly_refresh.py                           # the whole chain
    python scripts/monthly_refresh.py --from-stage predict

Stages, in order:
  fetch     python -m src.data_collection.fetch_target_series      (BCRP -> data/raw/bcrp/)
  load      python -m src.data_collection.load_to_bigquery          (raw JSON -> BigQuery `raw`)
  dbt       <dbt-bin> build, cwd pagos_dbt                          (the dbt environment, never in-process)
  snapshot  python -m src.model_training.snapshot                   (BigQuery -> parquet)
  predict   src.forecasting.predict                                 (the frozen model, newest snapshot)
  monitor   src.forecasting.monitor                                 (drift check against actuals)

IDEMPOTENCE IS DEFINED BY THE LAST PUBLISHED TARGET MONTH, not by data_version: every fetch creates
a new data_version, but BCRP publishes a new month only about once a month. If the newest snapshot's
last actual month equals the last_actual_month of the newest forecast in history.csv, `predict`
writes nothing, `monitor` still runs, and the run ends "no new month published" with exit 0.

Exit codes: 0 done (forecast made, or no new month); 2 done but the monitor says `alert`;
1 a stage failed. A failure is reported as stage, command and the last 20 lines of its output --
never a stack trace -- and BCRP bot protection (non-JSON replies) is a named error.
`forecasts/last_refresh.json` records stages, timings and outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.forecasting import monitor, predict, production
from src.model_training import dataset, tracking
from src.model_training.snapshot import latest_snapshot

STAGES = ("fetch", "load", "dbt", "snapshot", "predict", "monitor")
ETL_STAGES = STAGES[:4]
TAIL_LINES = 20
DBT_DIR = Path("pagos_dbt")
LAST_REFRESH = "last_refresh.json"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class StageError(RuntimeError):
    """A stage failed. Carries what a person needs: which stage, what command, the output tail."""

    kind = "StageFailed"

    def __init__(self, stage: str, command: str, detail: str, tail: str = "",
                 returncode: int | None = None, trace: str = ""):
        self.stage, self.command, self.detail, self.tail, self.returncode = (
            stage, command, detail, tail, returncode)
        self.trace = trace          # kept for last_refresh.json only, never printed
        super().__init__(f"stage '{stage}' failed: {detail}")


class BcrpBlockedError(StageError):
    """BCRP answered with something that is not JSON (typically an Imperva bot-protection page)."""

    kind = "BcrpBlockedError"


class DbtNotFoundError(StageError):
    kind = "DbtNotFoundError"


def tail_of(text: str, n: int = TAIL_LINES) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


def format_error(e: StageError) -> str:
    lines = [f"REFRESH FAILED at stage '{e.stage}'  [{e.kind}]", f"  {e.detail}",
             f"  command  : {e.command}"]
    if e.returncode is not None:
        lines.append(f"  exit code: {e.returncode}")
    if e.tail:
        lines.append(f"  last {TAIL_LINES} lines of output:")
        lines += [f"    | {ln}" for ln in e.tail.splitlines()]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Context and commands
# --------------------------------------------------------------------------- #
@dataclass
class Ctx:
    out_dir: Path
    config: Path
    snapshot: Path | None = None            # explicit override, else the newest local one
    dbt_bin: str | None = None
    root: Path = Path(".")
    # filled while running
    data_version: str | None = None
    target_month: str | None = None
    forecast_file: str | None = None
    point_forecast: float | None = None
    no_new_month: bool = False
    monitor_status: str | None = None
    monitor_exit: int = 0
    notes: list[str] = field(default_factory=list)


def resolve_dbt_bin(explicit: str | None, root: Path = Path(".")) -> str:
    """Explicit executable for dbt: --dbt-bin, then $DBT_BIN, then the repo's .venv-dbt.

    Deliberately NOT a bare `dbt` from PATH: dbt lives in its own environment (it cannot share one
    with MLflow), and the environment that is active here is the wrong one.
    """
    for cand in (explicit, os.environ.get("DBT_BIN")):
        if cand:
            return str(cand)
    for rel in (".venv-dbt/Scripts/dbt.exe", ".venv-dbt/bin/dbt"):
        p = root / rel
        if p.exists():
            return str(p.resolve())
    raise DbtNotFoundError(
        "dbt", "dbt build", "no dbt executable: pass --dbt-bin, set DBT_BIN, or create .venv-dbt "
        "(pip install -r requirements-dbt.txt in a separate venv)")


def stage_command(stage: str, ctx: Ctx) -> list[str] | None:
    """The subprocess a stage runs; None for the in-process stages."""
    py = sys.executable
    if stage == "fetch":
        return [py, "-m", "src.data_collection.fetch_target_series"]
    if stage == "load":
        return [py, "-m", "src.data_collection.load_to_bigquery"]
    if stage == "dbt":
        return [resolve_dbt_bin(ctx.dbt_bin, ctx.root), "build"]
    if stage == "snapshot":
        return [py, "-m", "src.model_training.snapshot"]
    return None


def run_subprocess(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """The one place a subprocess is started (patched in tests)."""
    env = {**os.environ, "PYTHONUTF8": "1"}
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)


def _run_external(stage: str, ctx: Ctx) -> None:
    cmd = stage_command(stage, ctx)
    shown = " ".join(str(c) for c in cmd)
    cwd = ctx.root / DBT_DIR if stage == "dbt" else ctx.root
    try:
        proc = run_subprocess(cmd, cwd=cwd)
    except FileNotFoundError as e:
        raise StageError(stage, shown, f"executable not found: {e}") from e
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if out.strip():
        print(out.rstrip())
    if proc.returncode != 0:
        tail = tail_of(proc.stderr if (proc.stderr or "").strip() else out)
        if stage == "fetch" and ("non-JSON" in out or "BcrpNonJsonResponse" in out):
            raise BcrpBlockedError(
                stage, shown,
                "BCRP did not return JSON (bot protection?). Retry later, run the refresh locally, "
                "or download by hand; do NOT try to get around it (plan O-13).",
                tail_of(out), proc.returncode)
        raise StageError(stage, shown, f"exit code {proc.returncode}", tail, proc.returncode)


# --------------------------------------------------------------------------- #
# In-process stages
# --------------------------------------------------------------------------- #
def newest_snapshot(ctx: Ctx) -> Path:
    return ctx.snapshot or latest_snapshot(ctx.root / "data" / "processed")


def last_actual_month(panel: pd.DataFrame, target_id: str) -> str:
    return str(monitor.actuals(panel, target_id).index[-1])


def last_forecast_actual_month(history_path: Path) -> str | None:
    """last_actual_month of the newest forecast in history.csv, or None when there is none."""
    h = monitor.load_history(history_path)
    if h.empty or "last_actual_month" not in h.columns:
        return None
    return str(h["last_actual_month"].max())


def _stage_predict(ctx: Ctx) -> None:
    snap = newest_snapshot(ctx)
    cfg = production.load(ctx.config)
    panel, meta = dataset.load_snapshot(snap)
    ctx.data_version = tracking.data_version(snap)
    published = last_actual_month(panel, cfg["target_id"])
    done = last_forecast_actual_month(ctx.out_dir / predict.HISTORY)
    if done is not None and published <= done:
        ctx.no_new_month = True
        print(f"no new month published: newest snapshot {snap.name} ends {published}, "
              f"the last forecast was made at {done}. Nothing written.")
        return
    rec = predict.forecast(panel, meta, cfg, data_version=ctx.data_version, cfg_path=ctx.config)
    path, _ = predict.write_outputs(rec, ctx.out_dir)
    ctx.target_month, ctx.forecast_file, ctx.point_forecast = (
        rec["target_month"], str(path), rec["point_forecast"])
    print(predict.render(rec))
    print(f"wrote {path}")


def _stage_monitor(ctx: Ctx) -> None:
    snap = newest_snapshot(ctx)
    rc = monitor.main(["--snapshot", str(snap), "--history", str(ctx.out_dir / predict.HISTORY),
                       "--out", str(ctx.out_dir / "monitor.json"), "--config", str(ctx.config)])
    ctx.monitor_exit = rc
    try:
        ctx.monitor_status = json.loads(
            (ctx.out_dir / "monitor.json").read_text(encoding="utf-8"))["status"]
    except (OSError, KeyError, ValueError):
        ctx.monitor_status = None


IN_PROCESS = {"predict": _stage_predict, "monitor": _stage_monitor}


def run_stage(stage: str, ctx: Ctx) -> None:
    """Run one stage; raises StageError. Patched wholesale in tests."""
    if stage in IN_PROCESS:
        try:
            IN_PROCESS[stage](ctx)
        except StageError:
            raise
        except Exception as e:  # noqa: BLE001 -- any failure becomes a stage error, not a traceback
            raise StageError(stage, f"src.forecasting.{stage}", f"{type(e).__name__}: {e}",
                             trace=traceback.format_exc()) from e
    else:
        _run_external(stage, ctx)


# --------------------------------------------------------------------------- #
# Plan and orchestration
# --------------------------------------------------------------------------- #
def plan(skip_etl: bool, from_stage: str | None) -> list[str]:
    start = "predict" if skip_etl else (from_stage or STAGES[0])
    return list(STAGES[STAGES.index(start):])


def describe(stages: list[str], ctx: Ctx) -> list[str]:
    out = []
    for s in stages:
        if s in IN_PROCESS:
            out.append(f"{s:<9} in-process  src.forecasting.{s}")
        else:
            try:
                cmd = " ".join(str(c) for c in stage_command(s, ctx))
            except DbtNotFoundError as e:
                cmd = f"(unresolved: {e.detail})"
            out.append(f"{s:<9} subprocess  {cmd}" + (f"   [cwd {DBT_DIR}]" if s == "dbt" else ""))
    return out


def preview_predict(ctx: Ctx) -> str:
    """Read-only: what `predict` would do right now (no model is fitted, nothing is written)."""
    try:
        snap = newest_snapshot(ctx)
        cfg = production.load(ctx.config)
        panel, _ = dataset.load_snapshot(snap)
        published = last_actual_month(panel, cfg["target_id"])
        done = last_forecast_actual_month(ctx.out_dir / predict.HISTORY)
    except Exception as e:  # noqa: BLE001
        return f"cannot preview: {type(e).__name__}: {e}"
    verdict = ("no forecast in history yet -> it WOULD forecast" if done is None else
               ("last published month unchanged -> 'no new month published', nothing written"
                if published <= done else f"{published} is newer than {done} -> it WOULD forecast"))
    return (f"newest local snapshot {snap.name}; last actual month {published}; "
            f"last forecast made at {done or '-'}; {verdict}")


def _write_summary(ctx: Ctx, started: datetime, records: list[dict], outcome: str,
                   error: StageError | None) -> Path:
    path = ctx.out_dir / LAST_REFRESH
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "outcome": outcome,                      # forecast | no_new_month | failed
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(sum(r["seconds"] for r in records), 2),
        "stages": records,
        "data_version": ctx.data_version,
        "target_month": ctx.target_month,
        "forecast_file": ctx.forecast_file,
        "point_forecast": ctx.point_forecast,
        "monitor_status": ctx.monitor_status,
        "monitor_exit": ctx.monitor_exit,
        "error": None if error is None else {
            "stage": error.stage, "kind": error.kind, "detail": error.detail,
            "command": error.command, "returncode": error.returncode, "stderr_tail": error.tail,
            "traceback": error.trace or None},
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def refresh(stages: list[str], ctx: Ctx) -> int:
    """Run the stages in order. Returns the process exit code (0 / 2 / 1)."""
    started = datetime.now(UTC)
    records: list[dict] = []
    error: StageError | None = None
    for stage in STAGES:
        if stage not in stages:
            records.append({"stage": stage, "status": "skipped", "seconds": 0.0})
            continue
        print(f"\n=== stage: {stage} ===")
        t0 = time.monotonic()
        try:
            run_stage(stage, ctx)
            records.append({"stage": stage, "status": "ok", "seconds": round(time.monotonic() - t0, 2)})
        except StageError as e:
            records.append({"stage": stage, "status": "failed",
                            "seconds": round(time.monotonic() - t0, 2)})
            error = e
            break

    if error is not None:
        seen = {r["stage"] for r in records}
        records += [{"stage": st, "status": "not_run", "seconds": 0.0}
                    for st in STAGES if st not in seen]
        print("\n" + format_error(error), file=sys.stderr)
        _write_summary(ctx, started, records, "failed", error)
        return 1

    if ctx.no_new_month:
        outcome = "no_new_month"
    elif ctx.target_month:
        outcome = "forecast"
    else:
        outcome = "monitor_only"          # predict was not part of this run
    _write_summary(ctx, started, records, outcome, None)
    if ctx.no_new_month:
        print("\nno new month published. Nothing to commit.")
    elif ctx.target_month:
        print(f"\nforecast for {ctx.target_month}: {ctx.point_forecast:,.1f}")
    if ctx.monitor_exit == 2:
        print("monitor: ALERT -- two consecutive months missed by more than 10%.")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run and write nothing")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--skip-etl", action="store_true",
                   help="start from the newest local snapshot (stages predict, monitor)")
    g.add_argument("--from-stage", choices=STAGES, help="start at this stage")
    ap.add_argument("--dbt-bin", default=None, help="dbt executable (else $DBT_BIN, else .venv-dbt)")
    ap.add_argument("--out-dir", default=str(predict.OUT_DIR))
    ap.add_argument("--config", default=str(production.DEFAULT_CONFIG))
    ap.add_argument("--snapshot", default=None, help="use this panel_*.parquet instead of the newest")
    a = ap.parse_args(argv)

    ctx = Ctx(out_dir=Path(a.out_dir), config=Path(a.config), dbt_bin=a.dbt_bin,
              snapshot=Path(a.snapshot) if a.snapshot else None)
    stages = plan(a.skip_etl, a.from_stage)

    print("monthly refresh plan:")
    for line in describe(stages, ctx):
        print(f"  {line}")
    if a.dry_run:
        if "predict" in stages:
            print(f"\npredict preview: {preview_predict(ctx)}")
        print("\n--dry-run: nothing run, nothing written.")
        return 0
    return refresh(stages, ctx)


if __name__ == "__main__":
    sys.exit(main())
