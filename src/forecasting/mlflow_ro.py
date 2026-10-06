"""Read-only access to the MLflow SQLite backend.

Opened with `mode=ro`, so nothing here can write a run, a tag or a metric. Used by the freeze
script and the figures script; the training path keeps using the MLflow client.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_ro(db: str | Path) -> sqlite3.Connection:
    db = Path(db)
    if not db.exists():
        raise FileNotFoundError(f"MLflow backend not found: {db}")
    return sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)


def find_parent_run(con: sqlite3.Connection, **tags: str) -> str:
    """Run id of the newest FINISHED run carrying exactly these tags.

    e.g. find_parent_run(con, model_family="ens3", feature_set="FS3_activity", target_id="t3",
    window="w2019", stage="cv", data_version="20261005T000000Z", protocol_version="1.7").
    """
    if not tags:
        raise ValueError("at least one tag is required")
    sql = ["select r.run_uuid from runs r"]
    args: list[str] = []
    for i, (k, v) in enumerate(tags.items()):
        sql.append(f"join tags t{i} on t{i}.run_uuid = r.run_uuid and t{i}.key = ? and t{i}.value = ?")
        args += [k, v]
    sql.append("where r.status = 'FINISHED' order by r.start_time desc limit 1")
    row = con.execute(" ".join(sql), args).fetchone()
    if row is None:
        raise LookupError(f"no finished MLflow run with tags {tags}")
    return row[0]


def run_params(con: sqlite3.Connection, run_id: str) -> dict[str, str]:
    return dict(con.execute("select key, value from params where run_uuid = ?", (run_id,)))


def run_metrics(con: sqlite3.Connection, run_id: str) -> dict[str, float]:
    return dict(con.execute(
        "select key, value from latest_metrics where run_uuid = ?", (run_id,)))
