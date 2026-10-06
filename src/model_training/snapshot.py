"""BigQuery -> parquet. The only module in this package that touches the warehouse.

    python -m src.model_training.snapshot

Writes two files into data/processed/:

    panel_{data_version}.parquet      levels, one row per month, one column per series
    series_meta_{data_version}.csv    col_name, category, kappa, transform, coverage

`data_version` is MAX(pulled_at) from raw.bcrp_observations and IS the filename,
so no run can claim a data version it did not load (docs/training_plan.md 4.0).

Why a snapshot at all, when the warehouse is right there:

  1. A multi-hour sweep must not straddle an ETL re-run. The warehouse is
     append-only and BCRP revises figures, so a run that queried at 21:00 and one
     that queried at 23:00 may not be comparable -- and nothing in the MLflow
     table would show it.
  2. Querying per fit spends free-tier quota for no benefit. The panel is ~80 KB.
  3. dataset.py becomes unit-testable offline against a fixture parquet, and
     container-ready at roadmap step 12 with no change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_collection.config import KAPPA_BY_COL, TRANSFORM_BY_COL

# google.cloud.bigquery is imported INSIDE the functions that need it, not at
# module level. train.py imports latest_snapshot() from here, so a top-level
# import would drag the whole cloud SDK into every training run and every unit
# test -- and the tests are meant to run offline against a fixture parquet.

PROJECT_ID = "pagos-telefono-26"
OUT_DIR = Path("data/processed")

PANEL_SQL = f"select * from `{PROJECT_ID}.marts.monthly_panel` order by obs_date"
COVERAGE_SQL = f"select * from `{PROJECT_ID}.marts.series_coverage`"
VERSION_SQL = f"select max(pulled_at) as data_version from `{PROJECT_ID}.raw.bcrp_observations`"
# The ONE Trends pull the panel was built from (stg_trends enforces a single pull_id).
TRENDS_PULL_SQL = f"select distinct pull_id from `{PROJECT_ID}.staging.stg_trends`"


def compose_version(bcrp_version: str, trends_pull_ids: list[str]) -> str:
    """`{bcrp}` or, once Google Trends is in the panel, `{bcrp}_gt{pull date}`.

    WHY THE SUFFIX (protocol 1.7). The BCRP timestamp alone no longer identifies the
    panel: a new Trends pull, or a different `trends_pull_id`, changes gt_yape_plin
    without changing MAX(pulled_at). Same filename, different data -- and the 1.6
    snapshot `panel_20261005T000000Z` is referenced by logged runs, which must never
    be overwritten (training_plan.md 4.0 rule 3). Exactly one pull may feed the panel
    (the single-pull rule); more than one is refused here as well as in dbt.
    """
    if not trends_pull_ids:
        return bcrp_version
    if len(trends_pull_ids) != 1:
        raise RuntimeError(
            f"stg_trends holds {len(trends_pull_ids)} pulls {sorted(trends_pull_ids)}; the "
            "single-pull rule allows exactly one. Check var trends_pull_id and rebuild dbt.")
    return f"{bcrp_version}_gt{trends_pull_ids[0].replace('-', '')}"


def data_version(client) -> str:
    """MAX(pulled_at) of the BCRP pulls, plus the Trends pull; a valid Windows filename.

    Colons are illegal in Windows filenames, so the ISO timestamp is compacted
    to 20260829T180322Z rather than kept in full.
    """
    from google.api_core.exceptions import NotFound  # lazy, as below

    ts = client.query(VERSION_SQL).to_dataframe()["data_version"].iloc[0]
    bcrp = pd.Timestamp(ts).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
    try:
        pulls = client.query(TRENDS_PULL_SQL).to_dataframe()["pull_id"].tolist()
    except NotFound:                      # warehouse built before protocol 1.7: no Trends
        pulls = []
    return compose_version(bcrp, [str(p) for p in pulls])


def build(client) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    from google.api_core.exceptions import NotFound  # lazy, as above

    version = data_version(client)

    panel = client.query(PANEL_SQL).to_dataframe()
    panel["obs_date"] = pd.to_datetime(panel["obs_date"])
    panel = panel.set_index("obs_date").sort_index()

    try:
        coverage = client.query(COVERAGE_SQL).to_dataframe()
    except NotFound as e:
        # The commonest first-run failure, and a bare 404 traceback does not say
        # what to do about it. The snapshot depends on a dbt model, so a stale
        # warehouse shows up here rather than at dbt time.
        raise RuntimeError(
            "marts.series_coverage does not exist yet.\n\n"
            "The snapshot needs it for the interior-gap assertion. Build it:\n"
            "    deactivate\n"
            "    .\\.venv-dbt\\Scripts\\Activate.ps1\n"
            "    cd pagos_dbt; dbt build; cd ..\n"
            "    deactivate\n"
            "    .\\.venv\\Scripts\\Activate.ps1\n\n"
            "If dbt has already run, check that models/marts/series_coverage.sql "
            "is present and that `dbt build` reported it as created."
        ) from e

    # kappa and transform come from config.py, NOT from the warehouse: they are
    # declared assumptions, not facts computed from data (training_plan.md 4.1).
    # Coverage is the opposite, which is why it IS a dbt model.
    meta = coverage.assign(
        kappa=coverage["col_name"].map(KAPPA_BY_COL),
        transform=coverage["col_name"].map(TRANSFORM_BY_COL),
    )

    # A panel column with no kappa would let dataset.py build a feature from a
    # month that was not knowable at forecast time. Fail here, loudly, rather
    # than three hours into a sweep.
    missing = sorted(set(panel.columns) - set(meta.loc[meta["kappa"].notna(), "col_name"]))
    if missing:
        raise RuntimeError(
            f"Panel columns with no kappa/transform in config.py: {missing}. "
            "Re-run load_to_bigquery.py and dbt build after editing config.py."
        )

    orphans = sorted(set(meta["col_name"]) - set(panel.columns))
    if orphans:
        raise RuntimeError(f"Series in metadata but not in the panel: {orphans}. Re-run dbt build.")

    return version, panel, meta


def main() -> None:
    from google.cloud import bigquery  # lazy: see the note at the top

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Query and report; write nothing.")
    args = ap.parse_args()

    client = bigquery.Client(project=PROJECT_ID)
    version, panel, meta = build(client)

    print(f"data_version : {version}")
    print(f"panel        : {panel.shape[0]} months x {panel.shape[1]} series")
    print(f"span         : {panel.index.min():%Y-%m} -> {panel.index.max():%Y-%m}")

    gaps = meta.loc[meta["interior_gaps"] != 0, "col_name"].tolist()
    if gaps:
        raise RuntimeError(
            f"Series with interior gaps: {gaps}. A hole inside a series' own coverage "
            "silently corrupts every lag and moving average built from it. Fix the "
            "warehouse before snapshotting."
        )
    print("interior gaps: none")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p_panel = OUT_DIR / f"panel_{version}.parquet"
    p_meta = OUT_DIR / f"series_meta_{version}.csv"
    panel.to_parquet(p_panel)
    meta.to_csv(p_meta, index=False)

    print(f"\nwrote {p_panel}  ({p_panel.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {p_meta}")


def latest_snapshot(out_dir: Path = OUT_DIR) -> Path:
    """Newest panel_*.parquet by data_version. Used as the default by train.py."""
    candidates = sorted(out_dir.glob("panel_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No snapshot in {out_dir}. Run: python -m src.model_training.snapshot"
        )
    return candidates[-1]


if __name__ == "__main__":
    main()
