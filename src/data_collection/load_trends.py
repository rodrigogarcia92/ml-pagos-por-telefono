"""Load Google Trends pulls into BigQuery `raw.trends_observations`.

Run from the project root:
    python -m src.data_collection.load_trends --dry-run      # offline, no credentials
    python -m src.data_collection.load_trends                # real load

Same pattern as load_to_bigquery.py (bcrp_observations): append-only, idempotent by
`source_batch`, no uniqueness constraint in BigQuery (dbt asserts it), FLOAT64 values.
Differences, each on purpose:

  * DRY RUN NEVER CONNECTS. It parses and validates every pull from disk and says
    what it would write, so it works on a machine with no GCP credentials. It does
    NOT ask BigQuery which pulls are already loaded; the real run does, and skips them.
  * LONG FORMAT, ONE ROW PER (pull, entity, month). The raw table keeps both
    entities as exported; stg_trends sums them (plan 4.5: consolidated = yape + plin).
  * ONLY COMPLETE MONTHS are loaded (and not the final month if the pulls disagree
    on it -- see trends.final_month_check). The incomplete pull month never reaches
    the warehouse, so no downstream model has to remember to exclude it.
  * EVERY PULL IS KEPT. Which one feeds the panel is decided in dbt, not here: the
    single-pull rule (plan 4.5) is a staging concern, `var('trends_pull_id')`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.data_collection import trends
from src.data_collection.trends import SERIES_CODE

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATIONS_COLUMNS = ["pull_id", "pull_date", "series_code", "entity", "obs_date",
                        "value", "source_batch"]


def _project_and_dataset() -> tuple[str | None, str]:
    """Read .env lazily, so the dry run does not need it to exist."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("GCP_PROJECT_ID"), os.getenv("BQ_DATASET_RAW", "raw")


def to_long(pull: trends.Pull) -> pd.DataFrame:
    """One pull -> rows for raw.trends_observations (complete months only)."""
    d = pull.data
    parts = []
    for entity in ("yape", "plin"):
        parts.append(pd.DataFrame({
            "pull_id": pull.pull_date,
            "pull_date": pd.Timestamp(pull.pull_date).date(),
            "series_code": SERIES_CODE,
            "entity": entity,
            "obs_date": d["date"].dt.date,
            "value": d[entity].astype(float),
            "source_batch": pull.csv.name,
        }))
    return pd.concat(parts, ignore_index=True)[OBSERVATIONS_COLUMNS]


def read_all(raw_dir: Path = trends.RAW_DIR) -> tuple[list[trends.Pull], pd.DataFrame]:
    """Parse and cross-check every pull. Raises on a missing/mismatched sidecar."""
    pulls = trends.load_pulls(raw_dir)
    if not pulls:
        raise SystemExit(f"No *_yape_plin.csv in {raw_dir}")
    return pulls, pd.concat([to_long(p) for p in pulls], ignore_index=True)


def _schema():
    from google.cloud import bigquery

    return [
        bigquery.SchemaField("pull_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("pull_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("series_code", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("entity", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("obs_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("value", "FLOAT64"),
        bigquery.SchemaField("source_batch", "STRING", mode="REQUIRED"),
    ]


def ensure_table(client, table_id: str) -> None:
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery

    try:
        client.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=_schema())
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH, field="obs_date")
        table.clustering_fields = ["pull_id"]
        client.create_table(table)
        print(f"  created {table_id}")


def already_loaded(client, table_id: str) -> set[str]:
    rows = client.query(f"SELECT DISTINCT source_batch FROM `{table_id}`").result()
    return {r.source_batch for r in rows}


def summarize(pulls: list[trends.Pull], long: pd.DataFrame) -> None:
    for p in pulls:
        n = int((long["pull_id"] == p.pull_date).sum())
        s = trends.consolidated(p.data)
        check = p.meta.get("final_month_check", {}).get("status", "?")
        print(f"  {p.csv.name}: {n} row(s) = 2 entities x {len(p.data)} months "
              f"({s.index[0]:%Y-%m} .. {s.index[-1]:%Y-%m}); final-month check: {check}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse, validate and report. Never connects to BigQuery.")
    ap.add_argument("--dir", default=str(trends.RAW_DIR))
    a = ap.parse_args()

    pulls, long = read_all(Path(a.dir))
    print(f"Found {len(pulls)} pull(s) in {a.dir}")
    summarize(pulls, long)

    if a.dry_run:
        project, dataset = _project_and_dataset()
        print(f"\nTarget table : {project or '<GCP_PROJECT_ID unset>'}.{dataset}.trends_observations")
        print(f"Would load   : {len(long):,} row(s) from {len(pulls)} pull(s) "
              "(BigQuery not consulted, so 'already loaded' pulls are not subtracted)")
        print("--dry-run: no data written.")
        return

    from google.cloud import bigquery

    project, dataset = _project_and_dataset()
    if not project:
        raise SystemExit("GCP_PROJECT_ID is not set (see .env.example). Use --dry-run offline.")
    table_id = f"{project}.{dataset}.trends_observations"

    client = bigquery.Client(project=project)
    ensure_table(client, table_id)
    seen = already_loaded(client, table_id)
    new = long[~long["source_batch"].isin(seen)]
    print(f"\nAlready in BigQuery : {len(seen)} file(s)")
    print(f"To load             : {new['source_batch'].nunique()} file(s), {len(new):,} row(s)")

    if not new.empty:
        out = new.copy()
        out["pull_date"] = pd.to_datetime(out["pull_date"]).dt.date
        job = client.load_table_from_dataframe(
            out, table_id,
            job_config=bigquery.LoadJobConfig(
                schema=_schema(), write_disposition=bigquery.WriteDisposition.WRITE_APPEND),
        )
        job.result()
        print(f"Loaded {len(out):,} row(s); {table_id} now holds "
              f"{client.get_table(table_id).num_rows:,} row(s).")
    else:
        print("Nothing new to load.")

    # series_metadata is a projection of config.py, rewritten WRITE_TRUNCATE, so this is
    # idempotent -- and it is what puts the gt_yape_plin row in the warehouse even when
    # the BCRP loader found nothing new and returned before writing it.
    from src.data_collection.load_to_bigquery import load_series_metadata

    print(f"series_metadata refreshed: {load_series_metadata(client)} row(s).")


if __name__ == "__main__":
    main()
