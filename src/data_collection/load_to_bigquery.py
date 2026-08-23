"""Load local BCRP snapshots into BigQuery `raw`.

Run from the project root:
    python -m src.data_collection.load_to_bigquery
    python -m src.data_collection.load_to_bigquery --dry-run

Design notes
------------
* **Append-only.** Re-pulls insert new rows rather than overwriting, so the
  full pull history is preserved. dbt's staging layer de-duplicates down to
  "latest value per series/month" — BCRP revises recent months, so the newest
  pull wins there, not here.
* **Idempotent.** Before loading, the script asks BigQuery which
  `source_batch` filenames it already holds and skips those files. Running it
  twice in a row is a no-op, not a duplicate load.
* **No uniqueness constraint.** BigQuery does not enforce keys. Uniqueness is
  asserted by dbt tests on the staging model instead — versioned, visible, and
  re-checked on every build.
* **`value` is FLOAT64, not NUMERIC** (a deviation from docs/data_sources.md
  §5.3, which was written for PostgreSQL). These are measured economic
  quantities — counts in thousands, amounts in millions of soles — not money
  being summed to the cent, so exact decimal arithmetic buys nothing and
  FLOAT64 avoids a float -> Decimal conversion on every load.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from src.data_collection.config import ALL_SERIES
from src.data_collection.parse import load_all_snapshots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
DATASET = os.getenv("BQ_DATASET_RAW", "raw")

OBSERVATIONS_TABLE = f"{PROJECT_ID}.{DATASET}.bcrp_observations"
METADATA_TABLE = f"{PROJECT_ID}.{DATASET}.series_metadata"

OBSERVATIONS_SCHEMA = [
    bigquery.SchemaField("series_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("obs_date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("pulled_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("source_batch", "STRING", mode="REQUIRED"),
]

METADATA_SCHEMA = [
    bigquery.SchemaField("series_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("col_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("category", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("frequency", "STRING", mode="REQUIRED"),
]


def ensure_tables(client: bigquery.Client) -> None:
    """Create the raw tables if they don't exist. Safe to call every run."""
    try:
        client.get_table(OBSERVATIONS_TABLE)
    except NotFound:
        table = bigquery.Table(OBSERVATIONS_TABLE, schema=OBSERVATIONS_SCHEMA)
        # Neither of these matters for performance at ~600 rows. They are here
        # because partitioning and clustering are how this table would be built
        # at any real scale, and the pattern is worth having in the repo.
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH, field="obs_date"
        )
        table.clustering_fields = ["series_code"]
        client.create_table(table)
        print(f"  created {OBSERVATIONS_TABLE}")

    try:
        client.get_table(METADATA_TABLE)
    except NotFound:
        client.create_table(bigquery.Table(METADATA_TABLE, schema=METADATA_SCHEMA))
        print(f"  created {METADATA_TABLE}")


def already_loaded(client: bigquery.Client) -> set[str]:
    """Filenames already present in bcrp_observations."""
    rows = client.query(
        f"SELECT DISTINCT source_batch FROM `{OBSERVATIONS_TABLE}`"
    ).result()
    return {r.source_batch for r in rows}


def load_observations(client: bigquery.Client, df: pd.DataFrame) -> int:
    """Append observations. Returns the row count loaded."""
    out = df[["series_code", "obs_date", "value", "pulled_at", "source_batch"]].copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date

    job = client.load_table_from_dataframe(
        out,
        OBSERVATIONS_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=OBSERVATIONS_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ),
    )
    job.result()
    return len(out)


def load_series_metadata(client: bigquery.Client) -> int:
    """Rewrite series_metadata from config.ALL_SERIES.

    WRITE_TRUNCATE, not append: this table is a projection of config.py, so
    regenerating it from the source of truth makes drift impossible.
    """
    meta = pd.DataFrame(
        [
            {
                "series_code": s["code"],
                "col_name": s["col_name"],
                "description": s["description"],
                "category": s["category"],
                "frequency": s["frequency"],
            }
            for s in ALL_SERIES
        ]
    )

    job = client.load_table_from_dataframe(
        meta,
        METADATA_TABLE,
        job_config=bigquery.LoadJobConfig(
            schema=METADATA_SCHEMA,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    return len(meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report what would be loaded, without writing to BigQuery.",
    )
    args = parser.parse_args()

    print(f"Project : {PROJECT_ID}")
    print(f"Dataset : {DATASET}\n")

    df = load_all_snapshots()

    client = bigquery.Client(project=PROJECT_ID)
    dataset = client.get_dataset(f"{PROJECT_ID}.{DATASET}")
    print(f"\nDataset location: {dataset.location}")

    ensure_tables(client)

    seen = already_loaded(client)
    new = df[~df["source_batch"].isin(seen)]
    n_files = new["source_batch"].nunique()

    if seen:
        print(f"Already in BigQuery : {len(seen)} file(s)")
    print(f"To load             : {n_files} file(s), {len(new):,} row(s)")

    if new.empty:
        print("\nNothing new to load. BigQuery is up to date.")
        return

    if args.dry_run:
        print("\n--dry-run: no data written.")
        print(new.groupby("series_code").size().to_string())
        return

    n_obs = load_observations(client, new)
    n_meta = load_series_metadata(client)

    total = client.get_table(OBSERVATIONS_TABLE).num_rows
    print(f"\nLoaded {n_obs:,} observation row(s) and {n_meta} metadata row(s).")
    print(f"{OBSERVATIONS_TABLE} now holds {total:,} row(s).")


if __name__ == "__main__":
    main()
