"""Pull every BCRP series defined in config.py and save one raw snapshot each.

Run from the project root:
    python -m src.data_collection.fetch_target_series

Each category gets its own start date (config.START_BY_CATEGORY): the
Yape/Plin targets only exist from 2024, while macro and CCE series carry
decades of history that a single shared start date would throw away.

Snapshots are immutable and named {pull_date}_{code}.json, so re-running on a
later date adds files rather than overwriting them. The BigQuery loader
appends the new pull; de-duplication to "latest value per series/month"
happens in dbt's staging layer.
"""

import time

import requests

from src.data_collection.bcrp_client import fetch_series, save_snapshot
from src.data_collection.config import BCRP_SERIES, START_BY_CATEGORY

END = "2026-12"      # end of the requested window; the API returns what exists
SLEEP_SECONDS = 0.3  # be polite to a free public API


def main() -> None:
    ok, failed = 0, []

    for series in BCRP_SERIES:
        code = series["code"]
        start = START_BY_CATEGORY[series["category"]]
        try:
            envelope = fetch_series(code, start, END)
            path = save_snapshot(envelope)
            n_periods = len(envelope["response"].get("periods", []))
            print(f"  ✓ {code:11s} {series['col_name']:26s} {start:>8s}  {n_periods:4d} periods  -> {path.name}")
            ok += 1
        except (requests.RequestException, ValueError) as e:
            print(f"  ✗ {code:11s} {series['col_name']:26s} {start:>8s}  FAILED: {e}")
            failed.append(code)
        time.sleep(SLEEP_SECONDS)

    print(f"\n{ok}/{len(BCRP_SERIES)} series saved to data/raw/bcrp/")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
