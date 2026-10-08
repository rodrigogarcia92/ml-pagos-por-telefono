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

import sys
import time
from datetime import date

import requests

from src.data_collection.bcrp_client import (
    BcrpError,
    BcrpNonJsonResponse,
    fetch_series,
    save_snapshot,
)
from src.data_collection.config import BCRP_SERIES, START_BY_CATEGORY

# End of the requested window; the API returns what exists. The current year's December, so a
# fixed year cannot silently truncate every pull from January onward (monthly refresh).
END = f"{date.today().year}-12"
SLEEP_SECONDS = 0.3  # be polite to a free public API
# A bot-protection page answers EVERY request the same way. After this many series in a row come
# back as non-JSON, stop instead of spending minutes of backoff on the rest (plan O-13).
MAX_CONSECUTIVE_NON_JSON = 2


def main() -> int:
    """Returns 0 when every series was saved, 1 otherwise (a partial pull is not a pull)."""
    ok, failed, blocked_in_a_row = 0, [], 0

    for series in BCRP_SERIES:
        code = series["code"]
        start = START_BY_CATEGORY[series["category"]]
        try:
            envelope = fetch_series(code, start, END)
            path = save_snapshot(envelope)
            n_periods = len(envelope["response"].get("periods", []))
            print(f"  ✓ {code:11s} {series['col_name']:26s} {start:>8s}  {n_periods:4d} periods  -> {path.name}")
            ok += 1
            blocked_in_a_row = 0
        except (BcrpError, requests.RequestException, ValueError) as e:
            print(f"  ✗ {code:11s} {series['col_name']:26s} {start:>8s}  FAILED: {e}")
            failed.append(code)
            blocked_in_a_row = blocked_in_a_row + 1 if isinstance(e, BcrpNonJsonResponse) else 0
            if blocked_in_a_row >= MAX_CONSECUTIVE_NON_JSON:
                print(f"\nBCRP answered {blocked_in_a_row} requests in a row with a non-JSON body "
                      "(bot protection?). Stopping. Retry later or download by hand; do not try to "
                      "get around it (plan O-13).")
                break
        time.sleep(SLEEP_SECONDS)

    print(f"\n{ok}/{len(BCRP_SERIES)} series saved to data/raw/bcrp/")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
