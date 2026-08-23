"""Pull every BCRP series defined in config.py and save one raw snapshot each.

Run from the project root:
    python -m src.data_collection.fetch_target_series
"""

import time

import requests

from src.data_collection.bcrp_client import fetch_series, save_snapshot
from src.data_collection.config import ALL_SERIES

START = "2024-1"  # earliest history for the Yape/Plin target series
END = "2026-12"   # end of the requested window; the API returns what exists
SLEEP_SECONDS = 0.3  # be polite to a free public API


def main() -> None:
    ok, failed = 0, []

    for series in ALL_SERIES:
        code = series["code"]
        try:
            envelope = fetch_series(code, START, END)
            path = save_snapshot(envelope)
            n_periods = len(envelope["response"].get("periods", []))
            print(f"  ✓ {code:11s} {series['col_name']:26s} {n_periods:3d} periods  -> {path.name}")
            ok += 1
        except (requests.RequestException, ValueError) as e:
            print(f"  ✗ {code:11s} {series['col_name']:26s} FAILED: {e}")
            failed.append(code)
        time.sleep(SLEEP_SECONDS)

    print(f"\n{ok}/{len(ALL_SERIES)} series saved to data/raw/bcrp/")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
