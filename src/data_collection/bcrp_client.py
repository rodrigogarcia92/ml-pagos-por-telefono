"""Client for the BCRP (Banco Central de Reserva del Perú) statistics API.

Docs: https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api
No authentication required.

URL shape (confirmed against live calls):
    https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codes}/json/{start}/{end}/{lang}
    codes  -> hyphen-separated series codes, e.g. "PN42672EM-PN42673EM"
    start/end -> monthly period as "YYYY-M" (e.g. "2024-1" = Jan 2024)
    lang   -> "esp" (default) or "ing"

IMPORTANT — why we request ONE code per call:
    The API accepts up to 10 codes per request, but it does NOT return the
    series in the order they were requested. A 10-code call was verified to
    come back in a different order entirely (position 0 held the last-listed
    macro series). Because the response identifies each series only by a long
    `name` string — never by its code — there is no reliable way to map values
    back to codes in a batched response.

    Requesting one code at a time makes the mapping correct by construction:
    the single series in the response is unambiguously the code we asked for.
    The cost is one HTTP request per series instead of one per ten, which at
    this project's scale (~20 series) is a few seconds.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import requests

BASE_URL = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

# project_root/src/data_collection/bcrp_client.py -> project_root/data/raw/bcrp
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "bcrp"

# Retries are for TRANSIENT failures only: a dropped connection or a body that is not JSON (an
# error page, a half-loaded response). 2 s, 4 s, 8 s, then give up and say what came back.
# Nothing here tries to get past bot protection -- no header spoofing, no challenge solving
# (plan O-13 forbids it); if the host serves a challenge page, retries will not help
# and the error names it so a person can decide.
BACKOFF_SECONDS = (2, 4, 8)
BODY_HEAD_CHARS = 200
_sleep = time.sleep          # indirection, so tests need not wait


class BcrpError(RuntimeError):
    """Base class for BCRP client failures that are not a bug in this code."""


class BcrpNonJsonResponse(BcrpError):
    """The API answered, but the body is not JSON -- typically an HTML bot-protection page.

    BCRP has served valid JSON labelled `text/html`, so this is decided by trying to parse the
    body, never by the Content-Type header.
    """

    def __init__(self, url: str, status: int, content_type: str | None, body_head: str):
        self.url, self.status, self.content_type, self.body_head = url, status, content_type, body_head
        super().__init__(
            f"BCRP returned a non-JSON body for {url}: HTTP {status}, "
            f"content-type {content_type!r}, body starts: {body_head!r}")


class BcrpNetworkError(BcrpError):
    """The request never produced a response (DNS, connection, timeout) after all retries."""


def _get_json(url: str) -> dict:
    """GET and parse, retrying transient failures. Raises BcrpError subclasses, nothing else
    from the transport."""
    last: Exception | None = None
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        if attempt:
            _sleep(BACKOFF_SECONDS[attempt - 1])
        try:
            response = requests.get(url, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = BcrpNetworkError(f"BCRP request failed for {url}: {type(e).__name__}: {e}")
            last.__cause__ = e
            continue
        try:
            payload = response.json()
        except ValueError:          # JSONDecodeError and requests' own subclass
            last = BcrpNonJsonResponse(url, response.status_code,
                                       response.headers.get("Content-Type"),
                                       response.text[:BODY_HEAD_CHARS])
            continue
        response.raise_for_status()
        return payload
    assert last is not None
    raise last


def fetch_series(code: str, start: str, end: str, lang: str = "esp") -> dict:
    """Fetch a single BCRP series.

    Returns an envelope, not the bare API response:

        {"code": ..., "api_name": ..., "url": ..., "response": {<verbatim JSON>}}

    `api_name` is the API's own label for the series, captured here so the
    raw.series_metadata table (docs/data_sources.md §5.3) can record what the
    provider actually calls each code.
    """
    url = f"{BASE_URL}/{code}/json/{start}/{end}/{lang}"
    payload = _get_json(url)

    series = payload.get("config", {}).get("series", [])
    if len(series) != 1:
        raise ValueError(
            f"{code}: expected exactly 1 series in the response, got {len(series)}. "
            "The code may be invalid or discontinued."
        )

    return {
        "code": code,
        "api_name": series[0]["name"],
        "url": url,
        "response": payload,
    }


def save_snapshot(envelope: dict, pull_date: date | None = None) -> Path:
    """Save a fetch_series envelope to data/raw/bcrp/ as an immutable snapshot.

    Filename: {pull_date}_{code}.json — one file per series, so `source_batch`
    in raw.bcrp_observations points straight back to the file that produced it.

    The verbatim API response is preserved under "response"; our provenance
    fields sit alongside it rather than being mixed into it.
    """
    pull_date = pull_date or date.today()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{pull_date.isoformat()}_{envelope['code']}.json"
    record = {**envelope, "pulled_at": pull_date.isoformat()}
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
