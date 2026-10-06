"""Diagnose what the BCRP statistics API is doing right now.

    python scripts/bcrp_probe.py

Standalone on purpose: imports nothing from src/, so it runs from any directory
and cannot be broken by the code it is diagnosing.

It sends ~12 requests (one per case below, 1 s apart -- polite to a free public
API), and for each records status, content type, size, redirects, timing,
whether the body parses as JSON, and whether that JSON still has the shape our
parser expects. Every raw body is saved, so the evidence can be read directly
rather than described.

    logs/bcrp_probe/<timestamp>/summary.txt     human-readable report
    logs/bcrp_probe/<timestamp>/summary.json    same, machine-readable
    logs/bcrp_probe/<timestamp>/<case>.body     raw response bodies

logs/ is gitignored, so none of this pollutes the repo.

WHAT THIS DELIBERATELY DOES NOT DO. If the host has put an anti-bot wall in
front of the API, this script reports that and stops there. It sends an honest
User-Agent that names the project; it does not impersonate a browser or try to
defeat a challenge page. A public institution blocking automated access is a
decision to respect, and the alternatives are in docs/data_sources.md.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import requests

HOST = "https://estadisticas.bcrp.gob.pe"
API = f"{HOST}/estadisticas/series/api"

# The production request shape, exactly as bcrp_client.fetch_series builds it.
PAY = "PN42200EM"    # n_transf_intra_agg -- the main target's parent series
MACRO = "PD04722MM"  # tasa_referencia -- a different table, for contrast

HONEST_UA = (
    "pagos-telefono-26/1.0 (academic research ETL; monthly pull of public BCRP "
    "series; python-requests)"
)

CASES = [
    # name                 url                                                  ua
    ("01_production",      f"{API}/{PAY}/json/2026-1/2026-12/esp",               None),
    ("02_honest_ua",       f"{API}/{PAY}/json/2026-1/2026-12/esp",               HONEST_UA),
    ("03_macro_series",    f"{API}/{MACRO}/json/2026-1/2026-12/esp",             HONEST_UA),
    ("04_lang_ing",        f"{API}/{PAY}/json/2026-1/2026-12/ing",               HONEST_UA),
    ("05_no_lang",         f"{API}/{PAY}/json/2026-1/2026-12",                   HONEST_UA),
    ("06_zero_padded",     f"{API}/{PAY}/json/2026-01/2026-12/esp",              HONEST_UA),
    ("07_no_dates",        f"{API}/{PAY}/json",                                  HONEST_UA),
    ("08_fmt_csv",         f"{API}/{PAY}/csv/2026-1/2026-12/esp",                HONEST_UA),
    ("09_fmt_xml",         f"{API}/{PAY}/xml/2026-1/2026-12/esp",                HONEST_UA),
    ("10_two_codes",       f"{API}/{PAY}-{MACRO}/json/2026-1/2026-12/esp",       HONEST_UA),
    ("11_docs_page",       f"{HOST}/estadisticas/series/ayuda/api",              HONEST_UA),
    ("12_html_view",       f"{HOST}/estadisticas/series/mensuales/resultados/{MACRO}/html",
                                                                                 HONEST_UA),
]

OUT = Path(__file__).resolve().parent.parent / "logs" / "bcrp_probe" / datetime.now().strftime("%Y%m%d_%H%M%S")


def inspect_json(body: bytes) -> dict:
    """Does the body still have the shape parse.py depends on?"""
    try:
        data = json.loads(body.decode("utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        return {"json": False, "json_error": f"{type(e).__name__}: {str(e)[:120]}"}

    info = {"json": True, "top_keys": sorted(data.keys()) if isinstance(data, dict) else type(data).__name__}
    if isinstance(data, dict):
        series = (data.get("config") or {}).get("series")
        periods = data.get("periods")
        info["has_config_series"] = isinstance(series, list)
        info["n_series"] = len(series) if isinstance(series, list) else None
        info["series_names"] = [s.get("name", "?")[:70] for s in series] if isinstance(series, list) else None
        info["has_periods"] = isinstance(periods, list)
        info["n_periods"] = len(periods) if isinstance(periods, list) else None
        if isinstance(periods, list) and periods:
            first, last = periods[0], periods[-1]
            info["first_period"] = first
            info["last_period"] = last
            info["period_keys"] = sorted(first.keys()) if isinstance(first, dict) else None
    return info


def classify(r: dict) -> str:
    """One-line diagnosis per case."""
    if r.get("exception"):
        e = r["exception"]
        if "SSL" in e:
            return "TLS FAILURE -- certificate chain or interception"
        if "Timeout" in e:
            return "TIMEOUT -- host slow or dropping requests"
        if "Connection" in e:
            return "CONNECTION FAILURE -- DNS, firewall, or host down"
        return f"EXCEPTION -- {e[:60]}"
    s, ct, j = r["status"], (r["content_type"] or "").lower(), r.get("inspect", {})
    if s == 429:
        return "RATE LIMITED (429)"
    if s in (401, 403):
        return f"REFUSED ({s}) -- blocked; check body for a WAF/bot page"
    if s == 404:
        return "NOT FOUND (404) -- URL scheme may have changed"
    if s >= 500:
        return f"SERVER ERROR ({s})"
    # Body first, header second. BCRP serves perfectly valid JSON labelled `text/html`, so a
    # Content-Type check up front reports a healthy API as an anti-bot page.
    if j.get("json"):
        served_as = f"  [served as {ct}]" if "html" in ct else ""
        if j.get("has_config_series") and j.get("has_periods"):
            return (f"OK -- expected schema, {j['n_periods']} periods, "
                    f"last={j.get('last_period', {}).get('name')}{served_as}")
        return f"JSON BUT NEW SHAPE -- keys={j.get('top_keys')}{served_as}"
    if s == 200 and "html" in ct and r["name"] not in ("11_docs_page", "12_html_view"):
        if r["size"] < 2000:
            return "HTML STUB on an API URL -- classic anti-bot challenge page"
        return "HTML on an API URL -- endpoint now serves a web page"
    if s == 200:
        return f"200, non-JSON ({ct or 'no content-type'}, {r['size']} B)"
    return f"HTTP {s}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"requests {requests.__version__}   writing to {OUT}\n")

    results = []
    for name, url, ua in CASES:
        headers = {"User-Agent": ua} if ua else {}
        rec = {"name": name, "url": url, "ua": "honest" if ua else "requests-default"}
        t0 = time.time()
        try:
            resp = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            body = resp.content
            rec.update(
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type"),
                size=len(body),
                elapsed_s=round(time.time() - t0, 2),
                final_url=resp.url,
                redirects=[h.status_code for h in resp.history],
                server=resp.headers.get("Server"),
                interesting_headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() in ("x-cdn", "x-iinfo", "set-cookie", "cf-ray",
                                     "x-akamai-transformed", "retry-after", "via",
                                     "x-powered-by", "strict-transport-security")
                },
                body_head=body[:400].decode("utf-8", "replace"),
            )
            (OUT / f"{name}.body").write_bytes(body)
            rec["inspect"] = inspect_json(body) if name not in ("11_docs_page", "12_html_view") else {}
        except Exception as e:  # noqa: BLE001
            rec.update(exception=f"{type(e).__name__}: {e}", elapsed_s=round(time.time() - t0, 2))
        rec["diagnosis"] = classify(rec)
        results.append(rec)
        print(f"  {name:18s} {str(rec.get('status', '---')):>4}  {rec['diagnosis']}")
        time.sleep(1.0)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"BCRP API probe  {datetime.now():%Y-%m-%d %H:%M}", "=" * 72, ""]
    for r in results:
        lines += [
            f"[{r['name']}]  {r['diagnosis']}",
            f"  url          {r['url']}",
            f"  ua           {r['ua']}",
        ]
        if "exception" in r:
            lines.append(f"  exception    {r['exception']}")
        else:
            lines += [
                f"  status       {r['status']}   redirects={r['redirects']}   {r['elapsed_s']}s",
                f"  final_url    {r['final_url']}",
                f"  type/size    {r['content_type']}   {r['size']} B   server={r['server']}",
            ]
            if r["interesting_headers"]:
                lines.append(f"  headers      {r['interesting_headers']}")
            if r.get("inspect"):
                lines.append(f"  json         {json.dumps(r['inspect'], ensure_ascii=False)[:400]}")
            lines.append(f"  body[:400]   {r['body_head'][:400]!r}")
        lines.append("")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nFull report: {OUT / 'summary.txt'}")


if __name__ == "__main__":
    main()
