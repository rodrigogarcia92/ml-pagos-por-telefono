"""Backlog 8.3 -- the probe classifies by the BODY first, the Content-Type header second."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bcrp_probe", Path("scripts/bcrp_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

GOOD = {"config": {"series": [{"name": "n"}]}, "periods": [{"name": "Jul.2026", "values": ["1"]}]}


def _rec(body: bytes, ctype: str | None, status: int = 200, name: str = "01_production") -> dict:
    return {"name": name, "status": status, "content_type": ctype, "size": len(body),
            "inspect": probe.inspect_json(body) if name not in ("11_docs_page", "12_html_view") else {}}


def test_valid_json_labelled_text_html_is_ok_not_an_anti_bot_stub():
    body = json.dumps(GOOD).encode()
    out = probe.classify(_rec(body, "text/html; charset=utf-8"))
    assert out.startswith("OK -- expected schema, 1 periods, last=Jul.2026")
    assert "served as text/html" in out and "STUB" not in out


def test_valid_json_with_the_right_header_is_plain_ok():
    out = probe.classify(_rec(json.dumps(GOOD).encode(), "application/json"))
    assert out.startswith("OK -- expected schema") and "served as" not in out


def test_json_with_a_bom_is_still_json():
    body = b"\xef\xbb\xbf" + json.dumps(GOOD).encode("utf-8")
    assert probe.classify(_rec(body, "text/html")).startswith("OK")


def test_json_in_a_new_shape_is_reported_as_such_whatever_the_header():
    out = probe.classify(_rec(json.dumps({"data": []}).encode(), "text/html"))
    assert out.startswith("JSON BUT NEW SHAPE") and "data" in out


def test_a_small_html_body_on_an_api_url_is_the_anti_bot_stub():
    body = b"<html><script>var challenge = 1;</script></html>"
    assert "HTML STUB" in probe.classify(_rec(body, "text/html"))


def test_a_large_html_body_is_a_web_page_not_a_stub():
    body = b"<html>" + b"x" * 5000 + b"</html>"
    assert probe.classify(_rec(body, "text/html")).startswith("HTML on an API URL")


def test_a_non_json_body_with_a_json_header_is_not_ok():
    out = probe.classify(_rec(b"not json at all", "application/json"))
    assert out.startswith("200, non-JSON") and "OK" not in out


def test_html_pages_that_are_expected_to_be_html_are_not_flagged():
    body = b"<html>" + b"x" * 5000 + b"</html>"
    assert probe.classify(_rec(body, "text/html", name="11_docs_page")).startswith("200, non-JSON")


@pytest.mark.parametrize("status,needle", [(429, "RATE LIMITED"), (403, "REFUSED"),
                                           (404, "NOT FOUND"), (503, "SERVER ERROR")])
def test_status_based_diagnoses_are_unchanged(status, needle):
    assert needle in probe.classify(_rec(b"", "text/html", status=status))


def test_transport_exceptions_are_unchanged():
    assert "TIMEOUT" in probe.classify({"exception": "ReadTimeout: slow"})
    assert "CONNECTION FAILURE" in probe.classify({"exception": "ConnectionError: dns"})
