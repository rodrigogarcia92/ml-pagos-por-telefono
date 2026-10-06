"""Backlog 8.2 -- BCRP client: named errors, bounded retries, no evasion. Offline (requests mocked)."""

from __future__ import annotations

import json

import pytest
import requests

from src.data_collection import bcrp_client, fetch_target_series
from src.data_collection.bcrp_client import (
    BcrpError,
    BcrpNetworkError,
    BcrpNonJsonResponse,
    fetch_series,
)

PAYLOAD = {"config": {"series": [{"name": "Transferencias intrabancarias"}]},
           "periods": [{"name": "Jan.2026", "values": ["1.0"]}]}


class Resp:
    def __init__(self, body: str, status: int = 200, ctype: str | None = "text/html; charset=utf-8"):
        self.text, self.status_code = body, status
        self.headers = {"Content-Type": ctype} if ctype else {}

    def json(self):
        return json.loads(self.text)          # JSONDecodeError is a ValueError

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


@pytest.fixture
def net(monkeypatch):
    """Script the sequence of responses/exceptions; record calls and sleeps."""
    state = {"calls": [], "sleeps": [], "script": []}

    def get(url, **kw):
        state["calls"].append((url, kw))
        item = state["script"].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(bcrp_client.requests, "get", get)
    monkeypatch.setattr(bcrp_client, "_sleep", state["sleeps"].append)
    return state


CHALLENGE = Resp("<html><script>" + "x" * 500 + "</script></html>", 200, "text/html")


def test_valid_json_labelled_text_html_is_accepted(net):
    net["script"] = [Resp(json.dumps(PAYLOAD), 200, "text/html")]
    env = fetch_series("PN42200EM", "2026-1", "2026-12")
    assert env["api_name"] == "Transferencias intrabancarias" and net["sleeps"] == []


def test_a_non_json_body_is_retried_three_times_with_backoff_then_named(net):
    net["script"] = [CHALLENGE] * 4
    with pytest.raises(BcrpNonJsonResponse) as ei:
        fetch_series("PN42200EM", "2026-1", "2026-12")
    e = ei.value
    assert len(net["calls"]) == 4 and net["sleeps"] == [2, 4, 8]
    assert e.status == 200 and e.content_type == "text/html"
    assert len(e.body_head) == 200 and e.body_head.startswith("<html><script>")
    msg = str(e)
    assert "HTTP 200" in msg and "text/html" in msg and "<html><script>" in msg
    assert isinstance(e, BcrpError) and "PN42200EM" in e.url


def test_a_transient_non_json_body_recovers_on_the_next_attempt(net):
    net["script"] = [CHALLENGE, Resp(json.dumps(PAYLOAD))]
    assert fetch_series("PN42200EM", "2026-1", "2026-12")["code"] == "PN42200EM"
    assert net["sleeps"] == [2] and len(net["calls"]) == 2


def test_network_errors_are_retried_and_then_named(net):
    net["script"] = [requests.ConnectionError("dns")] * 4
    with pytest.raises(BcrpNetworkError, match="ConnectionError"):
        fetch_series("PN42200EM", "2026-1", "2026-12")
    assert net["sleeps"] == [2, 4, 8]

    net["calls"].clear()
    net["sleeps"].clear()
    net["script"] = [requests.Timeout("slow"), requests.ConnectionError("blip"),
                     Resp(json.dumps(PAYLOAD))]
    assert fetch_series("PN42200EM", "2026-1", "2026-12")["code"] == "PN42200EM"
    assert net["sleeps"] == [2, 4]


def test_an_http_error_with_a_json_body_is_not_retried(net):
    net["script"] = [Resp(json.dumps({"error": "x"}), 500, "application/json")]
    with pytest.raises(requests.HTTPError):
        fetch_series("PN42200EM", "2026-1", "2026-12")
    assert len(net["calls"]) == 1 and net["sleeps"] == []


def test_a_wrong_shape_still_raises_the_original_value_error(net):
    net["script"] = [Resp(json.dumps({"config": {"series": []}}))]
    with pytest.raises(ValueError, match="expected exactly 1 series"):
        fetch_series("PN42200EM", "2026-1", "2026-12")


def test_the_client_sends_nothing_designed_to_look_like_a_browser(net):
    net["script"] = [Resp(json.dumps(PAYLOAD))]
    fetch_series("PN42200EM", "2026-1", "2026-12")
    _, kw = net["calls"][0]
    assert set(kw) == {"timeout"}           # no headers, no cookies, no session tricks
    assert bcrp_client.BACKOFF_SECONDS == (2, 4, 8)


# --------------------------------------------------------------------------- #
# fetch_target_series: stop early on a wall, and exit non-zero on a partial pull
# --------------------------------------------------------------------------- #
def test_the_fetch_stops_after_two_non_json_series_and_exits_1(monkeypatch, capsys):
    calls = []

    def blocked(code, start, end, lang="esp"):
        calls.append(code)
        raise BcrpNonJsonResponse("u", 200, "text/html", "<html>")

    monkeypatch.setattr(fetch_target_series, "fetch_series", blocked)
    monkeypatch.setattr(fetch_target_series.time, "sleep", lambda s: None)
    assert fetch_target_series.main() == 1
    out = capsys.readouterr().out
    assert len(calls) == fetch_target_series.MAX_CONSECUTIVE_NON_JSON
    assert "non-JSON" in out and "O-13" in out


def test_the_fetch_exits_0_only_when_every_series_was_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_target_series, "fetch_series",
                        lambda code, s, e, lang="esp": {"code": code, "response": PAYLOAD})
    monkeypatch.setattr(fetch_target_series, "save_snapshot", lambda env: tmp_path / f"{env['code']}.json")
    monkeypatch.setattr(fetch_target_series.time, "sleep", lambda s: None)
    assert fetch_target_series.main() == 0
