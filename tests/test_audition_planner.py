"""Tests for audition_planner.py's HTTP retry/timeout hardening (no server)."""

from __future__ import annotations

import json

import pytest

import audition_planner as ap


class FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(ap.time, "sleep", lambda _seconds: None)


def test_call_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ap.urllib.error.URLError("temporarily down")
        return FakeResponse({"ok": True})

    monkeypatch.setattr(ap.urllib.request, "urlopen", fake_urlopen)
    assert ap.call("http://x", "/prompt", {"a": 1}, retries=3, backoff=0) == {"ok": True}
    assert attempts["n"] == 3


def test_call_raises_server_error_after_retries(monkeypatch):
    def always_fail(request, timeout=None):
        raise ap.urllib.error.URLError("connection refused")

    monkeypatch.setattr(ap.urllib.request, "urlopen", always_fail)
    with pytest.raises(ap.ServerError, match="failed after 2 attempts"):
        ap.call("http://x", "/history/x", retries=2, backoff=0)


def test_wait_for_job_times_out(monkeypatch):
    # monotonic is called once to arm the deadline, then once in the loop.
    ticks = iter([0.0, 100.0])
    monkeypatch.setattr(ap.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(ap, "call", lambda *a, **k: {})
    with pytest.raises(ap.ServerError, match="no result after"):
        ap.wait_for_job("http://x", "pid", "lora.safetensors", timeout=50,
                        poll_interval=0, http_timeout=1, retries=1)


def test_wait_for_job_reports_success_and_error(monkeypatch):
    success = {"pid": {
        "status": {"status_str": "success"},
        "outputs": {"16": {"audio": [{"filename": "out.flac"}]}},
    }}
    monkeypatch.setattr(ap, "call", lambda *a, **k: success)
    assert ap.wait_for_job("http://x", "pid", "lora", timeout=10,
                           poll_interval=0, http_timeout=1, retries=1) is True

    failure = {"pid": {"status": {"status_str": "error", "messages": [["x", "boom"]]}}}
    monkeypatch.setattr(ap, "call", lambda *a, **k: failure)
    assert ap.wait_for_job("http://x", "pid", "lora", timeout=10,
                           poll_interval=0, http_timeout=1, retries=1) is False
