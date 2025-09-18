# tests/test_proxy.py
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List

import pytest


@pytest.fixture(autouse=True)
def clear_proxy_cache() -> None:
    """Ensure cache is empty before and after each test."""
    proxy.CACHE.clear()
    yield
    proxy.CACHE.clear()


def test_make_cache_key() -> None:
    k1 = proxy.make_cache_key("endpoint", {"a": 1})
    k2 = proxy.make_cache_key("endpoint", {"a": 1})
    k3 = proxy.make_cache_key("endpoint", {"a": 2})
    assert k1 == k2
    assert k1 != k3


def test_set_get_clear_cache() -> None:
    key = "testkey"
    proxy.set_cache(key, {"value": 123})
    got = proxy.get_cache(key)
    assert got == {"value": 123}
    # force expiry and verify get_cache returns None and removes entry
    proxy.CACHE[key]["expires_at"] = time.time() - 1
    assert proxy.get_cache(key) is None
    # verify clear_cache
    proxy.set_cache(key, {"value": 456})
    proxy.clear_cache()
    assert proxy.CACHE == {}


@pytest.mark.asyncio
async def test__proxy_json(monkeypatch) -> None:
    """_proxy_json should return a JSONResponse and set the cache on 200."""

    class FakeResponse:
        def __init__(self, json_data: Dict[str, Any], status_code: int = 200) -> None:
            self._json = json_data
            self.status_code = status_code

        def json(self) -> Dict[str, Any]:
            return self._json

    class FakeClient:
        def __init__(self, timeout: Any = None) -> None:
            del timeout

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        async def post(self, url: str, json: Dict[str, Any]) -> FakeResponse:
            del url, json
            return FakeResponse({"hello": "world"}, 200)

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    cache_key = "ck"
    response = await proxy._proxy_json("http://example", {"a": 1}, cache_key)
    body = json.loads(response.body)
    assert response.status_code == 200
    assert body == {"hello": "world"}
    assert cache_key in proxy.CACHE
    assert proxy.CACHE[cache_key]["value"] == {"hello": "world"}


@pytest.mark.asyncio
async def test__proxy_stream(monkeypatch) -> None:
    """_proxy_stream should stream formatted lines and set cache to list of formatted chunks."""
    lines = ["line1", "line2"]

    class FakeStreamContext:
        def __init__(self, lines_in: List[str]) -> None:
            self._lines = lines_in

        async def __aenter__(self) -> "FakeStreamContext":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        async def aiter_lines(self):
            for l in self._lines:
                yield l
                await asyncio.sleep(0)

    class FakeClient:
        def __init__(self, timeout: Any = None) -> None:
            del timeout

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            return None

        def stream(self, method: str, url: str, json: Dict[str, Any]) -> FakeStreamContext:
            del method, url, json
            return FakeStreamContext(lines)

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    cache_key = "stream_key"
    response = await proxy._proxy_stream("http://upstream", {"a": 1}, cache_key)

    collected: List[str] = []
    async for chunk in response.body_iterator:
        collected.append(chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk))

    expected = [l + "\n\n" for l in lines]
    assert collected == expected
    assert proxy.CACHE[cache_key]["value"] == expected


@pytest.mark.asyncio
async def test_proxy_request_cached_non_stream_and_stream() -> None:
    """proxy_request should return cached JSONResponse for non-stream and StreamingResponse for stream."""
    endpoint = "chat/completions"
    payload = {"foo": "bar"}
    cache_key = proxy.make_cache_key(endpoint, payload)

    proxy.set_cache(cache_key, {"cached": True})
    resp = await proxy.proxy_request(endpoint, payload, streaming=False)
    assert isinstance(resp, proxy.JSONResponse)
    assert json.loads(resp.body) == {"cached": True}

    proxy.set_cache(cache_key, ["chunk1\n\n", "chunk2\n\n"])
    resp2 = await proxy.proxy_request(endpoint, payload, streaming=True)
    collected: List[str] = []
    async for chunk in resp2.body_iterator:
        collected.append(chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk))
    assert collected == ["chunk1\n\n", "chunk2\n\n"]


@pytest.mark.asyncio
async def test_proxy_request_condensation(monkeypatch) -> None:
    """Simulate first _proxy_json failure then success, ensuring condense is called."""
    endpoint = "models"
    payload = {"q": "a", "model": "m", "messages": []}
    call_count = {"n": 0}

    async def fake_proxy_json(url: str, payload_arg: Dict[str, Any], cache_key: str):
        if call_count["n"] == 0:
            call_count["n"] += 1
            raise RuntimeError("simulated upstream failure")
        return proxy.JSONResponse(content={"ok": True}, status_code=200)

    monkeypatch.setattr(proxy, "_proxy_json", fake_proxy_json)

    called = {"condensed": False}

    def fake_condense(messages, model, func, target_tokens, cfg):
        called["condensed"] = True
        return {"messages": messages, "model": model, "max_tokens": 10}

    monkeypatch.setattr(proxy, "condense", fake_condense)

    resp = await proxy.proxy_request(endpoint, payload, streaming=False, enable_condensation=True)
    assert isinstance(resp, proxy.JSONResponse)
    assert resp.status_code == 200
    assert json.loads(resp.body) == {"ok": True}
    assert called["condensed"] is True


@pytest.mark.asyncio
async def test_proxy_request_no_condensation_returns_500(monkeypatch) -> None:
    async def fake_proxy_json(url: str, payload: Dict[str, Any], cache_key: str):
        raise RuntimeError("fail")

    monkeypatch.setattr(proxy, "_proxy_json", fake_proxy_json)

    resp = await proxy.proxy_request("models", {"a": 1}, streaming=False, enable_condensation=False)
    assert isinstance(resp, proxy.JSONResponse)
    assert resp.status_code == 500
    assert json.loads(resp.body)["error"] == "Unable to process request"


import pytest
from httpx import AsyncClient, ASGITransport
import proxy


@pytest.mark.asyncio
async def test_fastapi_endpoints_and_cache_clear(monkeypatch) -> None:
    async def fake_proxy_request(
            endpoint: str,
            payload: dict,
            streaming: bool,
            enable_condensation: bool = True,
    ):
        return proxy.JSONResponse(
            content={"endpoint": endpoint, "streaming": streaming},
            status_code=200,
        )

    monkeypatch.setattr(proxy, "proxy_request", fake_proxy_request)

    called = {"clear_called": False}

    def fake_clear_cache() -> None:
        called["clear_called"] = True

    monkeypatch.setattr(proxy, "clear_cache", fake_clear_cache)

    transport = ASGITransport(app=proxy.app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={"foo": "bar"})
        assert resp.json() == {"endpoint": "chat/completions", "streaming": False}

        resp = await client.post("/v1/completions", json={"foo": "bar", "stream": True})
        assert resp.json() == {"endpoint": "completions", "streaming": True}

        resp = await client.post("/v1/responses", json={"baz": 123})
        assert resp.json() == {"endpoint": "responses", "streaming": False}

        resp = await client.get("/models")
        assert resp.json() == {"endpoint": "models", "streaming": False}

        health = await client.get("/health")
        assert health.status_code == 200
        assert "status" in health.json()

        resp = await client.post("/cache/clear")
        assert resp.status_code == 200
        assert called["clear_called"] is True


@pytest.mark.asyncio
async def test_proxy_request_no_condensation_returns_500_streaming(monkeypatch) -> None:
    """When condensation is disabled and the upstream fails for a streaming request, return 500."""

    async def fake_proxy_stream(url: str, payload: Dict[str, Any], cache_key: str):
        raise RuntimeError("fail")

    monkeypatch.setattr(proxy, "_proxy_stream", fake_proxy_stream)
    resp = await proxy.proxy_request("models", {"a": 1}, streaming=True, enable_condensation=False)
    assert isinstance(resp, proxy.JSONResponse)
    assert resp.status_code == 500
    assert json.loads(resp.body)["error"] == "Unable to process request"


@pytest.mark.asyncio
async def test_health_check_direct() -> None:
    result = await proxy.health_check()
    assert result["status"] == "ok"
    assert result["cache_size"] == 0
    proxy.set_cache("a", 1)
    proxy.set_cache("b", 2)
    result2 = await proxy.health_check()
    assert result2["cache_size"] == 2
