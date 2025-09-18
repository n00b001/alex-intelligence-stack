#!/usr/bin/env python3
"""
Production-ready FastAPI proxy for OpenAI-compatible LLM APIs.
Features:
- Passthrough with caching and TTL
- Streaming and non-streaming support
- Cache clearing endpoint
- Configurable via env vars
- Context length protection with condensation fallback (placeholder)
- Structured logging with coloredlogs
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, List, AsyncGenerator

import coloredlogs
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from condenser import condense
from consts import Config

config = Config()

# ------------------------
# Logging
# ------------------------

logger = logging.getLogger("proxy")
coloredlogs.install(level=config.log_level, logger=logger)

# ------------------------
# Cache
# ------------------------

CacheEntry = Dict[str, Any]
CACHE: Dict[str, CacheEntry] = {}


def make_cache_key(endpoint: str, payload: Dict[str, Any]) -> str:
    key_data = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True)
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()


def get_cache(cache_key: str) -> Optional[Any]:
    entry = CACHE.get(cache_key)
    if entry and entry["expires_at"] > time.time():
        logger.debug(f"Cache hit for key {cache_key}")
        return entry["value"]
    elif entry:
        logger.debug(f"Cache expired for key {cache_key}")
        del CACHE[cache_key]
    return None


def set_cache(cache_key: str, value: Any) -> None:
    logger.debug(f"Setting cache for key {cache_key}")
    CACHE[cache_key] = {"value": value, "expires_at": time.time() + config.cache_ttl}


def clear_cache() -> None:
    logger.warning("Clearing entire cache")
    CACHE.clear()


# ------------------------
# Proxy logic
# ------------------------

async def proxy_request(
        endpoint: str,
        payload: Dict[str, Any],
        streaming: bool,
        enable_condensation: bool = True,
) -> JSONResponse | StreamingResponse:
    """
    Proxy a request to the upstream API with cache and fallback on condensation.
    """
    url = f"{config.base_url}/{endpoint}"
    cache_key = make_cache_key(endpoint, payload)

    cached = get_cache(cache_key)
    if cached:
        if streaming:
            async def replay() -> AsyncGenerator[str, None]:
                for chunk in cached:
                    yield chunk

            return StreamingResponse(replay(), media_type="text/event-stream")
        return JSONResponse(content=cached, status_code=200)

    attempt = 0
    target_tokens = int(config.context_length - payload.get("max_tokens", config.max_new_tokens))
    while attempt <= config.condense_retry_attempts:
        try:
            if streaming:
                return await _proxy_stream(url, payload, cache_key)
            return await _proxy_json(url, payload, cache_key)
        except Exception as e:
            if enable_condensation:
                logger.error(f"Request failed: {e}. Attempt {attempt + 1}/{config.truncation_retries}")
                payload = condense(
                    payload.get("messages", None),
                    payload.get("model"),
                    proxy_request,
                    target_tokens,
                    config,
                )
                attempt += 1
                target_tokens = int(target_tokens * config.safety_ratio)
            else:
                break

    logger.critical("All condensation retries failed")
    return JSONResponse(content={"error": "Unable to process request"}, status_code=500)


async def _proxy_json(url: str, payload: Dict[str, Any], cache_key: str) -> JSONResponse:
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=payload)
        resp_json = r.json()
        if r.status_code == 200:
            set_cache(cache_key, resp_json)
        return JSONResponse(content=resp_json, status_code=r.status_code)


async def _proxy_stream(url: str, payload: Dict[str, Any], cache_key: str) -> StreamingResponse:
    stream_chunks: List[str] = []

    async def event_generator() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                async for line in r.aiter_lines():
                    if line:
                        formatted = line + "\n\n"
                        stream_chunks.append(formatted)
                        yield formatted
        set_cache(cache_key, stream_chunks)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ------------------------
# FastAPI app
# ------------------------

app = FastAPI(title="LLM Proxy", version="1.0.0")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await proxy_request("chat/completions", body, streaming=body.get("stream", False))


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    return await proxy_request("completions", body, streaming=body.get("stream", False))


@app.post("/v1/responses")
async def responses(request: Request):
    body = await request.json()
    return await proxy_request("responses", body, streaming=body.get("stream", False))


@app.get("/models")
async def list_models():
    return await proxy_request("models", {}, streaming=False)


@app.get("/health")
async def health_check():
    return {"status": "ok", "cache_size": len(CACHE)}


@app.post("/cache/clear")
async def clear_cache_endpoint():
    clear_cache()
    return {"status": "cache_cleared"}


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=config.port, reload=True)
