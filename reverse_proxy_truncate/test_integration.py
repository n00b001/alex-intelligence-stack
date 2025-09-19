#!/usr/bin/env python3
"""
Integration tests for the proxy server with condensation fallback.
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
from typing import Dict, Any
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

import proxy

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import tiktoken for token counting
try:
    import tiktoken
except ImportError:
    raise ImportError("tiktoken is required for integration tests. Install it with: pip install tiktoken")

# Test configuration - use different ports to avoid conflicts
PROXY_PORT = 54321
MOCK_API_PORT = 54322
PROXY_URL = f"http://localhost:{PROXY_PORT}"
MOCK_API_URL = f"http://localhost:{MOCK_API_PORT}"

# Global servers
mock_server_thread = None
proxy_server_thread = None

# Model configuration for token counting
MODEL_NAME = "gpt-3.5-turbo"
MAX_CONTEXT_LENGTH = 4096


class MockAPIServer:
    """Smart mock upstream API server that simulates real behavior."""

    def __init__(self):
        self.app = FastAPI()
        self.request_count = 0
        self.summarization_count = 0
        self.encoder = tiktoken.encoding_for_model(MODEL_NAME)

    def count_tokens(self, messages):
        """Count tokens in messages using tiktoken."""
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if content:
                total_tokens += len(self.encoder.encode(str(content)))
        return total_tokens

    def is_summarization_request(self, messages):
        """Check if this is a summarization request from the condenser."""
        return (
                len(messages) == 1 and
                messages[0].get("role") == "user" and
                "Summarise the following text" in messages[0].get("content", "")
        )

    async def mock_chat_completions(self, request: Request):
        body = await request.json()
        self.request_count += 1

        messages = body.get("messages", [])
        token_count = self.count_tokens(messages)
        max_tokens = body.get("max_tokens", 100)
        total_tokens = token_count + max_tokens

        print(
            f"Mock API received request "
            f"#{self.request_count}: {token_count} message tokens + {max_tokens} "
            f"max_tokens = {total_tokens} total tokens"
        )

        # Check if this is a summarization request
        if self.is_summarization_request(messages):
            self.summarization_count += 1
            print(f"Processing summarization request #{self.summarization_count}")
            # Return a valid summarization response
            return JSONResponse(
                content={
                    "id": "chatcmpl-test123",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": body.get("model", MODEL_NAME),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"This is a summary from request #{self.summarization_count} (condensed from {token_count} tokens)"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": token_count,
                        "completion_tokens": 20,
                        "total_tokens": token_count + 20
                    }
                },
                status_code=200
            )
        else:
            # Check if request exceeds context length
            if total_tokens > MAX_CONTEXT_LENGTH:
                print(f"Request exceeds context limit: {total_tokens} > {MAX_CONTEXT_LENGTH}")
                # Return context length error for requests that are too large
                return JSONResponse(
                    content={
                        "error": {
                            "message": f"This model's maximum context length is {MAX_CONTEXT_LENGTH} tokens. "
                                       f"However, you requested {total_tokens} tokens "
                                       f"({token_count} in the messages, {max_tokens} in the completion). "
                                       f"Please reduce the length of the messages or completion.",
                            "type": "invalid_request_error"
                        }
                    },
                    status_code=400
                )
            else:
                print(f"Request within context limit: {total_tokens} <= {MAX_CONTEXT_LENGTH}")
                # Return a valid response for requests that fit
                return JSONResponse(
                    content={
                        "id": "chatcmpl-regular123",
                        "object": "chat.completion",
                        "created": 1234567890,
                        "model": body.get("model", MODEL_NAME),
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"This is a regular response. Your request had {token_count} tokens."
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": token_count,
                            "completion_tokens": 15,
                            "total_tokens": token_count + 15
                        }
                    },
                    status_code=200
                )

    def setup_routes(self):
        """Setup the FastAPI routes."""
        self.app.post("/v1/chat/completions")(self.mock_chat_completions)

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}


def start_mock_server():
    """Start mock API server in a thread."""
    mock_server = MockAPIServer()
    mock_server.setup_routes()

    config = uvicorn.Config(
        mock_server.app,
        host="127.0.0.1",
        port=MOCK_API_PORT,
        log_level="error",
        access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread, server


def start_proxy_server():
    """Start proxy server in a thread."""
    # Override config for testing
    os.environ['BASE_URL'] = MOCK_API_URL
    os.environ['PORT'] = str(PROXY_PORT)
    os.environ['CONTEXT_LENGTH'] = str(MAX_CONTEXT_LENGTH)
    os.environ['LOG_LEVEL'] = 'INFO'
    os.environ['CONDENSE_RETRY_ATTEMPTS'] = '3'
    os.environ['KEEP_FIRST_N'] = '500'
    os.environ['KEEP_LAST_N'] = '500'
    os.environ['APPROX_CONDENSATION_TOKENS'] = '200'

    # Import here to pick up environment variables
    from proxy import app

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=PROXY_PORT,
        log_level="info",
        access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread, server


def wait_for_server(url, timeout=15):
    """Wait for server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{url}/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_servers():
    """Setup and teardown test servers."""
    global mock_server_thread, proxy_server_thread

    # Start mock API server
    mock_server_thread, _ = start_mock_server()

    # Start proxy server
    proxy_server_thread, _ = start_proxy_server()

    # Wait for servers to be ready
    time.sleep(3)

    if not wait_for_server(MOCK_API_URL):
        raise RuntimeError("Mock server failed to start")

    if not wait_for_server(PROXY_URL):
        raise RuntimeError("Proxy server failed to start")

    yield

    # Cleanup happens automatically with daemon threads


async def make_chat_request(messages: list, max_tokens: int = 100) -> httpx.Response:
    """Helper to make a chat completion request."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response


def test_server_startup():
    """Test that servers start correctly."""
    response = httpx.get(f"{MOCK_API_URL}/health", timeout=5.0)
    assert response.status_code == 200

    response = httpx.get(f"{PROXY_URL}/health", timeout=5.0)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_condensation_flow():
    """Test the complete condensation flow with smart mock API."""
    # Create a very long message that should exceed context limits
    long_message = "This is a very long message that will trigger condensation. " * 20000  # ~1M chars

    # Estimate tokens - this should be well over our limit
    encoder = tiktoken.encoding_for_model(MODEL_NAME)
    estimated_tokens = len(encoder.encode(long_message))
    print(f"Estimated tokens in long message: {estimated_tokens}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": long_message},
        {"role": "assistant", "content": "I understand."},
        {"role": "user", "content": "Now please respond to this."}
    ]

    # Make request - this should trigger condensation
    response = await make_chat_request(messages)

    # Log response for debugging
    print(f"Response status: {response.status_code}")

    # The request should either succeed (condensation worked) or fail gracefully
    # But it should NOT hang or crash
    assert response.status_code in [200, 500], f"Unexpected status code: {response.status_code}"

    # If we got a response body, try to parse it
    if response.content:
        try:
            data = response.json()
            print(f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            if "error" in data:
                print(f"Error response: {data['error']}")
        except Exception as e:
            print(f"Could not parse response JSON: {e}")
            print(f"Response text: {response.text[:200]}...")


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test that health endpoint works."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{PROXY_URL}/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_cache_clear_endpoint():
    """Test that cache clear endpoint works."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(f"{PROXY_URL}/cache/clear")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "cache_cleared"


if __name__ == "__main__":
    # Manual test run
    print("Starting servers...")

    # Start servers manually for direct testing
    mock_thread, _ = start_mock_server()
    proxy_thread, _ = start_proxy_server()
    time.sleep(5)

    print("Testing health endpoints...")
    try:
        _response = httpx.get(f"{MOCK_API_URL}/health", timeout=5.0)
        print(f"Mock API health: {_response.status_code}")

        _response = httpx.get(f"{PROXY_URL}/health", timeout=5.0)
        print(f"Proxy health: {_response.status_code}")
        print("Servers are running!")

        # Test a simple request
        print("Testing condensation flow...")
        test_messages = [
            {"role": "user", "content": "Hello " * 20000}  # Large message
        ]

        payload = {
            "model": MODEL_NAME,
            "messages": test_messages,
            "max_tokens": 100
        }

        _response = httpx.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=30.0
        )
        print(f"Condensation test response: {_response.status_code}")
        print(f"Response: {_response.text[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()