#!/usr/bin/env python3
"""
Qwen OpenAI API Proxy Server

This application creates a FastAPI server that provides OpenAI API compatibility
while using the qwen-api library as the backend.
"""

import argparse
import asyncio
import base64
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import coloredlogs
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import qwen-api library
from qwen_api.client import Qwen
from qwen_api.core.exceptions import AuthError, QwenAPIError, RateLimitError
from qwen_api.core.types.chat import (
    ChatMessage,
    ChatResponse,
    ChatResponseStream,
    ImageBlock,
    TextBlock,
)

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)


class OpenAIMessageContentItem(BaseModel):
    """Model for OpenAI message content item."""
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, Any]] = None


class OpenAIMessage(BaseModel):
    """Model for OpenAI message."""
    role: str
    content: Union[str, List[OpenAIMessageContentItem]]


class OpenAIToolFunction(BaseModel):
    """Model for OpenAI tool function."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class OpenAITool(BaseModel):
    """Model for OpenAI tool."""
    type: str
    function: OpenAIToolFunction


class OpenAIChatRequest(BaseModel):
    """Model for OpenAI chat request."""
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 2048
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    web_search: Optional[bool] = False
    thinking: Optional[bool] = False
    web_development: Optional[bool] = False


class OpenAIChoiceDelta(BaseModel):
    """Model for OpenAI choice delta."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None


class OpenAIChoice(BaseModel):
    """Model for OpenAI choice."""
    index: int
    delta: Optional[OpenAIChoiceDelta] = None
    message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class OpenAIUsage(BaseModel):
    """Model for OpenAI usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatResponse(BaseModel):
    """Model for OpenAI chat response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Optional[OpenAIUsage] = None


class QwenOpenAIProxy:
    """Main class for the Qwen OpenAI API Proxy."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8123):
        """
        Initialize the Qwen OpenAI Proxy.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.qwen_client = Qwen()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="OpenAI-Compatible API (Qwen Backend)",
            description="A FastAPI server providing OpenAI API compatibility, using the `qwen-api` library as the backend.",
            version="1.0.0",
        )

        @app.get("/v1/models")
        async def list_models():
            """List available models that conform to OpenAI API spec."""
            current_timestamp = int(datetime.now().timestamp())

            # Define the models available in Qwen
            # These are the typical models available in Qwen's interface
            models = [
                {
                    "id": "qwen-max-latest",
                    "object": "model",
                    "created": current_timestamp - 86400,  # Yesterday
                    "owned_by": "qwen",
                    "permission": []
                },
                {
                    "id": "qwen-max",
                    "object": "model",
                    "created": current_timestamp - 172800,  # 2 days ago
                    "owned_by": "qwen",
                    "permission": []
                },
                {
                    "id": "qwen-plus",
                    "object": "model",
                    "created": current_timestamp - 259200,  # 3 days ago
                    "owned_by": "qwen",
                    "permission": []
                },
                {
                    "id": "qwen-turbo",
                    "object": "model",
                    "created": current_timestamp - 345600,  # 4 days ago
                    "owned_by": "qwen",
                    "permission": []
                }
            ]

            return {
                "object": "list",
                "data": models
            }

        @app.get("/v1/models/{model_id}")
        async def retrieve_model(model_id: str):
            """Retrieve a specific model that conforms to OpenAI API spec."""
            current_timestamp = int(datetime.now().timestamp())

            # Define the model details
            model_map = {
                "qwen-max-latest": {
                    "id": "qwen-max-latest",
                    "object": "model",
                    "created": current_timestamp - 86400,
                    "owned_by": "qwen",
                    "permission": []
                },
                "qwen-max": {
                    "id": "qwen-max",
                    "object": "model",
                    "created": current_timestamp - 172800,
                    "owned_by": "qwen",
                    "permission": []
                },
                "qwen-plus": {
                    "id": "qwen-plus",
                    "object": "model",
                    "created": current_timestamp - 259200,
                    "owned_by": "qwen",
                    "permission": []
                },
                "qwen-turbo": {
                    "id": "qwen-turbo",
                    "object": "model",
                    "created": current_timestamp - 345600,
                    "owned_by": "qwen",
                    "permission": []
                }
            }

            if model_id in model_map:
                return model_map[model_id]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_id}' not found"
                )

        @app.post("/v1/chat/completions")
        async def chat_completions_endpoint(
                request: Request, openai_request: OpenAIChatRequest
        ):
            """OpenAI-compatible /v1/chat/completions endpoint."""
            return await self._handle_chat_completions(request, openai_request)

        @app.get("/health")
        async def health_check():
            """Simple health check endpoint."""
            return {"status": "healthy"}

        return app

    def _convert_openai_message_to_qwen_message(
            self,
            openai_message: OpenAIMessage,
            web_search: bool,
            thinking: bool,
            web_development: bool,
    ) -> ChatMessage:
        """
        Convert an OpenAI message format to a qwen-api ChatMessage.

        Args:
            openai_message: The OpenAI message to convert
            web_search: Whether to enable web search
            thinking: Whether to enable thinking mode
            web_development: Whether to enable web development mode

        Returns:
            Converted ChatMessage object
        """
        content_str = ""
        blocks = []

        if isinstance(openai_message.content, str):
            content_str = openai_message.content
        elif isinstance(openai_message.content, list):
            for item in openai_message.content:
                if item.type == "text":
                    text = item.text or ""
                    content_str += text + " "
                    blocks.append(TextBlock(block_type="text", text=text))
                elif item.type == "image_url":
                    image_url_data = item.image_url or {}
                    image_url = image_url_data.get("url", "")

                    if image_url.startswith("data:"):
                        try:
                            header, base64_data = image_url.split(",", 1)
                            mime_type = (
                                header.split(":", 1)[1].split(";", 1)[0]
                                if ":" in header and ";" in header
                                else "image/jpeg"
                            )
                            file_result = self.qwen_client.chat.upload_file(
                                base64_data=base64_data
                            )
                            blocks.append(
                                ImageBlock(
                                    block_type="image",
                                    url=file_result.file_url,
                                    image_mimetype=file_result.image_mimetype,
                                )
                            )
                        except Exception as e:
                            raise HTTPException(
                                status_code=500, detail=f"Image upload failed: {str(e)}"
                            )
                    else:
                        blocks.append(
                            ImageBlock(
                                block_type="image",
                                url=image_url,
                                image_mimetype="image/jpeg",
                            )
                        )

        if isinstance(openai_message.content, list) and not blocks and content_str.strip():
            blocks.append(TextBlock(block_type="text", text=content_str.strip()))

        return ChatMessage(
            role=openai_message.role,
            content=content_str.strip(),
            web_search=web_search,
            thinking=thinking,
            web_development=web_development,
            blocks=blocks if blocks else None,
        )

    def _convert_qwen_response_to_openai_response(
            self,
            qwen_response: Union[ChatResponse, AsyncGenerator[ChatResponseStream, None]],
            model_name: str,
            request_id: str,
            created_timestamp: int,
    ) -> OpenAIChatResponse:
        """
        Convert a qwen-api non-streaming response to an OpenAI-compatible response.

        Args:
            qwen_response: The Qwen API response
            model_name: The model name
            request_id: The request ID
            created_timestamp: The creation timestamp

        Returns:
            Converted OpenAIChatResponse object
        """
        content = ""
        tool_calls = None

        if hasattr(qwen_response, "choices"):
            choice_obj = qwen_response.choices

            if hasattr(choice_obj, "message") and hasattr(choice_obj.message, "content"):
                content = choice_obj.message.content

            if (
                    hasattr(choice_obj, "message")
                    and hasattr(choice_obj.message, "tool_calls")
                    and choice_obj.message.tool_calls
            ):
                tool_calls = []
                for tc in choice_obj.message.tool_calls:
                    tool_call = {
                        "id": getattr(tc, "id", f"call_{uuid.uuid4().hex}"),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": (
                                getattr(tc.function, "name", "")
                                if hasattr(tc, "function")
                                else getattr(tc, "name", "")
                            ),
                            "arguments": (
                                getattr(tc.function, "arguments", "")
                                if hasattr(tc, "function")
                                else getattr(tc, "arguments", "")
                            ),
                        },
                    }
                    tool_calls.append(tool_call)

        openai_choice = OpenAIChoice(
            index=0,
            message={
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            finish_reason="tool_calls" if tool_calls else "stop",
        )

        usage = OpenAIUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        return OpenAIChatResponse(
            id=request_id,
            created=created_timestamp,
            model=model_name,
            choices=[openai_choice],
            usage=usage,
        )

    def _convert_qwen_chunk_to_openai_chunk(
            self,
            qwen_chunk: Any,
            model_name: str,
            request_id: str,
            created_timestamp: int,
            is_first_chunk: bool = False
    ) -> str:
        """
        Convert a qwen-api streaming chunk to an OpenAI-compatible SSE chunk.

        Args:
            qwen_chunk: The Qwen API streaming chunk
            model_name: The model name
            request_id: The request ID
            created_timestamp: The creation timestamp
            is_first_chunk: Whether this is the first chunk in the stream

        Returns:
            Formatted SSE chunk string
        """
        delta_content = ""
        delta_tool_calls = None
        finish_reason = None

        if hasattr(qwen_chunk, "choices") and len(qwen_chunk.choices) > 0:
            delta = qwen_chunk.choices[0].delta

            if hasattr(delta, "content") and delta.content:
                delta_content = delta.content

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                delta_tool_calls = []
                for tc in delta.tool_calls:
                    tool_call_delta = {
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": (
                                getattr(tc.function, "name", None)
                                if hasattr(tc, "function")
                                else getattr(tc, "name", None)
                            ),
                            "arguments": (
                                getattr(tc.function, "arguments", "")
                                if hasattr(tc, "function")
                                else getattr(tc, "arguments", "")
                            ),
                        },
                    }
                    delta_tool_calls.append(tool_call_delta)

            if hasattr(qwen_chunk.choices[0], "finish_reason"):
                finish_reason = qwen_chunk.choices[0].finish_reason

        delta_obj = {
            "content": delta_content,
            "tool_calls": delta_tool_calls,
        }

        # Add role only on the first chunk
        if is_first_chunk:
            delta_obj["role"] = "assistant"

        openai_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created_timestamp,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": delta_obj,
                    "finish_reason": finish_reason,
                }
            ],
        }

        return f"data: {json.dumps(openai_chunk)}\n\n"

    def _reset_chunk_state(self) -> None:
        """Reset the chunk state for streaming responses."""
        if hasattr(self._convert_qwen_chunk_to_openai_chunk, "role_sent"):
            delattr(self._convert_qwen_chunk_to_openai_chunk, "role_sent")

    async def _handle_chat_completions(
            self, request: Request, openai_request: OpenAIChatRequest
    ):
        """
        Handle the chat completions request.

        Args:
            request: The FastAPI request object
            openai_request: The OpenAI chat request

        Returns:
            Either a StreamingResponse or JSONResponse
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(datetime.now().timestamp())

        try:
            qwen_messages = [
                self._convert_openai_message_to_qwen_message(
                    msg,
                    web_search=openai_request.web_search,
                    thinking=openai_request.thinking,
                    web_development=openai_request.web_development,
                )
                for msg in openai_request.messages
            ]

            model_name = openai_request.model
            temperature = openai_request.temperature
            max_tokens = openai_request.max_tokens
            if max_tokens > 8192:
                logger.warning(f"max_tokens is over 8192: {max_tokens}, will clamp to 8192")
                max_tokens = 8192
            stream = openai_request.stream

            tools = None
            if openai_request.tools:
                tools = [tool.model_dump() for tool in openai_request.tools]

            if stream:
                self._reset_chunk_state()

                async def stream_generator():
                    is_first_chunk = True  # Track first chunk state here
                    try:
                        stream_response = await self.qwen_client.chat.acreate(
                            messages=qwen_messages,
                            model=model_name,
                            stream=True,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            tools=tools,
                        )

                        async for qwen_chunk in stream_response:
                            openai_chunk_str = self._convert_qwen_chunk_to_openai_chunk(
                                qwen_chunk,
                                model_name,
                                request_id,
                                created_timestamp,
                                is_first_chunk=is_first_chunk
                            )
                            is_first_chunk = False  # Set to False after first chunk
                            yield openai_chunk_str

                        yield "data: [DONE]\n\n"

                    except (AuthError, RateLimitError, QwenAPIError) as e:
                        error_detail = f"Qwen API Error: {str(e)}"
                        error_chunk = {
                            "error": {
                                "message": error_detail,
                                "type": type(e).__name__,
                                "param": None,
                                "code": None,
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        error_detail = f"Internal Error: {str(e)}"
                        error_chunk = {
                            "error": {
                                "message": error_detail,
                                "type": "internal_error",
                                "param": None,
                                "code": None,
                            }
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_generator(), media_type="text/event-stream"
                )

            else:
                try:
                    qwen_response = await self.qwen_client.chat.acreate(
                        messages=qwen_messages,
                        model=model_name,
                        stream=False,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                    )

                    openai_response = self._convert_qwen_response_to_openai_response(
                        qwen_response, model_name, request_id, created_timestamp
                    )
                    return JSONResponse(content=openai_response.model_dump())

                except (AuthError, RateLimitError, QwenAPIError) as e:
                    raise HTTPException(
                        status_code=500, detail=f"Qwen API Error: {str(e)}"
                    )
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Internal Error: {str(e)}"
                    )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Request Processing Error: {str(e)}"
            )

    def run(self) -> None:
        """Run the FastAPI application."""
        import uvicorn

        logger.info(f"Starting Qwen OpenAI Proxy server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Qwen OpenAI API Proxy Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8123,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging level
    coloredlogs.install(level=args.log_level, logger=logger)

    # Create and run the proxy server
    proxy = QwenOpenAIProxy(host=args.host, port=args.port)
    proxy.run()


if __name__ == "__main__":
    main()