#!/usr/bin/env python3
"""
condenser.py

Provides `condense` to reduce LLM chat histories by summarisation.

Key points:
- tiktoken is required and used with explicit Encoding typing.
- `target_tokens` is mandatory.
- f-strings used for logging.
- Head+tail combined size considered and summarised if their sum exceeds target.
- Non-text messages (images/audio/tools) are preserved and skipped for token counting.
- Splitting hierarchy: messages -> paragraphs -> lines -> sentences -> commas -> words -> chars.
- Tenacity retry decorators used for summariser calls.

API:
    async def condense(
        chat_history: Any,
        model: str,
        summarizer: SummarizerCallback,
        target_tokens: int,
        cfg: Optional[Config] = None,
    ) -> Any

The summarizer callable must follow the proxy_request signature:
    async def proxy_request(endpoint: str, payload: dict, streaming: bool, enable_condensation: bool) -> JSONResponse|StreamingResponse|dict
"""

from __future__ import annotations

import json
import logging
from itertools import batched
from typing import Any, Awaitable, Callable, Dict, Union
from typing import List, Optional

import coloredlogs
from fastapi.responses import JSONResponse, StreamingResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from consts import Config

# tiktoken is required for this program
try:
    import tiktoken
    from tiktoken import Encoding
except Exception as exc:  # pragma: no cover - must be installed
    raise ImportError(
        "tiktoken is required for condenser.py. Install it with: pip install tiktoken"
    ) from exc

# ------------------------
# Config + logging
# ------------------------

cfg_default = Config()
logger = logging.getLogger("condenser")
coloredlogs.install(level=cfg_default.log_level, logger=logger)

_ENCODING_FALLBACK = "cl100k_base"

ProxyRequestSignature = Callable[
    [str, Dict[str, Any], bool, bool],  # endpoint, payload, streaming, enable_condensation
    Awaitable[Union[JSONResponse, StreamingResponse, Dict[str, Any]]],
]


# ------------------------
# Tokeniser helpers
# ------------------------


def _get_encoder_for_model(model: Optional[str]) -> Optional[Encoding]:
    """Return a tiktoken.Encoding for the model or fallback encoding."""
    try:
        return tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding(_ENCODING_FALLBACK)
    except Exception:
        return tiktoken.get_encoding(_ENCODING_FALLBACK)


def _num_tokens(text: object, encoder: Optional[Encoding]) -> int:
    """Estimate tokens for `text` using tiktoken encoder (required) with conservative fallback."""
    if not text:
        return 0
    if encoder is not None:
        try:
            return len(encoder.encode(json.dumps(text)))
        except Exception:
            logger.debug("tiktoken encode failed; falling back to char heuristic")
    # Conservative fallback: 1 token per 4 chars
    return max(1, len(text) // 4)


# ------------------------
# Normalisation helpers
# ------------------------


def _is_textual(part: Any) -> bool:
    """Heuristic to determine if `part` is textual (not image/audio/file)."""
    if part is None:
        return False
    if isinstance(part, str):
        return True
    if isinstance(part, dict):
        low_keys = {k.lower() for k in part.keys()}
        typ = part.get("type") or part.get("mimeType") or part.get("mime")
        if isinstance(typ, str) and any(t in typ.lower() for t in ("image", "audio", "video", "file")):
            return False
        if any(k in low_keys for k in ("url", "uri", "filename", "file", "image_url", "audio_url")):
            return False
        for k in ("text", "content", "parts", "items"):
            if k in part:
                return _is_textual(part[k])
        # ambiguous -> treat as textual by default
        return True
    if isinstance(part, list):
        return any(_is_textual(p) for p in part)
    return False


def _extract_text_from_part(part: Any) -> str:
    """Extract textual content from nested structures. Return "" if none found."""
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if "text" in part and isinstance(part["text"], str):
            return part["text"]
        if "content" in part:
            return _extract_text_from_part(part["content"])
        if "parts" in part and isinstance(part["parts"], list):
            return "\n\n".join(_extract_text_from_part(p) for p in part["parts"])
        if "items" in part and isinstance(part["items"], list):
            return "\n\n".join(_extract_text_from_part(i) for i in part["items"])
        for key in ("message", "messages", "content", "text", "body"):
            if key in part:
                return _extract_text_from_part(part[key])
        try:
            return json.dumps(part)
        except Exception:
            return str(part)
    if isinstance(part, list):
        return "\n\n".join(_extract_text_from_part(p) for p in part)
    return str(part)


def _normalize_to_message_list(history: Any) -> List[Dict[str, Any]]:
    """Normalise various request shapes into a message list with metadata."""

    def normalize_message(role: str, content: Any) -> Dict[str, Any]:
        # Case 1: structured list (OpenAI multimodal)
        if isinstance(content, list) and all(isinstance(c, dict) and "type" in c for c in content):
            return {
                "role": role,
                "content": content,  # preserve structure
                "raw": None,
                "is_text": any(c.get("type") == "text" for c in content),
            }

        # Case 2: plain string
        if isinstance(content, str):
            return {"role": role, "content": content, "raw": None, "is_text": True}

        # Case 3: dict that looks like text
        if isinstance(content, dict):
            if "type" in content and content["type"] == "text":
                return {
                    "role": role,
                    "content": [content],
                    "raw": None,
                    "is_text": True,
                }
            # fallback: extract text if possible
            if "content" in content:
                return normalize_message(role, content["content"])
            if "text" in content:
                return normalize_message(role, content["text"])
            return {"role": role, "content": "", "raw": content, "is_text": False}

        # Case 4: fallback (numbers, etc.)
        return {"role": role, "content": str(content), "raw": None, "is_text": True}

    messages: List[Dict[str, Any]] = []
    if history is None:
        return messages

    # OpenAI chat style
    if isinstance(history, dict) and "messages" in history and isinstance(history["messages"], list):
        for m in history["messages"]:
            role = m.get("role", "user") if isinstance(m, dict) else "user"
            content = m.get("content") if isinstance(m, dict) else m
            messages.append(normalize_message(role, content))
        return messages

    # Responses / Completions style
    if isinstance(history, dict) and any(k in history for k in ("input", "prompt", "inputs")):
        key = next(k for k in ("input", "prompt", "inputs") if k in history)
        val = history[key]
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            messages.append(normalize_message("user", v))
        return messages

    # List form
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and ("role" in item or "content" in item):
                role = item.get("role", "user")
                content = item.get("content", item.get("text", item))
                messages.append(normalize_message(role, content))
            else:
                messages.append(normalize_message("user", item))
        return messages

    # Plain string
    if isinstance(history, str):
        return [normalize_message("user", history)]

    # Fallback
    return [normalize_message("user", history)]


# ------------------------
# Splitting / chunking helpers
# ------------------------


def _split_text_into_pieces(text: str, max_tokens: int, encoder: Optional[Encoding]) -> List[str]:
    """Split a long text into pieces each with at most max_tokens tokens.

    Uses tiktoken to tokenize, then itertools.batched to chunk tokens, then decode back to text.
    Guarantees no piece exceeds max_tokens.
    """
    if not text:
        return []

    tokens = encoder.encode(text)

    # Split tokens into chunks of at most `max_tokens`
    token_chunks = batched(tokens, max_tokens)

    # Decode each chunk back to text
    pieces = []
    for chunk in token_chunks:
        piece = encoder.decode(list(chunk))
        pieces.append(piece)

    return pieces


# ------------------------
# Tenacity retry decorator factory + summariser wrapper
# ------------------------


def _make_retry_decorator(attempts: int):
    return retry(stop=stop_after_attempt(max(1, attempts)), wait=wait_exponential(multiplier=1, max=60), reraise=True)


async def _call_summarizer_with_retries(
        proxy_request_callback: ProxyRequestSignature,
        prompt_text: str,
        cfg: Config,
        model: str
) -> str:
    """Call the summarizer callback with retries, always disabling condensation."""

    decorator = _make_retry_decorator(cfg.condense_retry_attempts)

    @decorator
    async def _invoke() -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": cfg.approx_condensation_tokens,
            "temperature": 0,
            "seed": 42,
            "top_p": 1,
        }

        response = await proxy_request_callback(
            "/v1/chat/completions",  # endpoint
            payload,
            streaming=False,
            enable_condensation=False
        )

        # Extract text from response
        if hasattr(response, "json"):  # JSONResponse
            data = await response.json()
        elif hasattr(response, "body_iterator"):  # StreamingResponse
            chunks = b""
            async for chunk in response.body_iterator:
                chunks += chunk
            try:
                data = json.loads(chunks)
            except Exception:
                data = chunks.decode(errors="ignore")
        else:
            data = response

        # Support different response formats
        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"])
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "message" in first:
                    return str(first["message"].get("content", ""))
        # fallback to str coercion
        return str(data)

    got_response = await _invoke()
    return got_response


async def _summarize_text_using_summarizer(
        proxy_request_callback: ProxyRequestSignature,
        text: str,
        cfg: Config,
        model: str,
) -> str:
    instruction = (
        "Summarise the following text concisely and factually.\n"
        "Return a short, self-contained summary suitable to stand in for the full text.\n"
        f"TEXT:\n\n{text}"
    )
    return await _call_summarizer_with_retries(proxy_request_callback, instruction, cfg, model)


# ------------------------
# Main condense algorithm
# ------------------------


async def condense(
        chat_history: Any,
        model: str,
        proxy_request_callback: ProxyRequestSignature,
        target_tokens: int,
        cfg: Optional[Config] = None,
) -> Any:
    """Condense chat_history down to approximately target_tokens.

    The proxy_request callable must follow the proxy_request signature and return an extractable object.
    """
    if cfg is None:
        cfg = cfg_default

    encoder = _get_encoder_for_model(model)

    logger.info(f"Starting condensation: target_tokens={target_tokens} model={model}")

    messages = _normalize_to_message_list(chat_history)

    # Count only textual tokens
    total_tokens = _num_tokens(chat_history, encoder)
    logger.debug(f"Original total tokens estimate={total_tokens} messages={len(messages)}")
    if total_tokens <= target_tokens:
        logger.warning("It seems no condensation required; will condense anyway...")

    # Build head and tail by token budget (keep_first_n / keep_last_n)
    remaining = messages[:]
    head: List[Dict[str, Any]] = []
    head_tokens = 0
    while remaining and head_tokens < cfg.keep_first_n:
        m = remaining.pop(0)
        head.append(m)
        if m.get("is_text", True):
            head_tokens += _num_tokens(m, encoder)

    tail: List[Dict[str, Any]] = []
    tail_tokens = 0
    while remaining and tail_tokens < cfg.keep_last_n:
        m = remaining.pop()
        tail.insert(0, m)
        if m.get("is_text", True):
            tail_tokens += _num_tokens(m, encoder)

    logger.debug(f"Kept head_tokens={head_tokens} tail_tokens={tail_tokens} middle_messages={len(remaining)}")

    # If combined head+tail exceed target, summarise head and tail independently
    combined_head_tail = head_tokens + tail_tokens
    if combined_head_tail > target_tokens:
        logger.warning(
            f"Head+tail tokens ({combined_head_tail}) exceed target ({target_tokens}); summarising head and tail separately")

        # Summarise head if present
        if head:
            head_text = "\n\n".join(m["content"] for m in head if m.get("is_text", True))
            if not head_text:
                head = [{"role": "assistant", "content": "[non-text head content]", "is_text": True, "raw": None}]
                head_tokens = _num_tokens(head[0]["content"], encoder)
            else:
                summary = await _summarize_text_using_summarizer(proxy_request_callback, head_text, cfg, model)
                head = [{"role": "assistant", "content": summary.strip(), "is_text": True, "raw": None}]
                head_tokens = _num_tokens(head[0]["content"], encoder)

        # Summarise tail if present
        if tail:
            tail_text = "\n\n".join(m["content"] for m in tail if m.get("is_text", True))
            if not tail_text:
                tail = [{"role": "assistant", "content": "[non-text tail content]", "is_text": True, "raw": None}]
                tail_tokens = _num_tokens(tail[0]["content"], encoder)
            else:
                summary = await _summarize_text_using_summarizer(proxy_request_callback, tail_text, cfg, model)
                tail = [{"role": "assistant", "content": summary.strip(), "is_text": True, "raw": None}]
                tail_tokens = _num_tokens(tail[0]["content"], encoder)

    # Recompute remaining middle texts
    middle_messages = remaining
    middle_texts = [m["content"] for m in middle_messages if m.get("is_text", True)]

    # If no middle left, return head+tail
    if not middle_texts:
        condensed = head + tail
        logger.info("No middle to condense; returning trimmed history")
        return _rebuild_history_from_messages(chat_history, condensed)

    # Setup attempt loop where we reduce chunk sizes if condensation doesn't reach target
    approx_tokens = cfg.approx_condensation_tokens
    for attempt in range(max(1, cfg.condense_construction_attempts)):
        logger.info(f"Condensation attempt {attempt + 1} with approx_chunk_tokens={approx_tokens}")

        # available budget for a chunk is the remaining context capacity
        available_context = max(256,
                                int(cfg.context_length * cfg.safety_ratio) - cfg.template_prompt_tokens - cfg.max_new_tokens)
        chunk_budget = max(64, min(approx_tokens, max(128, available_context - head_tokens - tail_tokens)))
        logger.debug(f"available_context={available_context} chunk_budget={chunk_budget}")

        # Build chunks by message boundaries first
        chunks: List[str] = []
        i = 0
        while i < len(middle_texts):
            # expand j as far as possible
            j = i
            current_tokens = 0
            while j < len(middle_texts):
                piece_tokens = _num_tokens(middle_texts[j], encoder)
                if current_tokens + piece_tokens > chunk_budget:
                    break
                current_tokens += piece_tokens
                j += 1

            if j == i:
                # single message too big for chunk_budget -> split message
                sub_pieces = _split_text_into_pieces(middle_texts[i], chunk_budget, encoder)
                for sp in sub_pieces:
                    chunks.append(sp)
                i += 1
            else:
                # messages i..j-1 form a chunk
                chunk = "\n\n".join(middle_texts[i:j])
                chunks.append(chunk)
                i = j

        logger.debug(f"Built {len(chunks)} chunks")

        # Summarise each chunk
        chunk_summaries: List[str] = []
        failed = False
        for idx, chunk in enumerate(chunks):
            try:
                summary = await _summarize_text_using_summarizer(
                    proxy_request_callback, chunk, cfg, model
                )
                chunk_summaries.append(summary.strip())
                logger.debug(
                    f"Chunk {idx} summarised (approx {_num_tokens(chunk, encoder)} tokens) -> {len(summary)} chars")
            except Exception as exc:  # pragma: no cover - runtime failures
                logger.exception(f"Summariser failed for chunk {idx}: {exc}")
                failed = True
                break

        if failed:
            approx_tokens = max(128, int(approx_tokens * cfg.truncation_factor))
            logger.warning(f"Reducing approx_chunk_tokens to {approx_tokens} and retrying")
            continue

        # Final summarise of concatenated chunk summaries
        joined_summaries = "\n\n".join(chunk_summaries)
        try:
            final_summary = await _summarize_text_using_summarizer(
                proxy_request_callback, joined_summaries, cfg, model
            )
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.exception(f"Final summariser step failed: {exc}")
            approx_tokens = max(128, int(approx_tokens * cfg.truncation_factor))
            continue

        summary_message = {"role": "assistant", "content": final_summary.strip(), "is_text": True, "raw": None}
        condensed_messages = head + [summary_message] + tail

        condensed_tokens = sum(_num_tokens(m["content"], encoder) for m in condensed_messages if m.get("is_text", True))
        logger.info(f"Condensed tokens estimate={condensed_tokens} target={target_tokens}")

        if condensed_tokens <= target_tokens:
            logger.info(f"Condensation succeeded on attempt {attempt + 1}")
            return _rebuild_history_from_messages(chat_history, condensed_messages)

        # reduce approx_tokens and retry
        approx_tokens = max(128, int(approx_tokens * cfg.truncation_factor))
        logger.warning(
            f"Condensed too large ({condensed_tokens} > {target_tokens}), reducing chunk size to {approx_tokens} and retrying")

    # Exhausted attempts -> fallback truncated message
    logger.error("Condensation exhausted retries; falling back to hard truncation")
    truncated_mid = cfg.truncation_message
    fallback_messages = head + [{"role": "assistant", "content": truncated_mid, "is_text": True, "raw": None}] + tail
    return _rebuild_history_from_messages(chat_history, fallback_messages)


# ------------------------
# Rebuild original shape
# ------------------------


def _rebuild_history_from_messages(original: Any, messages: List[Dict[str, Any]]) -> Any:
    """Attempt to rebuild history into a shape similar to original input.

    For messages that were originally non-text we restore the raw object where possible.
    """
    if original is None:
        return messages

    if isinstance(original, dict) and "messages" in original:
        rebuilt = []
        for m in messages:
            if m.get("raw") is not None:
                rebuilt.append(m["raw"])
            else:
                rebuilt.append({"role": m.get("role", "assistant"), "content": m.get("content", "")})
        return {**original, "messages": rebuilt}

    if isinstance(original, dict) and any(k in original for k in ("input", "prompt", "inputs")):
        key = next(k for k in ("input", "prompt", "inputs") if k in original)
        if key == "inputs":
            return {**original, key: [m["content"] for m in messages]}
        return {**original, key: "\n\n".join(m["content"] for m in messages)}

    if isinstance(original, str):
        return "\n\n".join(m["content"] for m in messages)

    # default
    rebuilt = []
    for m in messages:
        if m.get("raw") is not None:
            rebuilt.append(m["raw"])
        else:
            rebuilt.append({"role": m.get("role", "assistant"), "content": m.get("content", "")})
    return rebuilt


__all__ = ["condense"]
