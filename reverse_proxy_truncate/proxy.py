import json
import logging
import os
import traceback
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, AsyncGenerator, Callable, Dict

import coloredlogs
import httpx
import tiktoken
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

# ===============================================================================
# Logging & Configuration
# ===============================================================================

logger = logging.getLogger("proxy")


@dataclass
class Config:
    base_url: str = os.getenv("VLLM_URL", "http://hades-ubuntu.arpa:8124/v1")
    port: int = int(os.getenv("PORT", "8123"))
    context_length: int = int(os.getenv("CONTEXT", "90000"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "20000"))
    template_prompt_tokens: int = int(os.getenv("TEMPLATE_PROMPT_TOKENS", "1700"))
    keep_first_n: int = int(os.getenv("KEEP_FIRST_N", "10000"))
    keep_last_n: int = int(os.getenv("KEEP_LAST_N", "10000"))
    safety_ratio: float = float(os.getenv("SAFETY_RATIO", "0.999"))
    truncation_message: str = os.getenv(
        "TRUNCATION_MESSAGE", "\n\n"
    )
    truncation_retries: int = int(os.getenv("TRUNCATION_RETRIES", "10"))
    truncation_factor: float = float(os.getenv("TRUNCATION_FACTOR", "0.9"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    condense_retry_attempts: int = int(os.getenv("CONDENSE_RETRY_ATTEMPTS", "10"))
    approx_condensation_tokens: int = int(os.getenv("APPROX_CONDENSATION_TOKENS", "10000"))
    condense_construction_attempts: int = int(os.getenv("CONDENSE_CONSTRUCTION_ATTEMPTS", "10"))
    condense_truncate_retry_attempts: int = int(os.getenv("CONDENSE_TRUNCATE_RETRY_ATTEMPTS", "500"))


config = Config()
coloredlogs.install(level=config.log_level.upper(), logger=logger)

# Global AsyncOpenAI client for reuse
global_client = AsyncOpenAI(api_key="dummy", base_url=config.base_url)

# ===============================================================================
# Tokenization & Truncation Logic
# ===============================================================================

_tokenizer = tiktoken.get_encoding("o200k_base")


def _token_count(text_list: object) -> int:
    """Counts the number of tokens in the text."""
    text = json.dumps(text_list)
    return len(_tokenizer.encode(text))


#
#
# def _get_text_from_payload(payload: Dict[str, Any]) -> Tuple[str, Optional[str]]:
#     """Extracts text content and the key for updating from a payload."""
#     if "messages" in payload:
#         messages = payload.get("messages", [])
#         content_parts = []
#         for m in messages:
#             content = m.get("content")
#             if isinstance(content, str):
#                 content_parts.append(content)
#             elif isinstance(content, list):
#                 for part in content:
#                     if isinstance(part, dict) and part.get("type") == "text":
#                         content_parts.append(part.get("text", "").strip())
#         return " ".join(content_parts), "messages"
#     if "messages" in payload:
#         return payload.get("messages", ""), "messages"
#     return "", None

#
# def _update_payload_text(payload: Dict[str, Any], text: str) -> None:
#     """Updates the text in the payload based on the key."""
#     payload["messages"] = [{"role": "user", "content": text}]


# def truncate_payload(payload: Dict[str, Any], config: Config, truncation_factor: float) -> Dict[str, Any]:
#     """Truncates the input payload to fit within the model's context window."""
#     text, key_to_update = _get_text_from_payload(payload)
#     if not key_to_update:
#         return payload
#
#     tokens = _tokenizer.encode(text)
#     num_input_tokens = len(tokens)
#     max_new_tokens = payload.get("max_tokens", config.max_new_tokens)
#
#     available_for_output = int(
#         (config.context_length - num_input_tokens - config.template_prompt_tokens)
#         * config.safety_ratio * truncation_factor
#     )
#
#     if available_for_output < 0:
#         available_for_output = 0
#
#     if max_new_tokens > available_for_output:
#         logger.warning(
#             f"Capping max_tokens: {max_new_tokens} -> {available_for_output}"
#         )
#         payload["max_tokens"] = max_new_tokens
#         max_new_tokens = available_for_output
#
#     allowed_input_tokens = int(
#         (config.context_length - max_new_tokens - config.template_prompt_tokens)
#         * config.safety_ratio
#     )
#
#     if num_input_tokens > allowed_input_tokens:
#         logger.warning(
#             f"Truncating input: {num_input_tokens} -> {allowed_input_tokens} tokens"
#         )
#         k_first, k_last = config.keep_first_n, config.keep_last_n
#
#         if k_first + k_last > allowed_input_tokens:
#             ratio = (
#                 k_first / (k_first + k_last) if (k_first + k_last) > 0 else 0.5
#             )
#             k_first = int(allowed_input_tokens * ratio)
#             k_last = allowed_input_tokens - k_first
#
#         if k_first + k_last >= num_input_tokens:
#             truncated_tokens = tokens[-allowed_input_tokens:]
#             truncated_text = config.truncation_message + _tokenizer.decode(
#                 truncated_tokens
#             )
#         else:
#             truncated_text = (
#                     _tokenizer.decode(tokens[:k_first])
#                     + config.truncation_message
#                     + _tokenizer.decode(tokens[-k_last:])
#             )
#         _update_payload_text(payload, truncated_text)
#
#     return payload


# Condense prompt logic
CONDENSE_PROMPT = """
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like file names, full code snippets, function signatures, file edits, etc
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
6. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
7. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
8. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Your summary should avoid the following sections:

1. Directory listings of files longer than 10 files.
2. Incomplete summaries
3. Summaries without context

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

5. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

6. Current Work:
   [Precise description of current work]

7. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response. 

"""
CONDENSE_PROMPT_TOKEN_COUNT = _token_count(CONDENSE_PROMPT)


#
# def _get_messages_from_payload(payload):
#     return_messages = []
#     role = payload.get("role", "Unknown")
#     prompt = payload.get("prompt", None)
#     if prompt is not None:
#         # message = f"Role: {role}\nMessage: {prompt}"
#         message = {
#             "role": role,
#             "content": prompt,
#         }
#         return_messages = [message]
#         # all_messages.append(message)
#     else:
#         messages = payload.get("messages", {})
#         if isinstance(messages, list):
#             return_messages = messages
#         else:
#             logger.warning(f"Unknown type: {type(messages)}")
#
#     return return_messages
#     # if messages is not None:
#     #     for message in messages:
#     #         if message is None:
#     #             continue
#     #         role = message.get("role", "Unknown")
#     #         contents = message.get("content", {})
#     #         if contents is None:
#     #             continue
#     #         single_message = f"Role: {role}\n"
#     #         if isinstance(contents, str):
#     #             single_message += f"Message: {contents}"
#     #         elif isinstance(contents, list):
#     #             for content in contents:
#     #                 if content is None:
#     #                     continue
#     #                 if isinstance(content, str):
#     #                     text = content
#     #                 elif isinstance(content, dict):
#     #                     text = content.get("text", "")
#     #                 else:
#     #                     raise RuntimeError(f"Unknown type: {type(content)}")
#     #                 single_message += f"Message: {text}"
#     #         else:
#     #             raise RuntimeError(f"Unknown type: {type(contents)}")
#     #         all_messages.append(single_message)
#

# async def _condense_prompt(payload: Dict[str, Any], config: Config, max_history_size: int) -> \
#         Dict[str, Any]:
#     """Attempts to condense a long prompt into a shorter, more manageable version."""
#     text_list = _get_messages_from_payload(payload)
#     flat_text = "\n".join(text_list)
#
#     k_first, k_last = config.keep_first_n, config.keep_last_n
#
#     text_tokens = _tokenizer.encode(flat_text)
#
#     first_text_tokens = text_tokens[:k_first]
#     first_text = _tokenizer.decode(first_text_tokens)
#
#     last_text_tokens = text_tokens[-k_last:]
#     last_text = _tokenizer.decode(last_text_tokens)
#
#     # conversation_history = (
#     #     f"Snippet of start of conversation: {first_text}\n\n"
#     #     f"Conversation summary: {text}\n\n"
#     #     f"Snippet of end of conversation: {last_text}"
#     # )
#     condensed_payload = {
#         "model": payload["model"],
#         "messages": text_list,
#         # "n": config.context_length - _token_count(text),
#         "max_tokens": max(0, min(config.max_new_tokens, config.context_length - max_history_size)),
#         "temperature": 0,
#         "top_p": 1,
#         "frequency_penalty": 2,
#         "presence_penalty": 2,
#     }
#
#     # Try to get a condensed version with retries
#     for attempt in range(config.condense_retry_attempts):
#         flat_text_token_count = _token_count(flat_text)
#         if max_history_size < 0:
#             logger.warning(
#                 f"Something has gone wrong, max_history_size is below zero: {max_history_size}"
#             )
#             pass
#         if flat_text_token_count > max_history_size:
#             percent_of_context = max_history_size / flat_text_token_count
#             number_of_elements = int(len(text_list) * percent_of_context)
#             prompt = text_list[-number_of_elements:]
#             condensed_payload["messages"] = prompt
#             new_flat_prompt = "\n".join(prompt)
#             new_prompt_size = _token_count(new_flat_prompt)
#             logger.warning(
#                 f"truncating conversation for condensation, from: {flat_text_token_count} down to: {new_prompt_size}..."
#             )
#             condensed_payload["max_tokens"] = max(0,
#                                                   min(config.max_new_tokens, config.context_length - new_prompt_size))
#         condensed_payload["messages"] = text_list + [CONDENSE_PROMPT]
#
#         try:
#             result = await global_client.completions.create(**condensed_payload)
#             condensed_text = result.choices[0].text.strip()
#
#             condensed_text = (
#                 f"Snippet of start of conversation:\n {first_text}\n\n---\n\n"
#                 f"Conversation summary:\n {condensed_text}\n\n---\n\n"
#                 f"Snippet of end of conversation:\n {last_text}\n\n---\n\n"
#             )
#
#             # Update the payload with condensed text
#             _update_payload_text(payload, condensed_text)
#             logger.info(f"Prompt condensed successfully after {attempt + 1} attempts")
#
#             logger.info(f"Old prompt: {flat_text_token_count} tokens")
#             logger.info(f"Condensed prompt: {_token_count(condensed_text)} tokens")
#             logger.debug(f"Old prompt: \n{text_list}")
#             logger.debug("*" * 100)
#             logger.debug(f"New text: \n{condensed_text}")
#             logger.debug("*" * 100)
#             break
#         except Exception as e:
#             logger.warning(f"Condensation attempt {attempt + 1}/{config.condense_retry_attempts} failed: {e}")
#             if attempt > 0:
#                 new_max_history_size = int(max_history_size * 1.5)
#                 logger.info(
#                     f"Condensation increasing max_history_size from {max_history_size} to: "
#                     f"{new_max_history_size}"
#                 )
#                 max_history_size = new_max_history_size
#                 if max_history_size > config.context_length:
#                     logger.error(f"Tried full {config.context_length} for history, and still failed...")
#                     break
#             if attempt < config.condense_retry_attempts - 1:
#                 # Exponential backoff would go here
#                 pass
#             else:
#                 raise e
#
#     return payload


async def _stream_response_generator(
        stream: AsyncGenerator[Any, None],
) -> AsyncGenerator[str, None]:
    """Yields data chunks for a streaming response."""
    async for chunk in stream:
        yield f"data: {json.dumps(chunk.to_dict())}\n\n"


async def condense_prompt(original_payload: dict, maximum_prompt_length: int) -> list[Any] | None:
    iterable_payload = {
        "model": original_payload["model"],
        "max_tokens": config.approx_condensation_tokens,
        "temperature": 0,
        "seed": 1,
        "top_p": 0.8,
        "frequency_penalty": 1.05,
        "presence_penalty": 1.05,
        "stream": False,
    }
    original_prompt_copy = deepcopy(original_payload)
    original_prompt_list = original_prompt_copy["messages"]
    original_prompt_list_token_count = _token_count(original_prompt_list)

    for attempt in range(1, config.condense_construction_attempts + 1):
        logger.warning(f"Condense construct {attempt} out of {config.condense_retry_attempts}")

        # we use the OG payload here, otherwise there will be a tonne of condense messages added
        iterable_payload["messages"] = possible_truncate_and_append_condense_message(
            maximum_prompt_length, original_payload, original_prompt_list_token_count
        )

        try:
            # call API
            health_response = await health()
            if health_response.status_code == 500:
                error_message = f"LLM is unhealthy, will not complete this request: {health_response.body}"
                logger.warning(error_message)
                return None
            result = await global_client.chat.completions.create(**iterable_payload)
            condensed_response = result.choices[0].message.content.strip()

            new_history = []

            if config.keep_first_n > 0:
                percent_of_context_first = config.keep_first_n / original_prompt_list_token_count
                keep_first_n_summary = truncate_payload_by_percent(
                    original_prompt_list_token_count, original_payload, percent_of_context_first, maximum_prompt_length,
                    take_back=False,
                )
                new_history.extend(keep_first_n_summary)

            condensed_response_formatted = {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": condensed_response,
                    "cache_control": {"type": "ephemeral"},
                }]
            }
            new_history.append(condensed_response_formatted)

            if config.keep_last_n > 0:
                percent_of_context_last = config.keep_last_n / original_prompt_list_token_count
                keep_last_n_summary = truncate_payload_by_percent(
                    original_prompt_list_token_count, original_payload, percent_of_context_last, maximum_prompt_length,
                    take_back=True,
                )
                new_history.extend(keep_last_n_summary)

            length_of_new_prompt = _token_count(new_history)
            if length_of_new_prompt > maximum_prompt_length:
                reduced_new_tokens = config.context_length - length_of_new_prompt

                if reduced_new_tokens <= 0:
                    new_maximum_prompt_length = int(maximum_prompt_length * 0.9)
                    logger.error(
                        f"Condensed prompt {length_of_new_prompt} "
                        f"is larger than the whole context {config.context_length}, "
                        f"will not be able to generate any new tokens.  "
                        f"Retrying by reducing maximum_prompt_length "
                        f"from {maximum_prompt_length} to {new_maximum_prompt_length}"
                    )
                    maximum_prompt_length = new_maximum_prompt_length
                    continue
                logger.warning(
                    f"New prompt is longer than expected, expected: {maximum_prompt_length} got: {length_of_new_prompt} "
                    f"will reduce length of generated tokens, from {iterable_payload['max_tokens']} to {reduced_new_tokens}"
                )
                iterable_payload["max_tokens"] = reduced_new_tokens

            logger.info(f"Condensed {original_prompt_list_token_count} to {length_of_new_prompt}")
            return new_history
        except Exception as e:
            traceback.print_exc()
            new_maximum_prompt_length = int(maximum_prompt_length * 0.9)
            logger.warning(
                f"Had problem, will reduce maximum_prompt_length from {maximum_prompt_length} to {new_maximum_prompt_length}"
            )
            maximum_prompt_length = new_maximum_prompt_length
    raise RuntimeError("Could not condense history")


def possible_truncate_and_append_condense_message(
        maximum_prompt_length: int, original_payload: dict, original_prompt_list_token_count: int
):
    # todo: idea:
    #   rather than truncating the messages and throwing away the rest, what about:
    #       1. find as many messages that will fit in the context
    #       2. ask the LLM for a summary
    #       3. get next batch
    #       4. repeat
    #       5. summarize the summaries
    history: list = original_payload["messages"]
    if original_prompt_list_token_count > maximum_prompt_length:
        logger.warning(
            f"history is too long {original_prompt_list_token_count} > {maximum_prompt_length}.  Will truncate"
        )

        percent_of_context = maximum_prompt_length / original_prompt_list_token_count
        truncated_history = truncate_payload_by_percent(
            original_prompt_list_token_count, original_payload, percent_of_context, maximum_prompt_length,
            take_back=True
        )
        history = truncated_history

    condense_message = {
        "role": "system",
        "content": CONDENSE_PROMPT,
    }
    history.append(condense_message)
    return history


def truncate_payload_by_percent(
        length_of_old_prompt: int, payload: dict, percent_of_context: float,
        maximum_prompt_length: int, take_back: bool
) -> list:
    logger.debug(
        f"Will try to take {percent_of_context * 100:.2f}% of {length_of_old_prompt} tokens, "
        f"from {len(payload['messages'])} messages"
    )

    for attempt in range(1, config.condense_truncate_retry_attempts + 1):
        prompt_list = deepcopy(payload.get("messages", []))

        logger.debug(f"Truncate {attempt} out of {config.condense_truncate_retry_attempts + 1}")

        logger.debug("Truncating number of history messages...")
        truncated_history = truncate_list_based_on_percent(
            percent_of_context, prompt_list, take_back
        )

        if _token_count(truncated_history) > maximum_prompt_length:
            logger.info("Truncating length of each message...")
            for t in truncated_history:
                content = t.get("content", None)
                if content is not None:
                    if isinstance(content, list):
                        for c in content:
                            if _token_count(truncated_history) > maximum_prompt_length:
                                truncated_content = truncate_list_based_on_percent(
                                    percent_of_context, c, take_back
                                )
                                t["content"] = truncated_content
                    else:
                        if _token_count(truncated_history) > maximum_prompt_length:
                            truncated_content = truncate_list_based_on_percent(
                                percent_of_context, content, take_back
                            )
                            t["content"] = truncated_content

        truncated_history_token_count = _token_count(truncated_history)
        # if truncated_history_token_count > maximum_prompt_length:
        #     new_percent_of_context = percent_of_context * 0.9
        #     logger.debug(
        #         f"Truncated prompt is still too large: {truncated_history_token_count} > {maximum_prompt_length} "
        #         f"reducing percent context from: {percent_of_context * 100:.3f}% to {new_percent_of_context * 100:.3f}"
        #     )
        #     percent_of_context = new_percent_of_context
        #     continue
        logger.debug(
            f"Truncated history from {length_of_old_prompt} to {truncated_history_token_count}"
        )
        return truncated_history
    raise RuntimeError("Failed to truncate!")


def truncate_list_based_on_percent(
        percent_of_context: float, prompt_list: list[Any], take_back: bool
) -> list:
    len_of_prompt_list = len(prompt_list)
    if len_of_prompt_list < 1:
        raise RuntimeError(f"len_of_prompt_list ({len_of_prompt_list}) < 1")
    elif len_of_prompt_list == 1:
        logger.info("Only one message is in the history - so will retain it")
        return prompt_list

    number_of_elements_to_keep = ceil(len_of_prompt_list * percent_of_context) - 1
    if number_of_elements_to_keep <= 0:
        logger.info("I have no other options in truncate_list_based_on_percent to make the prompt smaller.")
        number_of_elements_to_keep = 1

    if take_back:
        slice_for_lists = slice(-number_of_elements_to_keep, len_of_prompt_list)
    else:
        slice_for_lists = slice(0, number_of_elements_to_keep)

    logger.info(
        f"slice_for_lists: {slice_for_lists}, "
        f"number_of_elements_to_keep: {number_of_elements_to_keep}, "
        f"len_of_prompt_list: {len_of_prompt_list}, "
        f"percent_of_context: {percent_of_context * 100.0}:.2f %, "
        f"take_back: {take_back}"
    )
    truncated_history: list = prompt_list[slice_for_lists]
    logger.info(f"Truncating list from: {len_of_prompt_list} elements down to: {len(truncated_history)}")
    return truncated_history


async def _handle_completion_request(
        og_payload: Dict[str, Any],
        api_call: Callable[..., Any]
) -> Response:
    """Handles the logic for a single completion request, including retries."""
    is_stream = og_payload.pop("stream", False)
    og_payload.pop("reasoning", None)

    new_payload = deepcopy(og_payload)
    new_payload.update({
        "top_p": 0.8,
        "frequency_penalty": 1.05,
        "presence_penalty": 1.05,
        "stream": is_stream,
    })
    maximum_prompt_length = config.context_length - (
            new_payload.get("max_tokens", config.approx_condensation_tokens) + CONDENSE_PROMPT_TOKEN_COUNT
            + config.keep_first_n + config.keep_last_n
    )
    for attempt in range(0, config.condense_retry_attempts + 1):
        try:
            payload_token_count = _token_count(new_payload)
            max_tokens = new_payload.get("max_tokens", config.max_new_tokens)
            if payload_token_count > config.context_length:
                logger.warning(
                    f"Will probably fail, request tokens is greater than context size "
                    f"{payload_token_count} > {config.context_length}"
                )
            elif payload_token_count > config.context_length - max_tokens:
                logger.warning(
                    f"Will probably fail, max_tokens + prompt is greater than context size "
                    f"{max_tokens} + {payload_token_count} > {config.context_length}"
                )

            return await attempt_request(api_call, is_stream, new_payload)
        except Exception as e:
            logger.warning("Failed to call the API, will try and condense...")
            new_prompt = await condense_prompt(new_payload, maximum_prompt_length)
            if new_prompt is None:
                return JSONResponse(status_code=500, content="Upstream failed healthcheck, will close this request")
            new_payload["messages"] = new_prompt
    return JSONResponse(status_code=500, content="Failed to get completion")


async def attempt_request(api_call, is_stream, payload):
    response = await health()
    if response.status_code == 500:
        error_message = f"Healthcheck failed, will not call LLM: {response.body}"
        logger.warning(error_message)
        return JSONResponse(content=error_message, status_code=500)

    if is_stream:
        stream = await api_call(**payload)
        return StreamingResponse(
            _stream_response_generator(stream), media_type="text/event-stream"
        )
    else:
        result = await api_call(**payload)
        return JSONResponse(content=result.to_dict())


"""Creates and configures the FastAPI application."""
app = FastAPI()


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    models_page = await global_client.models.list()
    return JSONResponse(content=[model.to_dict() for model in models_page.data])


@app.get("/health")
async def health() -> JSONResponse:
    try:
        health_status = await global_client.get("../health", cast_to=httpx.Response)
        if int(health_status.status_code / 100) == 2:
            return JSONResponse(content="OK")
    except Exception as e:
        logger.warning(e)
    return JSONResponse(status_code=500, content={"error": "upstream is unhealthy"})


@app.post("/v1/chat/completions")
async def chat_completions_handler(request: Request) -> Response:
    try:
        payload = await request.json()
        return await _handle_completion_request(payload, global_client.chat.completions.create)
    except Exception as e:
        logger.error(f"Error in chat/completions: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/completions")
async def completions_handler(request: Request) -> Response:
    try:
        logger.warning("/v1/completions is deprecated, this program may crash. Use /v1/chat/completions")
        payload = await request.json()
        return await _handle_completion_request(payload, global_client.completions.create)
    except Exception as e:
        logger.error(f"Error in completions: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
