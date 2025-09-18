from __future__ import annotations

import os
from dataclasses import dataclass


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
    truncation_message: str = os.getenv("TRUNCATION_MESSAGE", "\n\n")
    truncation_retries: int = int(os.getenv("TRUNCATION_RETRIES", "10"))
    truncation_factor: float = float(os.getenv("TRUNCATION_FACTOR", "0.9"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    condense_retry_attempts: int = int(os.getenv("CONDENSE_RETRY_ATTEMPTS", "10"))
    approx_condensation_tokens: int = int(os.getenv("APPROX_CONDENSATION_TOKENS", "10000"))
    condense_construction_attempts: int = int(os.getenv("CONDENSE_CONSTRUCTION_ATTEMPTS", "10"))
    condense_truncate_retry_attempts: int = int(os.getenv("CONDENSE_TRUNCATE_RETRY_ATTEMPTS", "500"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", 24 * 60 * 60))
