"""LLM vision-based OCR module with response logging and multi-model fallback."""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import httpx
import openai
import anthropic
from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds

# Transient error strings that should trigger a retry
TRANSIENT_MARKERS = [
    "429", "rate", "resource_exhausted",  # rate limits
    "disconnected", "server disconnected",  # connection drops
    "timeout", "timed out", "deadline",  # timeouts
    "502", "503", "504",  # server errors
    "connection reset", "connection refused",
    "broken pipe", "eof occurred",
]


@dataclass
class OCRResult:
    """Result from an OCR call with metadata."""
    text: str
    finish_reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    is_fallback: bool = False


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is transient and worth retrying."""
    exc_str = str(exc).lower()
    if any(marker in exc_str for marker in TRANSIENT_MARKERS):
        return True
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code", 0)
        if code in (429, 500, 502, 503, 504):
            return True
    if isinstance(exc, (httpx.RemoteProtocolError, httpx.ConnectError,
                        httpx.ReadTimeout, httpx.WriteTimeout,
                        httpx.ConnectTimeout, httpx.PoolTimeout,
                        ConnectionError, TimeoutError, OSError)):
        return True
    return False


async def ocr_image_openai(image_bytes: bytes, prompt: str, api_key: str, model: str,
                           base_url: Optional[str] = None) -> OCRResult:
    """Send an image to OpenAI-compatible vision API and extract text."""
    kwargs = {"api_key": api_key, "timeout": 120.0}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**kwargs)
    b64 = base64.b64encode(image_bytes).decode()
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
        max_tokens=16384,
    )
    choice = response.choices[0]
    return OCRResult(
        text=choice.message.content or "",
        finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
        provider="openai" if not base_url else "xai",
        model=model,
    )


async def ocr_image_anthropic(image_bytes: bytes, prompt: str, api_key: str, model: str) -> OCRResult:
    """Send an image to Anthropic vision API via streaming (required for Opus)."""
    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=600.0)
    b64 = base64.b64encode(image_bytes).decode()

    text = ""
    stop_reason = None
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=16384,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        ) as stream:
            async for chunk in stream.text_stream:
                text += chunk
            final = await stream.get_final_message()
            stop_reason = str(final.stop_reason) if final.stop_reason else None
    except anthropic.APIStatusError as exc:
        if "content filtering" in str(exc).lower() or "blocked" in str(exc).lower():
            logger.warning("Anthropic content filter blocked output: %s", exc)
            return OCRResult(text="", finish_reason="CONTENT_FILTER", provider="anthropic", model=model)
        raise

    return OCRResult(
        text=text,
        finish_reason=stop_reason,
        provider="anthropic",
        model=model,
    )


async def ocr_image_gemini(image_bytes: bytes, prompt: str, api_key: str, model: str) -> OCRResult:
    """Send an image to Google Gemini vision API and extract text."""
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000),
    )
    response = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    finish_reason = None
    text = ""
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = str(candidate.finish_reason) if candidate.finish_reason else None
        if candidate.content and candidate.content.parts:
            text = candidate.content.parts[0].text or ""

    return OCRResult(
        text=text,
        finish_reason=finish_reason,
        provider="gemini",
        model=model,
    )


async def _ocr_dispatch(image_bytes: bytes, prompt: str, provider: str, model: str, api_key: str) -> OCRResult:
    """Route to the correct LLM provider for OCR."""
    if provider == "anthropic":
        return await ocr_image_anthropic(image_bytes, prompt, api_key, model)
    if provider == "gemini":
        return await ocr_image_gemini(image_bytes, prompt, api_key, model)
    if provider == "xai":
        return await ocr_image_openai(image_bytes, prompt, api_key, model, base_url="https://api.x.ai/v1")
    return await ocr_image_openai(image_bytes, prompt, api_key, model)


def _is_content_blocked(result: OCRResult) -> bool:
    """Check if the response was blocked by any content filter (recitation, safety, etc)."""
    if result.finish_reason:
        fr = result.finish_reason.upper()
        if "RECITATION" in fr or "CONTENT_FILTER" in fr or "BLOCKED" in fr:
            return True
    return False


def _is_empty_response(result: OCRResult) -> bool:
    """Check if the response is effectively empty."""
    return len(result.text.strip()) < 10


async def _ocr_with_retries(image_bytes: bytes, prompt: str, provider: str, model: str, api_key: str) -> OCRResult:
    """OCR with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await _ocr_dispatch(image_bytes, prompt, provider, model, api_key)
        except Exception as exc:
            if _is_transient(exc) and attempt < MAX_RETRIES - 1:
                delay = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning("Transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                               attempt + 1, MAX_RETRIES, type(exc).__name__, delay)
                await asyncio.sleep(delay)
            else:
                raise
    return await _ocr_dispatch(image_bytes, prompt, provider, model, api_key)


def _try_fallback(result: OCRResult) -> bool:
    """Check if we should try the next fallback model."""
    return _is_content_blocked(result) or (_is_empty_response(result) and result.finish_reason is not None)


async def ocr_image(image_bytes: bytes, config: Config, context_prefix: str = "") -> str:
    """OCR with multi-model fallback chain.

    Chain: Primary → Fallback-1 → Fallback-2
    Falls through on blocked/empty responses. Same prompt for all models.

    Returns:
        Raw LLM output (may contain ---CONTEXT--- delimiter).
    """
    full_prompt = context_prefix + config.ocr_prompt

    # Step 1: Primary model
    result = await _ocr_with_retries(image_bytes, full_prompt,
                                     config.llm_provider, config.llm_model, config.llm_api_key)
    _log_result("Primary", result)

    if not _try_fallback(result):
        return result.text

    logger.warning("Primary model blocked/empty (%s, %d chars).", result.finish_reason, len(result.text))

    # Step 2: First fallback
    if config.fallback_provider and config.fallback_api_key:
        logger.info("Falling back to %s/%s...", config.fallback_provider, config.fallback_model)
        try:
            result = await _ocr_with_retries(image_bytes, full_prompt,
                                             config.fallback_provider, config.fallback_model,
                                             config.fallback_api_key)
            result.is_fallback = True
            _log_result("Fallback-1", result)

            if not _try_fallback(result):
                logger.info("Fallback-1 succeeded (%d chars)", len(result.text))
                return result.text

            logger.warning("Fallback-1 also blocked/empty (%s, %d chars).",
                           result.finish_reason, len(result.text))
        except Exception:
            logger.exception("Fallback-1 failed")

    # Step 3: Second fallback
    if config.fallback2_provider and config.fallback2_api_key:
        logger.info("Falling back to %s/%s...", config.fallback2_provider, config.fallback2_model)
        try:
            result = await _ocr_with_retries(image_bytes, full_prompt,
                                             config.fallback2_provider, config.fallback2_model,
                                             config.fallback2_api_key)
            result.is_fallback = True
            _log_result("Fallback-2", result)

            if not _is_empty_response(result):
                logger.info("Fallback-2 succeeded (%d chars)", len(result.text))
                return result.text
            else:
                logger.error("Fallback-2 also empty (%s, %d chars)",
                             result.finish_reason, len(result.text))
        except Exception:
            logger.exception("Fallback-2 failed")

    logger.error("All models exhausted — returning best available result (%d chars)", len(result.text))
    return result.text


def _log_result(label: str, result: OCRResult) -> None:
    """Log OCR response metadata."""
    text_len = len(result.text) if result.text else 0
    has_context = "---CONTEXT---" in result.text if result.text else False
    preview = (result.text[:150].replace("\n", " ") + "...") if text_len > 150 else (result.text or "").replace("\n", " ")

    logger.info("[%s] %s/%s — finish_reason=%s, text=%d chars, has_context=%s",
                label, result.provider, result.model, result.finish_reason, text_len, has_context)

    if text_len == 0:
        logger.warning("[%s] ⚠️ EMPTY RESPONSE — finish_reason=%s", label, result.finish_reason)
    elif text_len < 50:
        logger.warning("[%s] ⚠️ Very short response: '%s'", label, preview)
    else:
        logger.debug("[%s] Preview: %s", label, preview)
