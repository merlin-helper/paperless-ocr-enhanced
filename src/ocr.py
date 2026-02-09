"""LLM vision-based OCR module."""

import asyncio
import base64
import logging

import openai
import anthropic
from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


async def ocr_image_openai(image_bytes: bytes, config: Config) -> str:
    """Send an image to OpenAI vision API and extract text."""
    client = openai.AsyncOpenAI(api_key=config.llm_api_key)
    b64 = base64.b64encode(image_bytes).decode()
    response = await client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                ],
            }
        ],
        max_tokens=16384,
    )
    return response.choices[0].message.content or ""


async def ocr_image_anthropic(image_bytes: bytes, config: Config) -> str:
    """Send an image to Anthropic vision API and extract text."""
    client = anthropic.AsyncAnthropic(api_key=config.llm_api_key)
    b64 = base64.b64encode(image_bytes).decode()
    response = await client.messages.create(
        model=config.llm_model,
        max_tokens=16384,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {"type": "text", "text": config.ocr_prompt},
                ],
            }
        ],
    )
    return response.content[0].text if response.content else ""


async def ocr_image_gemini(image_bytes: bytes, config: Config) -> str:
    """Send an image to Google Gemini vision API and extract text."""
    client = genai.Client(api_key=config.llm_api_key)
    response = await client.aio.models.generate_content(
        model=config.llm_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            config.ocr_prompt,
        ],
    )
    return response.text or ""


async def _ocr_dispatch(image_bytes: bytes, config: Config) -> str:
    """Route to the correct LLM provider for OCR."""
    if config.llm_provider == "anthropic":
        return await ocr_image_anthropic(image_bytes, config)
    if config.llm_provider == "gemini":
        return await ocr_image_gemini(image_bytes, config)
    return await ocr_image_openai(image_bytes, config)


async def ocr_image(image_bytes: bytes, config: Config) -> str:
    """OCR with exponential backoff on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await _ocr_dispatch(image_bytes, config)
        except Exception as exc:
            # Check for rate-limit (429) errors across providers
            is_rate_limit = False
            exc_str = str(exc).lower()
            if "429" in exc_str or "resource_exhausted" in exc_str or "rate" in exc_str:
                is_rate_limit = True
            if hasattr(exc, "status_code") and getattr(exc, "status_code", 0) == 429:
                is_rate_limit = True

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning("Rate limited (attempt %d/%d), retrying in %.1fs...",
                               attempt + 1, MAX_RETRIES, delay)
                await asyncio.sleep(delay)
            else:
                raise
