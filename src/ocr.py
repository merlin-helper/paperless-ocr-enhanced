"""LLM vision-based OCR module."""

import base64
import logging

import openai
import anthropic

from .config import Config

logger = logging.getLogger(__name__)


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


async def ocr_image(image_bytes: bytes, config: Config) -> str:
    """Route to the correct LLM provider for OCR."""
    if config.llm_provider == "anthropic":
        return await ocr_image_anthropic(image_bytes, config)
    return await ocr_image_openai(image_bytes, config)
