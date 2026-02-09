"""Configuration from environment variables."""

import os
import logging

DEFAULT_OCR_PROMPT = (
    "Extract all text from this document page exactly as written. "
    "Preserve the original layout, structure, paragraphs, and line breaks as closely as possible. "
    "Include all headers, footers, captions, table content, and any visible text. "
    "Do not add any commentary, explanation, or formatting markers. "
    "Return only the raw text content."
)

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


class Config:
    def __init__(self) -> None:
        self.paperless_url = os.environ.get("PAPERLESS_URL", "").rstrip("/")
        self.paperless_token = os.environ.get("PAPERLESS_TOKEN", "")
        self.llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        self.llm_model = os.environ.get(
            "LLM_MODEL",
            {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-20250514", "gemini": "gemini-2.0-flash"}.get(
                self.llm_provider, "gpt-4o"
            ),
        )
        self.llm_api_key = os.environ.get("LLM_API_KEY", "")
        self.trigger_tag = os.environ.get("TRIGGER_TAG", "ocr-redo")
        self.complete_tag = os.environ.get("COMPLETE_TAG", "ocr-complete")
        self.failed_tag = os.environ.get("FAILED_TAG", "ocr-failed")
        self.poll_interval = int(os.environ.get("POLL_INTERVAL", "30"))
        self.ocr_prompt = os.environ.get("OCR_PROMPT", DEFAULT_OCR_PROMPT)
        self.log_level = LOG_LEVELS.get(
            os.environ.get("LOG_LEVEL", "info").lower(), logging.INFO
        )
        self.max_pages = int(os.environ.get("MAX_PAGES", "0"))

    def validate(self) -> None:
        errors: list[str] = []
        if not self.paperless_url:
            errors.append("PAPERLESS_URL is required")
        if not self.paperless_token:
            errors.append("PAPERLESS_TOKEN is required")
        if not self.llm_api_key:
            errors.append("LLM_API_KEY is required")
        if self.llm_provider not in ("openai", "anthropic", "gemini"):
            errors.append(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{self.llm_provider}'")
        if errors:
            raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))
