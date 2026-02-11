"""Configuration from environment variables."""

import os
import logging

DEFAULT_OCR_PROMPT = (
    "Extract all text from this document page as clean, well-structured markdown.\n\n"
    "Tables: Convert ANY tabular or columnar data into proper markdown tables using | pipes | "
    "and |---| separator rows. This includes lab results, financial line items, invoices, "
    "itemized lists with values, schedules, and any data arranged in columns — even if the "
    "original uses spaces or visual alignment instead of grid lines. Never preserve "
    "space-aligned columnar layouts; always convert them to pipe-delimited markdown tables.\n\n"
    "Headings: Use proper markdown heading hierarchy — # for title, ## for sections, "
    "### for subsections.\n\n"
    "Formatting: Use **bold** for form field labels and key terms. Preserve all numbers, "
    "dates, dollar amounts, and reference numbers exactly as written.\n\n"
    "Images: For images, charts, maps, logos, diagrams, signatures, or other non-text visual "
    "elements, provide a brief description in [square brackets], e.g. "
    "[US map showing regional divisions]. Do not render visual elements as ASCII art.\n\n"
    "Return only the document content — no commentary or explanation.\n\n"
    "OUTPUT FORMAT: Return the extracted content first, then on a new line write exactly:\n"
    "---CONTEXT---\n"
    "Followed by a brief 1-2 sentence summary of what this page contributes to the document "
    "(document type, key entities, table structures, form patterns, important details). "
    "This context helps process subsequent pages."
)

# Context window settings
CONTEXT_IMMEDIATE_CHARS = 1500   # chars from previous page to include
CONTEXT_SUMMARY_MAX_CHARS = 1500  # max chars for rolling document summary
CONTEXT_CONDENSE_EVERY = 10      # condense rolling summary every N pages

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
        self.context_immediate_chars = int(os.environ.get(
            "CONTEXT_IMMEDIATE_CHARS", str(CONTEXT_IMMEDIATE_CHARS)))
        self.context_summary_max_chars = int(os.environ.get(
            "CONTEXT_SUMMARY_MAX_CHARS", str(CONTEXT_SUMMARY_MAX_CHARS)))
        self.context_condense_every = int(os.environ.get(
            "CONTEXT_CONDENSE_EVERY", str(CONTEXT_CONDENSE_EVERY)))

        # Fallback model config
        self.fallback_provider = os.environ.get("FALLBACK_PROVIDER", "").lower() or None
        self.fallback_model = os.environ.get("FALLBACK_MODEL", "")
        self.fallback_api_key = os.environ.get("FALLBACK_API_KEY", "")

        # Second fallback (tertiary) model config
        self.fallback2_provider = os.environ.get("FALLBACK2_PROVIDER", "").lower() or None
        self.fallback2_model = os.environ.get("FALLBACK2_MODEL", "")
        self.fallback2_api_key = os.environ.get("FALLBACK2_API_KEY", "")

    def validate(self) -> None:
        errors: list[str] = []
        if not self.paperless_url:
            errors.append("PAPERLESS_URL is required")
        if not self.paperless_token:
            errors.append("PAPERLESS_TOKEN is required")
        if not self.llm_api_key:
            errors.append("LLM_API_KEY is required")
        if self.llm_provider not in ("openai", "anthropic", "gemini"):
            errors.append(f"LLM_PROVIDER must be 'openai', 'anthropic', or 'gemini', got '{self.llm_provider}'")
        if errors:
            raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))
