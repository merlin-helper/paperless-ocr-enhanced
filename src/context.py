"""Document context management for multi-page OCR."""

import logging
from dataclasses import dataclass, field

from .config import Config

logger = logging.getLogger(__name__)

CONTEXT_DELIMITER = "---CONTEXT---"


@dataclass
class DocumentContext:
    """Maintains rolling context across pages of a document."""
    total_pages: int = 0
    current_page: int = 0
    previous_page_text: str = ""
    rolling_summary: str = ""
    page_contexts: list[str] = field(default_factory=list)

    def build_context_prompt(self, config: Config) -> str:
        """Build context prefix to prepend to the OCR prompt."""
        parts: list[str] = []

        # Page position
        parts.append(f"[Page {self.current_page} of {self.total_pages}]")

        # Rolling document summary (accumulated understanding)
        if self.rolling_summary:
            parts.append(
                f"\n--- DOCUMENT CONTEXT (from pages 1-{self.current_page - 1}) ---\n"
                f"{self.rolling_summary}"
            )

        # Immediate context from previous page
        if self.previous_page_text:
            truncated = self.previous_page_text[-config.context_immediate_chars:]
            if len(self.previous_page_text) > config.context_immediate_chars:
                truncated = "..." + truncated
            parts.append(
                f"\n--- PREVIOUS PAGE (end of page {self.current_page - 1}) ---\n"
                f"{truncated}"
            )

        if len(parts) <= 1:
            # First page — no context yet
            return f"{parts[0]}\n\n"

        parts.append("\n--- NOW PROCESS THIS PAGE ---\n")
        return "\n".join(parts) + "\n\n"

    def update_after_page(self, raw_output: str, config: Config) -> str:
        """Parse LLM output, extract context note, update rolling summary.

        Returns the clean OCR text (without the context delimiter block).
        """
        # Split on context delimiter
        if CONTEXT_DELIMITER in raw_output:
            parts = raw_output.split(CONTEXT_DELIMITER, 1)
            ocr_text = parts[0].rstrip()
            context_note = parts[1].strip()
        else:
            ocr_text = raw_output.rstrip()
            context_note = ""

        # Store page text for immediate context
        self.previous_page_text = ocr_text

        # Accumulate context notes
        if context_note:
            self.page_contexts.append(f"p{self.current_page}: {context_note}")
            logger.debug("[context] Page %d note: %s", self.current_page,
                         context_note[:100])

        # Update rolling summary
        self._update_rolling_summary(config)

        return ocr_text

    def _update_rolling_summary(self, config: Config) -> None:
        """Rebuild rolling summary from accumulated page contexts."""
        if not self.page_contexts:
            return

        # Every N pages, condense to keep summary compact
        if (self.current_page % config.context_condense_every == 0
                and len(self.page_contexts) > config.context_condense_every):
            # Keep the condensed summary of older pages + recent individual notes
            older = self.page_contexts[:-config.context_condense_every]
            recent = self.page_contexts[-config.context_condense_every:]

            # Compress older notes into a single summary line
            older_text = " | ".join(older)
            if len(older_text) > config.context_summary_max_chars // 2:
                older_text = older_text[-(config.context_summary_max_chars // 2):]
                older_text = "..." + older_text

            self.rolling_summary = (
                f"Earlier pages summary: {older_text}\n"
                f"Recent pages: {' | '.join(recent)}"
            )
        else:
            self.rolling_summary = " | ".join(self.page_contexts)

        # Hard cap on summary length
        if len(self.rolling_summary) > config.context_summary_max_chars:
            self.rolling_summary = (
                "..." + self.rolling_summary[-(config.context_summary_max_chars - 3):]
            )
