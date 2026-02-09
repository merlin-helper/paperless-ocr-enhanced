"""Main entry point for paperless-ocr-enhanced."""

import asyncio
import logging
import signal
import sys

from .config import Config
from .paperless import PaperlessClient
from .pdf import render_pages_to_images, build_searchable_pdf
from .ocr import ocr_image

logger = logging.getLogger("paperless-ocr-enhanced")

shutdown_event = asyncio.Event()


def handle_signal(*_: object) -> None:
    logger.info("Shutdown signal received")
    shutdown_event.set()


async def process_document(doc: dict, config: Config, client: PaperlessClient,
                           trigger_tag_id: int, complete_tag_id: int, failed_tag_id: int) -> None:
    """Process a single document: download, OCR, rebuild, upload."""
    doc_id = doc["id"]
    title = doc.get("title", f"Document {doc_id}")
    logger.info("Processing document %d: %s", doc_id, title)

    try:
        # Download original PDF
        logger.info("[%d] Downloading original PDF...", doc_id)
        pdf_bytes = await client.download_original(doc_id)
        logger.info("[%d] Downloaded %d bytes", doc_id, len(pdf_bytes))

        # Render pages to images
        logger.info("[%d] Rendering pages to images...", doc_id)
        images = render_pages_to_images(pdf_bytes, max_pages=config.max_pages)
        logger.info("[%d] Rendered %d pages", doc_id, len(images))

        # OCR each page via LLM vision
        page_texts: list[str] = []
        for i, img in enumerate(images):
            logger.info("[%d] OCR page %d/%d via %s (%s)...", doc_id, i + 1, len(images),
                        config.llm_provider, config.llm_model)
            text = await ocr_image(img, config)
            page_texts.append(text)
            preview = text[:100].replace("\n", " ")
            logger.debug("[%d] Page %d text preview: %s", doc_id, i + 1, preview)
            # Brief delay between pages to avoid rate limits
            if i < len(images) - 1:
                await asyncio.sleep(1.0)

        # Build searchable PDF
        logger.info("[%d] Building searchable PDF...", doc_id)
        searchable_pdf = build_searchable_pdf(pdf_bytes, page_texts, max_pages=config.max_pages)
        logger.info("[%d] Searchable PDF: %d bytes", doc_id, len(searchable_pdf))

        # Update tags: remove trigger, add complete
        current_tags: list[int] = doc.get("tags", [])
        new_tags = [t for t in current_tags if t != trigger_tag_id]
        if complete_tag_id not in new_tags:
            new_tags.append(complete_tag_id)

        # Update the document content and tags
        # We set the content field so Paperless indexes the extracted text
        full_text = "\n\n--- Page Break ---\n\n".join(page_texts)
        await client.update_document(doc_id, {
            "tags": new_tags,
            "content": full_text,
        })
        logger.info("[%d] Updated tags and content for document", doc_id)

        # Also upload the searchable PDF as archived version by overwriting
        # Note: Paperless doesn't have a direct "replace archived" endpoint,
        # so we update the content field which makes it searchable.
        # The original file remains unchanged.
        logger.info("[%d] ✅ Complete", doc_id)

    except Exception:
        logger.exception("[%d] Failed to process document", doc_id)
        # Add failed tag
        try:
            current_tags = doc.get("tags", [])
            new_tags = [t for t in current_tags if t != trigger_tag_id]
            if failed_tag_id not in new_tags:
                new_tags.append(failed_tag_id)
            await client.update_document(doc_id, {"tags": new_tags})
            logger.info("[%d] Added '%s' tag", doc_id, config.failed_tag)
        except Exception:
            logger.exception("[%d] Failed to add failure tag", doc_id)


async def poll_loop(config: Config) -> None:
    """Main polling loop."""
    client = PaperlessClient(config)

    try:
        # Resolve tag IDs
        trigger_tag_id = await client.ensure_tag(config.trigger_tag)
        complete_tag_id = await client.ensure_tag(config.complete_tag)
        failed_tag_id = await client.ensure_tag(config.failed_tag)
        logger.info("Tag IDs — trigger=%d, complete=%d, failed=%d",
                     trigger_tag_id, complete_tag_id, failed_tag_id)

        while not shutdown_event.is_set():
            try:
                docs = await client.get_documents_by_tag(trigger_tag_id)
                if docs:
                    logger.info("Found %d document(s) to process", len(docs))
                    for doc in docs:
                        if shutdown_event.is_set():
                            break
                        await process_document(doc, config, client,
                                               trigger_tag_id, complete_tag_id, failed_tag_id)
                else:
                    logger.debug("No documents found with tag '%s'", config.trigger_tag)
            except Exception:
                logger.exception("Error during poll cycle")

            # Wait for poll interval or shutdown
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=config.poll_interval)
            except asyncio.TimeoutError:
                pass

    finally:
        await client.close()


def main() -> None:
    config = Config()

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    logger.info("paperless-ocr-enhanced starting")
    logger.info("Paperless URL: %s", config.paperless_url)
    logger.info("LLM provider: %s (%s)", config.llm_provider, config.llm_model)
    logger.info("Trigger tag: %s | Complete tag: %s", config.trigger_tag, config.complete_tag)
    logger.info("Poll interval: %ds | Max pages: %s",
                config.poll_interval, config.max_pages or "unlimited")

    try:
        config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    try:
        loop.run_until_complete(poll_loop(config))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        loop.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
