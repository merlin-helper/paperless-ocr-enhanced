"""PDF rendering and searchable PDF creation using PyMuPDF."""

import logging

import fitz  # pymupdf

logger = logging.getLogger(__name__)


def render_pages_to_images(pdf_bytes: bytes, max_pages: int = 0, dpi: int = 300) -> list[bytes]:
    """Render each page of a PDF to a PNG image. Returns list of PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []
    total = len(doc)
    if max_pages > 0:
        total = min(total, max_pages)
    for i in range(total):
        page = doc[i]
        # Render at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        images.append(pix.tobytes("png"))
        logger.debug("Rendered page %d/%d (%dx%d)", i + 1, total, pix.width, pix.height)
    doc.close()
    return images


def build_searchable_pdf(original_pdf_bytes: bytes, page_texts: list[str], max_pages: int = 0) -> bytes:
    """Build a searchable PDF by overlaying invisible text on original pages.

    Opens the original PDF and inserts invisible text on each page so the
    visual appearance is unchanged but the text is searchable/selectable.
    """
    doc = fitz.open(stream=original_pdf_bytes, filetype="pdf")
    total = len(doc)
    if max_pages > 0:
        total = min(total, max_pages)

    for i in range(total):
        if i >= len(page_texts) or not page_texts[i].strip():
            continue

        page = doc[i]
        text = page_texts[i]
        rect = page.rect

        # Insert text as invisible (render mode 3 = invisible) using a text writer
        # We'll use a small font and place text line by line
        fontsize = 10
        writer = fitz.TextWriter(rect)

        # Calculate line positions
        lines = text.split("\n")
        y = rect.y0 + fontsize + 2
        line_height = fontsize * 1.2

        for line in lines:
            if not line.strip():
                y += line_height
                continue
            if y + fontsize > rect.y1:
                break  # No more room on page
            try:
                writer.append((rect.x0 + 5, y), line, fontsize=fontsize)
            except Exception:
                # Skip lines that can't be encoded
                pass
            y += line_height

        # Write with render mode 3 (invisible text)
        writer.write_text(page, render_mode=3, color=(0, 0, 0))

    output = doc.tobytes(deflate=True)
    doc.close()
    return output
