# paperless-ocr-enhanced

LLM vision-based OCR for [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). Replaces Tesseract OCR with GPT-4o or Claude for dramatically better text extraction, especially on complex layouts, handwriting, and low-quality scans.

## How it works

1. Polls Paperless-ngx for documents tagged with a trigger tag (default: `ocr-redo`)
2. Downloads the original PDF
3. Renders each page as a high-resolution image using PyMuPDF
4. Sends each page to an LLM vision API (OpenAI or Anthropic) for text extraction
5. Updates the document's content field in Paperless-ngx with the extracted text, making it fully searchable
6. Removes the trigger tag and adds a completion tag (default: `ocr-complete`)
7. On failure, adds an `ocr-failed` tag and moves on

## Quick Start

```yaml
# docker-compose.yml
services:
  paperless-ocr-enhanced:
    image: ghcr.io/merlin-helper/paperless-ocr-enhanced:latest
    environment:
      PAPERLESS_URL: "http://paperless-ngx:8000"
      PAPERLESS_TOKEN: "your-api-token"
      LLM_API_KEY: "sk-..."
    restart: unless-stopped
```

Then tag any document in Paperless-ngx with `ocr-redo` and it will be automatically re-OCR'd.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PAPERLESS_URL` | ✅ | — | Paperless-ngx base URL |
| `PAPERLESS_TOKEN` | ✅ | — | Paperless-ngx API token |
| `LLM_API_KEY` | ✅ | — | OpenAI or Anthropic API key |
| `LLM_PROVIDER` | | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | | `gpt-4o` | Model to use for vision OCR |
| `TRIGGER_TAG` | | `ocr-redo` | Tag that triggers processing |
| `COMPLETE_TAG` | | `ocr-complete` | Tag added after success |
| `FAILED_TAG` | | `ocr-failed` | Tag added on failure |
| `POLL_INTERVAL` | | `30` | Seconds between polls |
| `OCR_PROMPT` | | *(built-in)* | Custom prompt for vision model |
| `LOG_LEVEL` | | `info` | `debug`, `info`, `warn`, `error` |
| `MAX_PAGES` | | `0` | Max pages per doc (0 = unlimited) |

## Getting a Paperless-ngx API Token

In Paperless-ngx, go to **Settings → API** or run:

```bash
docker exec -it paperless python manage.py get_api_token <username>
```

## Building

```bash
docker build -t paperless-ocr-enhanced .
```

## License

MIT
