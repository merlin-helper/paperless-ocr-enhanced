"""Paperless-ngx API client."""

import logging
from typing import Any

import httpx

from .config import Config

logger = logging.getLogger(__name__)


class PaperlessClient:
    """Async client for the Paperless-ngx REST API."""

    def __init__(self, config: Config) -> None:
        self.base_url = config.paperless_url
        self.headers = {
            "Authorization": f"Token {config.paperless_token}",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=120.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_tag_id(self, tag_name: str) -> int | None:
        """Get tag ID by name. Creates the tag if it doesn't exist when getting complete/failed tags."""
        client = await self._get_client()
        resp = await client.get("/api/tags/", params={"name__iexact": tag_name})
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0]["id"]
        return None

    async def ensure_tag(self, tag_name: str) -> int:
        """Get or create a tag by name, returning its ID."""
        tag_id = await self.get_tag_id(tag_name)
        if tag_id is not None:
            return tag_id
        logger.info("Creating tag '%s'", tag_name)
        client = await self._get_client()
        resp = await client.post("/api/tags/", json={"name": tag_name})
        resp.raise_for_status()
        return resp.json()["id"]

    async def get_documents_by_tag(self, tag_id: int) -> list[dict[str, Any]]:
        """List all documents with the given tag ID."""
        from urllib.parse import urlparse, parse_qs
        client = await self._get_client()
        results: list[dict[str, Any]] = []
        params: dict[str, Any] = {"tags__id__in": tag_id, "page_size": 100}
        while True:
            resp = await client.get("/api/documents/", params=params)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            next_url = data.get("next")
            if not next_url:
                break
            # Parse the next URL to extract query params
            # (httpx strips query params from full URLs when base_url is set)
            parsed = urlparse(next_url)
            params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}
        return results

    async def download_original(self, doc_id: int) -> bytes:
        """Download the original file for a document."""
        client = await self._get_client()
        resp = await client.get(f"/api/documents/{doc_id}/download/")
        resp.raise_for_status()
        return resp.content

    async def upload_document(self, pdf_bytes: bytes, filename: str, metadata: dict[str, Any] | None = None) -> int | None:
        """Upload a new document via multipart form. Returns task ID or None."""
        client = await self._get_client()
        files = {"document": (filename, pdf_bytes, "application/pdf")}
        data: dict[str, Any] = {}
        if metadata:
            if metadata.get("title"):
                data["title"] = metadata["title"]
            if metadata.get("correspondent"):
                data["correspondent"] = metadata["correspondent"]
            if metadata.get("document_type"):
                data["document_type"] = metadata["document_type"]
            if metadata.get("storage_path"):
                data["storage_path"] = metadata["storage_path"]
            if metadata.get("tags"):
                for tag_id in metadata["tags"]:
                    data.setdefault("tags", [])
                    data["tags"].append(tag_id)
            if metadata.get("created"):
                data["created"] = metadata["created"]
        resp = await client.post("/api/documents/post_document/", files=files, data=data)
        resp.raise_for_status()
        task_id = resp.text.strip().strip('"')
        logger.debug("Upload task: %s", task_id)
        return task_id

    async def update_document(self, doc_id: int, updates: dict[str, Any]) -> None:
        """PATCH a document's metadata."""
        client = await self._get_client()
        resp = await client.patch(f"/api/documents/{doc_id}/", json=updates)
        resp.raise_for_status()

    async def get_document(self, doc_id: int) -> dict[str, Any]:
        """Get full document metadata."""
        client = await self._get_client()
        resp = await client.get(f"/api/documents/{doc_id}/")
        resp.raise_for_status()
        return resp.json()
