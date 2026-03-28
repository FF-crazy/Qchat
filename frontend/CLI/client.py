"""HTTP client for communicating with the Qchat backend."""

from typing import AsyncIterator
import httpx


class QchatClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

    async def close(self) -> None:
        await self._client.aclose()

    # ── providers ──

    async def get_providers(self) -> list[str]:
        resp = await self._client.get("/local/providers")
        resp.raise_for_status()
        return resp.json()

    # ── config ──

    async def get_config(self) -> dict:
        resp = await self._client.get("/local/config/current")
        resp.raise_for_status()
        return resp.json()

    async def update_config(self, **kwargs) -> dict:
        resp = await self._client.post("/local/config/update", json=kwargs)
        resp.raise_for_status()
        return resp.json()

    # ── models ──

    async def get_models(self) -> list[dict]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        data = resp.json()
        return data.get("model_list", [])

    async def refresh_models(self) -> list[dict]:
        resp = await self._client.post("/models/refresh")
        resp.raise_for_status()
        data = resp.json()
        return data.get("model_list", [])

    # ── messages ──

    async def post_message(self, model: str, messages: list[dict], **kwargs) -> dict:
        body = {
            "model": model,
            "messages": {"context": messages},
            "stream": False,
            **kwargs,
        }
        resp = await self._client.post("/messages", json=body)
        resp.raise_for_status()
        return resp.json()

    async def post_message_stream(
        self, model: str, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        body = {
            "model": model,
            "messages": {"context": messages},
            "stream": True,
            **kwargs,
        }
        async with self._client.stream("POST", "/messages", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                yield data
