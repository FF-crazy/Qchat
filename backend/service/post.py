from ast import mod
import httpx
from pydantic import BaseModel

from backend.models.gemini import GeminiModelList, GeminiRequestWithModel, GeminiResponse
from backend.models.local import Provider
from backend.models.openai import (
  REASONING_EFFORT,
  OpenAIChunkResponse,
  OpenAIError,
  OpenAIModelList,
  OpenAIMessageRequest,
  OpenAIMessageResponse,
)
import certifi

from backend.service.context import ContextManager

OPENAI_V1_CHAT = "/v1/chat/completions"
OPENAI_V1_MODEL = "/v1/models"
GEMINI_V1BETA = "/v1beta/models"

class MessagePoster:
    def __init__(self, provider: Provider) -> None:
        self.HTTP_CLIENT = httpx.AsyncClient(timeout=None, verify=certifi.where())
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {provider.api_key}",
            "x-goog-api-key": f"{provider.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Qchat/1.0",
        }
        self.provider = provider

    def _raise_openai_error(self, exc: httpx.HTTPStatusError) -> None:
        try:
            err = OpenAIError.model_validate(exc.response.json())
            raise RuntimeError(f"OpenAI API error: {err.error.message}") from exc
        except Exception:
            raise exc

    async def openai_post(self, payload: OpenAIMessageRequest) -> OpenAIMessageResponse:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=f"{self.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.headers,
                json=payload.model_dump(),
            )
            response.raise_for_status()
            data = response.json()
            return OpenAIMessageResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def openai_post_stream(self, payload: OpenAIMessageRequest):
        stream_payload = payload.model_dump()
        stream_payload["stream"] = True
        try:
            async with self.HTTP_CLIENT.stream(
                "POST",
                url=f"{self.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.headers,
                json=stream_payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    yield OpenAIChunkResponse.model_validate_json(data)
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def openai_post_stream_text(self, payload: OpenAIMessageRequest):
        async for chunk in self.openai_post_stream(payload):
            for choice in chunk.choices:
                if choice.delta.content:
                    yield choice.delta.content

    async def openai_post_stream_text_collect(
        self, payload: OpenAIMessageRequest
    ) -> str:
        parts: list[str] = []
        async for text in self.openai_post_stream_text(payload):
            parts.append(text)
        return "".join(parts)

    async def openai_get_model_list(self) -> OpenAIModelList:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.get(
                url=f"{self.provider.base_url}{OPENAI_V1_MODEL}",
                headers=self.headers,
            )
            response.raise_for_status()
            return OpenAIModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def gemini_post(self, req: GeminiRequestWithModel) -> GeminiResponse:
        url: str = (
            f"{self.provider.base_url}{GEMINI_V1BETA}/{req.model}:generateContent"
        )
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=url,
                headers=self.headers,
                json=req.request_body.model_dump()
            )
            response.raise_for_status()
            data = response.json()
            return GeminiResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            raise e
    async def gemini_get_model_list(self) -> GeminiModelList:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.get(
                url=f"{self.provider.base_url}{GEMINI_V1BETA}",
                headers=self.headers,
                )
            response.raise_for_status()
            return GeminiModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise e


class RequestBuilder(BaseModel):
    context_manager: ContextManager

    def build_openai_request(
        self, model: str, stream: bool, reasoning_effort: REASONING_EFFORT
    ) -> OpenAIMessageRequest:
        req = OpenAIMessageRequest(
            model=model,
            reasoning_effort=reasoning_effort,
            messages=self.context_manager.context,
            stream=stream,
        )
        if req.stream:
            req.stream_options = {"include_usage": True}
        return req
