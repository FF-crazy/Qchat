from typing import Literal
import httpx
from pydantic import BaseModel

from backend.models.gemini import GeminiModelList, GeminiRequest, GeminiRequestWithModel, GeminiResponse
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

class RequestBuilder(BaseModel):
    context_manager: ContextManager
    provider: Provider
    header: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "Qchat/1.0",
    }

    def build_openai_header(self) -> dict[str, str]:
        ret: dict[str, str] = self.header.copy()
        ret["Authorization"] = f"Bearer {self.provider.api_key}"
        return ret
    
    def build_gemini_header(self) -> dict[str, str]:
        ret: dict[str, str] = self.header.copy()
        ret["x-goog-api-key"] = f"{self.provider.api_key}"
        return ret

    def build_openai_request_body(
        self, model: str, stream: bool, reasoning_effort: REASONING_EFFORT | None
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

    def build_gemini_request_body(
        self,
        model: str,
        stream: bool,
        reasoning_effort: Literal["low", "high"] | None,
    ) -> GeminiRequestWithModel:
        
        return GeminiRequestWithModel(model=model, request_body=GeminiRequest())

class MessagePoster:
    def __init__(
        self, provider: Provider, context_manager: ContextManager | None = None
    ) -> None:
        self.HTTP_CLIENT = httpx.AsyncClient(timeout=None, verify=certifi.where())
        # TODO: Modify this after ai_model completion
        if context_manager is None:
            context_manager = ContextManager(context=list(), MAX_TOKEN=64000)
        self.request_builder = RequestBuilder(
            context_manager=context_manager, provider=provider
        )

    def _raise_openai_error(self, exc: httpx.HTTPStatusError) -> None:
        try:
            err = OpenAIError.model_validate(exc.response.json())
            raise RuntimeError(f"OpenAI API error: {err.error.message}") from exc
        except Exception:
            raise exc

    async def openai_post(self, payload: OpenAIMessageRequest) -> OpenAIMessageResponse:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.request_builder.build_openai_header(),
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
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.request_builder.build_openai_header(),
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
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_MODEL}",
                headers=self.request_builder.build_openai_header(),
            )
            response.raise_for_status()
            print(response.json())
            return OpenAIModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def gemini_post(self, req: GeminiRequestWithModel) -> GeminiResponse:
        url: str = (
            f"{self.request_builder.provider.base_url}{GEMINI_V1BETA}/{req.model}:generateContent"
        )
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=url,
                headers=self.request_builder.build_gemini_header(),
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
                url=f"{self.request_builder.provider.base_url}{GEMINI_V1BETA}",
                headers=self.request_builder.build_gemini_header(),
                )
            response.raise_for_status()
            return GeminiModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise e
