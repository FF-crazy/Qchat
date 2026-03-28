from typing import Any, Literal
from collections.abc import AsyncIterator, Callable
import json
import httpx
from pydantic import BaseModel
import certifi

from backend.models.anthropic import (
    AnthropicModelList,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicThinkingBlock,
    AnthropicToolCall,
)
from backend.models.gemini import (
    GeminiConfig,
    GeminiModelList,
    GeminiRequest,
    GeminiRequestAddition,
    GeminiResponse,
    GeminiThinkingConfig,
)
from backend.models.local import CanonicalMessage, Provider, QchatModelList, QchatRequest, QchatResponse
from backend.models.openai import (
    Reasoning_effort,
    OpenAIChunkResponse,
    OpenAIError,
    OpenAIModelList,
    OpenAIMessageRequest,
    OpenAIMessageResponse,
)

from backend.service.context import ContextManager
from backend.service.model_converter import MessageConverter, ModelListConverter

OPENAI_V1_CHAT = "/v1/chat/completions"
OPENAI_V1_MODEL = "/v1/models"
GEMINI_V1BETA = "/v1beta/models"
ANTHROPIC_V1_MESSAGE = "/v1/messages"
ANTHROPIC_V1_MODEL = "/v1/models"


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

    def build_anthropic_header(self) -> dict[str, str]:
        ret: dict[str, str] = self.header.copy()
        ret["x-api-key"] = f"{self.provider.api_key}"
        ret["anthropic-version"] = "2023-06-01"
        ret["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        return ret

    def build_header(self) -> dict[str, str]:
        provider_type = self.provider.provider_type
        match provider_type:
            case "openai":
                return self.build_openai_header()
            case "gemini":
                return self.build_gemini_header()
            case "anthropic":
                return self.build_anthropic_header()
            case _:
                raise ValueError(f"Unsupported provider type: {provider_type}")

    def build_openai_request_body(
        self,
        model: str,
        stream: bool,
        reasoning_effort: Reasoning_effort | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> OpenAIMessageRequest:
        messages = self.context_manager.export(ContextManager.encode_openai)
        req = OpenAIMessageRequest(
            model=model,
            reasoning_effort=reasoning_effort,
            messages=messages,
            stream=stream,
        )
        if temperature is not None:
            req.temperature = temperature
        if max_tokens is not None:
            req.max_tokens = max_tokens
        if req.stream:
            req.stream_options = {"include_usage": True}
        return req

    def build_gemini_request_body(
        self,
        model: str,
        stream: bool,
        reasoning_effort: Literal["low", "high"] | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GeminiRequestAddition:
        system_instruction, contents = self.context_manager.export(
            ContextManager.encode_gemini
        )
        config = GeminiConfig()
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.maxOutputTokens = max_tokens
        if reasoning_effort is not None:
            config.thinkingConfig = GeminiThinkingConfig(
                includeThoughts=True, thinkingLevel=reasoning_effort
            )
        return GeminiRequestAddition(
            model=model,
            stream=stream,
            request_body=GeminiRequest(
                generationConfig=config,
                contents=contents,
                systemInstruction=system_instruction,
            ),
        )

    def _calculate_reasoning_effort(self, reason_effort: Reasoning_effort, max_token: int) -> int:
        match reason_effort:
            case "xhigh":
                return int(max_token * 0.8)
            case "high":
                return int(max_token * 0.6)
            case "medium":
                return int(max_token * 0.4)
            case "low":
                return int(max_token * 0.2)

    def build_anthropic_request_body(
        self,
        model: str,
        stream: bool,
        reasoning_effort: Reasoning_effort | None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AnthropicRequest:
        system_blocks, messages = self.context_manager.export(
            ContextManager.encode_anthropic
        )
        thinking = None
        max_tokens = max_tokens or 64000
        if reasoning_effort is not None:
            thinking = AnthropicThinkingBlock(
                type="enabled",
                budget_tokens=self._calculate_reasoning_effort(reasoning_effort, max_tokens),
            )
        return AnthropicRequest(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            system=system_blocks or None,
            messages=messages,
            stream=stream,
        )




class MessagePoster:
    def __init__(
        self, provider: Provider, context_manager: ContextManager | None = None
    ) -> None:
        self.HTTP_CLIENT = httpx.AsyncClient(timeout=None, verify=certifi.where())
        # TODO: Modify this after ai_model completion
        if context_manager is None:
            context_manager = ContextManager(context=list())
        self.request_builder = RequestBuilder(
            context_manager=context_manager, provider=provider
        )

    def switch_context(self, context_manager: ContextManager) -> None:
        self.request_builder.context_manager = context_manager

    def switch_provider(self, provider: Provider) -> None:
        self.request_builder.provider = provider


    def _raise_openai_error(self, exc: httpx.HTTPStatusError) -> None:
        try:
            err = OpenAIError.model_validate(exc.response.json())
            raise RuntimeError(f"OpenAI API error: {err.error.message}") from exc
        except Exception:
            raise exc

    @staticmethod
    async def iter_sse_json_payloads(lines: AsyncIterator[str]) -> AsyncIterator[str]:
        data_lines: list[str] = []
        async for raw_line in lines:
            line = raw_line.rstrip("\r")
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines).strip()
                    if payload and payload != "[DONE]":
                        yield payload
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue

        if data_lines:
            payload = "\n".join(data_lines).strip()
            if payload and payload != "[DONE]":
                yield payload

    async def openai_post(self, payload: OpenAIMessageRequest) -> OpenAIMessageResponse:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.request_builder.build_header(),
                json=payload.model_dump(),
            )
            response.raise_for_status()
            data = response.json()
            return OpenAIMessageResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def openai_post_stream(self, payload: OpenAIMessageRequest):
        try:
            async with self.HTTP_CLIENT.stream(
                method="POST",
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_CHAT}",
                headers=self.request_builder.build_header(),
                json=payload.model_dump(),
                timeout=600,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data: str = line[6:]
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

    async def openai_get_model_list(self) -> OpenAIModelList:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.get(
                url=f"{self.request_builder.provider.base_url}{OPENAI_V1_MODEL}",
                headers=self.request_builder.build_header(),
            )
            response.raise_for_status()
            return OpenAIModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            self._raise_openai_error(e)
            raise RuntimeError()  # never run this line

    async def gemini_post(self, payload: GeminiRequestAddition) -> GeminiResponse:
        url: str = f"{self.request_builder.provider.base_url}{GEMINI_V1BETA}/{payload.model}:generateContent"
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=url,
                headers=self.request_builder.build_header(),
                json=payload.request_body.model_dump(),
            )
            response.raise_for_status()
            data = response.json()
            return GeminiResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            raise e

    async def gemini_post_stream(self, payload: GeminiRequestAddition):
        url = f"{self.request_builder.provider.base_url}{GEMINI_V1BETA}/{payload.model}:streamGenerateContent?alt=sse"
        try:
            async with self.HTTP_CLIENT.stream(
                method="POST",
                url=url,
                headers=self.request_builder.build_header(),
                json=payload.request_body.model_dump(),
                timeout=600,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    yield GeminiResponse.model_validate_json(data)
        except httpx.HTTPStatusError as e:
            raise e

    async def gemini_post_stream_text(self, payload: GeminiRequestAddition):
        async for chunk in self.gemini_post_stream(payload):
            for candidate in chunk.candidates:
                for part in candidate.content.parts:
                    if part.text:
                        yield part.text

    async def gemini_get_model_list(self) -> GeminiModelList:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.get(
                url=f"{self.request_builder.provider.base_url}{GEMINI_V1BETA}",
                headers=self.request_builder.build_header(),
            )
            response.raise_for_status()
            return GeminiModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise e

    async def anthropic_post(self, payload: AnthropicRequest) -> AnthropicResponse:
        try:
            response: httpx.Response = await self.HTTP_CLIENT.post(
                url=f"{self.request_builder.provider.base_url}{ANTHROPIC_V1_MESSAGE}",
                headers=self.request_builder.build_header(),
                json=payload.model_dump(),
            )
            response.raise_for_status()
            data = response.json()
            return AnthropicResponse.model_validate(data)
        except httpx.HTTPStatusError as e:
            raise e

    async def anthropic_post_stream(self, payload: AnthropicRequest):
        try:
            async with self.HTTP_CLIENT.stream(
                method="POST",
                url=f"{self.request_builder.provider.base_url}{ANTHROPIC_V1_MESSAGE}",
                headers=self.request_builder.build_header(),
                json=payload.model_dump(),
                timeout=600,
            ) as response:
                response.raise_for_status()
                async for payload_json in self.iter_sse_json_payloads(response.aiter_lines()):
                    yield AnthropicStreamEvent.model_validate_json(payload_json)
        except httpx.HTTPStatusError as e:
            raise e

    async def anthropic_post_stream_text(self, payload: AnthropicRequest):
        async for event in self.anthropic_post_stream(payload):
            if event.type == "content_block_delta":
                if event.delta and event.delta.get("type") == "text_delta":
                    text = event.delta.get("text")
                    if text:
                        yield text

    async def anthropic_post_stream_tool_calls(self, payload: AnthropicRequest):
        buffers: dict[int, AnthropicToolCall] = {}
        async for event in self.anthropic_post_stream(payload):
            tool_call = ContextManager.decode_anthropic_stream_tool_call(event, buffers)
            if tool_call is not None:
                yield tool_call

    async def anthropic_get_model_list(self) -> AnthropicModelList:
        try:
            response = await self.HTTP_CLIENT.get(
                url=f"{self.request_builder.provider.base_url}{ANTHROPIC_V1_MODEL}",
                headers=self.request_builder.build_header(),
            )
            response.raise_for_status()
            return AnthropicModelList.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise e

    async def get_model_list(self) -> OpenAIModelList | GeminiModelList | AnthropicModelList:
        provider_type = self.request_builder.provider.provider_type
        match provider_type:
            case "openai":
                return await self.openai_get_model_list()
            case "gemini":
                return await self.gemini_get_model_list()
            case "anthropic":
                return await self.anthropic_get_model_list()
            case _:
                raise ValueError(f"Unsupported provider type: {provider_type}")

    async def get_qchat_model_list(self) -> QchatModelList:
        raw_model_list = await self.get_model_list()
        return ModelListConverter.to_qchat_model_list(raw_model_list)

    async def _post_provider_message(self, payload: Any) -> Any:
        match payload:
            case OpenAIMessageRequest():
                if payload.stream:
                    return self.openai_post_stream(payload)
                return await self.openai_post(payload)
            case GeminiRequestAddition():
                if payload.stream:
                    return self.gemini_post_stream(payload)
                return await self.gemini_post(payload)
            case AnthropicRequest():
                if payload.stream:
                    return self.anthropic_post_stream(payload)
                return await self.anthropic_post(payload)
            case _:
                raise ValueError(f"Unsupported payload type: {type(payload)}")

    @staticmethod
    def _iter_text_parts(messages: list[CanonicalMessage]) -> list[str]:
        texts: list[str] = []
        for message in messages:
            for part in message.content:
                if part.type == "text" and part.text:
                    texts.append(part.text)
        return texts

    @staticmethod
    def _sse_delta(text: str) -> str:
        return f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"

    async def post_message(self, request: QchatRequest) -> QchatResponse:
        if request.stream:
            raise ValueError("For stream requests, call post_message_stream().")
        provider_request = MessageConverter.to_provider_request(
            self.request_builder.provider.provider_type,
            request,
        )
        response = await self._post_provider_message(provider_request)
        return MessageConverter.to_qchat_response(response)

    async def post_message_stream(self, request: QchatRequest) -> AsyncIterator[str]:
        if not request.stream:
            raise ValueError("post_message_stream() requires request.stream=True")

        provider_request = MessageConverter.to_provider_request(
            self.request_builder.provider.provider_type,
            request,
        )

        match provider_request:
            case OpenAIMessageRequest():
                async for chunk in self.openai_post_stream(provider_request):
                    for text in self._iter_text_parts(ContextManager.decode_openai_stream(chunk)):
                        yield self._sse_delta(text)
            case GeminiRequestAddition():
                async for chunk in self.gemini_post_stream(provider_request):
                    for text in self._iter_text_parts(ContextManager.decode_gemini(chunk)):
                        yield self._sse_delta(text)
            case AnthropicRequest():
                async for chunk in self.anthropic_post_stream(provider_request):
                    for text in self._iter_text_parts(ContextManager.decode_anthropic_stream(chunk)):
                        yield self._sse_delta(text)
            case _:
                raise ValueError(f"Unsupported request type: {type(provider_request)}")

        yield "data: [DONE]\n\n"
