from collections.abc import Callable
from time import time
from typing import Literal

from backend.models.anthropic import AnthropicRequest, AnthropicResponse
from backend.models.anthropic import AnthropicModelList
from backend.models.gemini import GeminiConfig, GeminiRequest, GeminiRequestAddition, GeminiResponse
from backend.models.gemini import GeminiModelList
from backend.models.local import (
    Context_type,
    QchatModelInfo,
    QchatModelList,
    QchatRequest,
    QchatResponse,
)
from backend.models.openai import OpenAIMessageRequest, OpenAIMessageResponse, OpenAIModelList
from backend.service.context import ContextManager


class ModelListConverter:
    @classmethod
    def to_qchat_model_list(
        cls,
        model_list: OpenAIModelList | GeminiModelList | AnthropicModelList,
        support_context_resolver: Callable[[str], list[Context_type]] | None = None,
    ) -> QchatModelList:
        if support_context_resolver is None:
            resolver: Callable[[str], list[Context_type]] = cls._default_context_resolver
        else:
            resolver = support_context_resolver
        match model_list:
            case OpenAIModelList():
                return cls.from_openai(model_list, resolver)
            case GeminiModelList():
                return cls.from_gemini(model_list, resolver)
            case AnthropicModelList():
                return cls.from_anthropic(model_list, resolver)
            case _:
                raise ValueError(f"Unsupported model list type: {type(model_list)}")

    @staticmethod
    def _default_context_resolver(_name: str) -> list[Context_type]:
        return ["text"]

    @classmethod
    def from_openai(
        cls,
        model_list: OpenAIModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=model.id,
                    support_context=support_context_resolver(model.id),
                )
                for model in model_list.data
            ]
        )

    @classmethod
    def from_gemini(
        cls,
        model_list: GeminiModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=cls._normalize_gemini_name(model.name),
                    support_context=support_context_resolver(
                        cls._normalize_gemini_name(model.name)
                    ),
                )
                for model in model_list.models
            ]
        )

    @classmethod
    def from_anthropic(
        cls,
        model_list: AnthropicModelList,
        support_context_resolver: Callable[[str], list[Context_type]],
    ) -> QchatModelList:
        return QchatModelList(
            model_list=[
                QchatModelInfo(
                    model_name=model.id,
                    support_context=support_context_resolver(model.id),
                )
                for model in model_list.data
            ]
        )

    @staticmethod
    def _normalize_gemini_name(name: str) -> str:
        if name.startswith("models/"):
            return name[len("models/") :]
        return name


class MessageConverter:
    @staticmethod
    def to_provider_request(
        provider_type: Literal["openai", "gemini", "anthropic"],
        request: QchatRequest,
    ) -> OpenAIMessageRequest | GeminiRequestAddition | AnthropicRequest:
        match provider_type:
            case "openai":
                return OpenAIMessageRequest(
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    messages=request.messages.export(ContextManager.encode_openai),
                    stream=request.stream,
                    stream_options={"include_usage": True} if request.stream else None,
                )
            case "gemini":
                system_instruction, contents = request.messages.export(
                    ContextManager.encode_gemini
                )
                return GeminiRequestAddition(
                    model=request.model,
                    stream=request.stream,
                    request_body=GeminiRequest(
                        generationConfig=GeminiConfig(
                            maxOutputTokens=request.max_tokens,
                            temperature=request.temperature,
                        ),
                        contents=contents,
                        systemInstruction=system_instruction,
                    ),
                )
            case "anthropic":
                system_blocks, messages = request.messages.export(
                    ContextManager.encode_anthropic
                )
                return AnthropicRequest(
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=system_blocks or None,
                    messages=messages,
                    stream=request.stream,
                )
            case _:
                raise ValueError(f"Unsupported provider type: {provider_type}")

    @staticmethod
    def to_qchat_response(
        response: OpenAIMessageResponse | GeminiResponse | AnthropicResponse,
    ) -> QchatResponse:
        now = int(time())
        match response:
            case OpenAIMessageResponse():
                return QchatResponse(
                    id=response.id,
                    object=response.object,
                    created=response.created,
                    model=response.model,
                    choices=[choice.model_dump() for choice in response.choices],
                    usage=response.usage.model_dump(),
                )
            case GeminiResponse():
                return QchatResponse(
                    id=response.responseId or "",
                    object="gemini.response",
                    created=now,
                    model=response.modelVersion or "",
                    choices=[candidate.model_dump() for candidate in response.candidates],
                    usage=response.usageMetadata.model_dump(),
                )
            case AnthropicResponse():
                return QchatResponse(
                    id=response.id,
                    object=response.type,
                    created=now,
                    model=response.model,
                    choices=[content.model_dump() for content in response.content],
                    usage=response.usage.model_dump(),
                )
            case _:
                raise ValueError(f"Unsupported response type: {type(response)}")
