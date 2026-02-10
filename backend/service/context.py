from typing import Any, Literal, cast
from collections.abc import Callable, Sequence
from pydantic import BaseModel

from backend.models.anthropic import (
    AnthropicContentBlock,
    AnthropicMessageBlock,
    AnthropicResponse,
    AnthropicStreamEvent,
)
from backend.models.gemini import GeminiMessageBlock, GeminiMessagePart, GeminiResponse
from backend.models.openai import (
    OpenAIChunkResponse,
    OpenAIMessageBlock,
    OpenAIMessageResponse,
)

Role = Literal["system", "user", "assistant"]
PartType = Literal["text", "thinking"]


class ContentPart(BaseModel):
    type: PartType
    text: str
    signature: str | None = None


class CanonicalMessage(BaseModel):
    role: Role
    content: list[ContentPart]


class ContextManager(BaseModel):
    context: list[CanonicalMessage]
    usage_total_tokens: int | None = None

    def add_message(self, message: CanonicalMessage) -> None:
        self.context.append(message)

    def switch_context(self, messages: list[CanonicalMessage]) -> None:
        self.context = messages

    def add_from_decoder(
        self, decoder: Callable[[Any], list[CanonicalMessage]], payload: Any
    ) -> None:
        self.context.extend(decoder(payload))

    def update_context(self, usage: Any) -> None:
        self.usage_total_tokens = self._extract_total_tokens(usage)

    def export(self, encoder: Callable[[Sequence[CanonicalMessage]], Any]) -> Any:
        return encoder(self.context)


    @staticmethod
    def _only_text(parts: Sequence[ContentPart]) -> list[str]:
        return [part.text for part in parts if part.type == "text"]

    @classmethod
    def encode_openai(
        cls, context: Sequence[CanonicalMessage]
    ) -> list[OpenAIMessageBlock]:
        messages: list[OpenAIMessageBlock] = []
        for message in context:
            text = "".join(cls._only_text(message.content))
            messages.append(OpenAIMessageBlock(role=message.role, content=text))
        return messages

    @staticmethod
    def encode_anthropic(
        context: Sequence[CanonicalMessage],
    ) -> tuple[list[AnthropicContentBlock], list[AnthropicMessageBlock]]:
        system_blocks: list[AnthropicContentBlock] = []
        messages: list[AnthropicMessageBlock] = []
        for message in context:
            blocks: list[AnthropicContentBlock] = []
            for part in message.content:
                if part.type == "text":
                    blocks.append(AnthropicContentBlock(type="text", text=part.text))
                else:
                    blocks.append(
                        AnthropicContentBlock(
                            type="thinking",
                            thinking=part.text,
                            signature=part.signature,
                        )
                    )
            if message.role == "system":
                system_blocks.extend(blocks)
            else:
                messages.append(AnthropicMessageBlock(role=message.role, content=blocks))
        return system_blocks, messages

    @classmethod
    def encode_gemini(
        cls, context: Sequence[CanonicalMessage]
    ) -> tuple[list[GeminiMessagePart] | None, list[GeminiMessageBlock]]:
        system_parts: list[GeminiMessagePart] = []
        contents: list[GeminiMessageBlock] = []
        for message in context:
            parts = [
                GeminiMessagePart(text=text)
                for text in cls._only_text(message.content)
            ]
            if message.role == "system":
                system_parts.extend(parts)
            else:
                contents.append(GeminiMessageBlock(role=message.role, parts=parts))
        return (system_parts or None), contents

    @staticmethod
    def decode_openai(response: OpenAIMessageResponse) -> list[CanonicalMessage]:
        messages: list[CanonicalMessage] = []
        for choice in response.choices:
            content = [
                ContentPart(type="text", text=choice.message.content or ""),
            ]
            role = ContextManager._normalize_role(choice.message.role)
            messages.append(CanonicalMessage(role=role, content=content))
        return messages

    @staticmethod
    def decode_openai_stream(chunk: OpenAIChunkResponse) -> list[CanonicalMessage]:
        messages: list[CanonicalMessage] = []
        for choice in chunk.choices:
            if choice.delta.content:
                messages.append(
                    CanonicalMessage(
                        role=ContextManager._normalize_role(choice.delta.role),
                        content=[ContentPart(type="text", text=choice.delta.content)],
                    )
                )
        return messages

    @staticmethod
    def decode_anthropic(response: AnthropicResponse) -> list[CanonicalMessage]:
        content: list[ContentPart] = []
        for block in response.content:
            if block.type == "text":
                content.append(ContentPart(type="text", text=block.text or ""))
            elif block.type == "thinking":
                content.append(
                    ContentPart(
                        type="thinking",
                        text=block.thinking or "",
                        signature=block.signature,
                    )
                )
        return [CanonicalMessage(role=ContextManager._normalize_role(response.role), content=content)]

    @staticmethod
    def decode_anthropic_stream(event: AnthropicStreamEvent) -> list[CanonicalMessage]:
        if event.type != "content_block_delta" or not event.delta:
            return []
        if event.delta.get("type") == "text_delta":
            text = event.delta.get("text") or ""
            if not text:
                return []
            return [
                CanonicalMessage(
                    role="assistant",
                    content=[ContentPart(type="text", text=text)],
                )
            ]
        if event.delta.get("type") == "thinking_delta":
            thinking = event.delta.get("thinking") or ""
            signature = event.delta.get("signature")
            if not thinking:
                return []
            return [
                CanonicalMessage(
                    role="assistant",
                    content=[
                        ContentPart(
                            type="thinking", text=thinking, signature=signature
                        )
                    ],
                )
            ]
        return []

    @staticmethod
    def decode_gemini(response: GeminiResponse) -> list[CanonicalMessage]:
        messages: list[CanonicalMessage] = []
        for candidate in response.candidates:
            parts = []
            for part in candidate.content.parts:
                if part.text:
                    parts.append(ContentPart(type="text", text=part.text))
            if parts:
                messages.append(
                    CanonicalMessage(
                        role=ContextManager._normalize_role(candidate.content.role),
                        content=parts,
                    )
                )
        return messages

    @staticmethod
    def _extract_total_tokens(usage: Any) -> int | None:
        if usage is None:
            return None
        if hasattr(usage, "total_tokens"):
            return getattr(usage, "total_tokens")
        if hasattr(usage, "totalTokenCount"):
            return getattr(usage, "totalTokenCount")
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            return getattr(usage, "input_tokens") + getattr(usage, "output_tokens")
        if isinstance(usage, dict):
            if "total_tokens" in usage:
                return usage["total_tokens"]
            if "totalTokenCount" in usage:
                return usage["totalTokenCount"]
            if "input_tokens" in usage and "output_tokens" in usage:
                return usage["input_tokens"] + usage["output_tokens"]
        return None

    @staticmethod
    def _normalize_role(role: str | None) -> Role:
        if role in ("system", "user", "assistant"):
            return cast(Role, role)
        return "assistant"
