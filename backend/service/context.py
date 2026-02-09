from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel

from backend.models.anthropic import AnthropicContentBlock, AnthropicMessageBlock
from backend.models.gemini import GeminiMessageBlock, GeminiMessagePart
from backend.models.openai import OpenAIMessageBlock

Role = Literal["system", "user", "assistant"]
PartType = Literal["text", "thinking"]


class ContentPart(BaseModel):
    type: PartType
    text: str
    signature: str | None = None


class CanonicalMessage(BaseModel):
    role: Role
    content: list[ContentPart]


def _only_text(parts: Sequence[ContentPart]) -> list[str]:
    return [part.text for part in parts if part.type == "text"]


def encode_openai(context: Sequence[CanonicalMessage]) -> list[OpenAIMessageBlock]:
    messages: list[OpenAIMessageBlock] = []
    for message in context:
        text = "".join(_only_text(message.content))
        messages.append(OpenAIMessageBlock(role=message.role, content=text))
    return messages


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


def encode_gemini(
    context: Sequence[CanonicalMessage],
) -> tuple[list[GeminiMessagePart] | None, list[GeminiMessageBlock]]:
    system_parts: list[GeminiMessagePart] = []
    contents: list[GeminiMessageBlock] = []
    for message in context:
        parts = [GeminiMessagePart(text=text) for text in _only_text(message.content)]
        if message.role == "system":
            system_parts.extend(parts)
        else:
            contents.append(GeminiMessageBlock(role=message.role, parts=parts))
    return (system_parts or None), contents


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


class ContextManager(BaseModel):
    context: list[CanonicalMessage]
    usage_total_tokens: int | None = None
    MAX_TOKEN: int

    def add_message(self, message: CanonicalMessage) -> None:
        self.context.append(message)

    def switch_context(self, messages: list[CanonicalMessage]) -> None:
        self.context = messages

    def update_context(self, usage: Any) -> None:
        self.usage_total_tokens = _extract_total_tokens(usage)

    def export(self, encoder: Callable[[Sequence[CanonicalMessage]], Any]) -> Any:
        return encoder(self.context)

    def check_context_approach(self) -> bool:
        if self.usage_total_tokens is None:
            return False
        return self.MAX_TOKEN * 0.8 <= self.usage_total_tokens
