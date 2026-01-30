from pydantic import BaseModel
from typing import Any

class AnthropicContentBlock(BaseModel):
    content_type: str
    text: str | None
    tool_use_id: str | None = None
    content: str | list[dict[str, Any]]
    content_id: str | None
    name: str | None
    content_input: dict[str, Any] | None

class AnthropicMessage(BaseModel):
    role: str
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]

class AnthropicMessageRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict[str, Any]] | None = None
    max_tokens: int
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    tools: list[AnthropicTool] | None = None
    stop_sequences: list[str] | None = None

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int