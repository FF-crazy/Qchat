from pydantic import BaseModel, Field
from typing import Literal
from typing import Any

class Provider(BaseModel):
    provider_id: int = Field(default=0)
    provider_name: str
    provider_type: str = Field(alias="type")
    base_url: str
    api_key: str

    model_config = {
        "populate_by_name": True,
    }

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool
    temperature: float | None = None
    max_tokens: int | None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None
    stop: str | list[str] | None
