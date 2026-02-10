from pydantic import BaseModel
from typing import Any, Literal

class AnthropicContentBlock(BaseModel):
  type: str
  thinking: str | None = None
  signature: str | None = None
  text: str | None = None
  model_config = {"extra": "allow"}

class AnthropicMessageBlock(BaseModel):
  role: str
  content: list[AnthropicContentBlock]

class AnthropicThinkingBlock(BaseModel):
    type: Literal["enabled", "adaptive"] = "enabled" # adaptive just for opus 4.6
    budget_tokens: int | None = None

class AnthropicRequest(BaseModel):
  model: str
  max_tokens: int = 64000
  temperature: float | None = None
  thinking: AnthropicThinkingBlock | None = None
  system: list[AnthropicContentBlock] | None = None
  messages: list[AnthropicMessageBlock]
  stream: bool = False

class AnthropicUsage(BaseModel):
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int

class AnthropicResponse(BaseModel):
    model: str
    id: str
    type: str
    role: str
    content: list[AnthropicContentBlock]
    stop_reason: str = "end_turn"
    stop_sequence: None = None
    usage: AnthropicUsage

class AnthropicStreamEvent(BaseModel):
    type: Literal[
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
        "ping",
    ]
    index: int | None = None
    message: dict[str, Any] | None = None
    content_block: dict[str, Any] | None = None
    delta: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    model_config = {"extra": "allow"}

class AnthropicModelInfo(BaseModel):
  id: str
  created_at: str
  display_name: str
  type: str = "model"

class AnthropicModelList(BaseModel):
  data: list[AnthropicModelInfo]
  first_id: str
  has_more: bool = False
  last_id: str
