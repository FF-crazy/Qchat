from pydantic import BaseModel
from typing import Any

class AnthropicContentBlock(BaseModel):
  type: str
  text: str

class AnthropicMessageBlock(BaseModel):
  role: str
  content: list[AnthropicContentBlock]

class AnthropicRequest(BaseModel):
  model: str
  max_tokens: int = 64000
  messages: list[AnthropicMessageBlock]

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

