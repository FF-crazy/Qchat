from pydantic import BaseModel
from typing import Any

class OpenAIMessage(BaseModel):
  role: str
  content: str
  refusal: str | None = None
  annotations: list[dict[str, Any]] | None = None
  model_config = {"extra": "allow"}

class OpenAIMessageRequest(BaseModel):
  model: str
  max_tokens: int = 64000
  messages: list[OpenAIMessage]

class OpenAIResponseBlock(BaseModel):
  index: int
  message: OpenAIMessage
  finish_reason: str = "stop"

class OpenAIUsage(BaseModel):
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  prompt_tokens_details: dict[str, int] | None = None
  completion_tokens_details: dict[str, int] | None = None

class OpenAIMessageResponse(BaseModel):
  id: str
  object: str
  created: int
  model: str
  choices: list[OpenAIResponseBlock]
  usage: OpenAIUsage
  model_config = {"extra": "allow"}
