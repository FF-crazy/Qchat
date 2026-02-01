from pydantic import BaseModel
from typing import Any, Literal

type REASONING_EFFORT = Literal["low", "medium", "high", "xhigh"]

class OpenAIMessage(BaseModel):
  role: str | None = None
  content: str | None = None
  refusal: str | None = None
  annotations: list[dict[str, Any]] | None = None
  model_config = {"extra": "allow"}

class OpenAIMessageRequest(BaseModel):
  model: str
  max_tokens: int = 64000
  temperature: float | None = None
  top_p: float| None = None
  reasoning_effort: REASONING_EFFORT | None = None
  messages: list[OpenAIMessage]
  stream: bool = False
  stream_options: dict[str, bool] | None = None

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

class OpenAIChunkChoice(BaseModel):
  """using stream"""
  index: int
  delta: OpenAIMessage
  finish_reason: str | None = None

class OpenAIChunkResponse(BaseModel):
  """using stream"""
  id: str
  object: str
  created: int
  model: str
  choices: list[OpenAIChunkChoice]
  usage: OpenAIUsage
  model_config = {"extra": "allow"}



class OpenAIMessageResponse(BaseModel):
  id: str
  object: str
  created: int
  model: str
  choices: list[OpenAIResponseBlock]
  usage: OpenAIUsage
  model_config = {"extra": "allow"}

class OpenAIErrorDetail(BaseModel):
  message: str
  type: str
  param: str | None = None
  code: str | None = None

class OpenAIError(BaseModel):
  error: OpenAIErrorDetail
