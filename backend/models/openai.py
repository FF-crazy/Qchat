from pydantic import BaseModel
from typing import Any, Literal

type Reasoning_effort = Literal["low", "medium", "high", "xhigh"]
type Tool_choice = Literal["auto", "none", "required"] | dict[str, Any]

class OpenAIToolFunctionBlock(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    required: list[str] | None = None
    arguments: str | None = None
    model_config = {"extra": "allow"}

class OpenAITool(BaseModel):
    type: str = "function"
    function: OpenAIToolFunctionBlock

class OpenAIToolCallMessage(BaseModel):
    index: int | None = None
    id: str | None = None
    type: str | None = "function"
    function: OpenAIToolFunctionBlock | None = None
    model_config = {"extra": "allow"}

class OpenAIMessageBlock(BaseModel):
    """using stream and normal"""

    role: str | None = None
    tool_call_id : str | None = None
    content: str | None = None
    tool_calls: list[OpenAIToolCallMessage] | None = None
    refusal: str | None = None
    annotations: list[dict[str, Any]] | None = None
    model_config = {"extra": "allow"}


class OpenAIMessageRequest(BaseModel):
    model: str
    max_tokens: int = 64000
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: Reasoning_effort | None = None
    messages: list[OpenAIMessageBlock]
    stream: bool = False
    stream_options: dict[str, bool] | None = None
    tools: list[OpenAITool] | None = None
    tool_choice: Tool_choice | None = None


class OpenAIResponseBlock(BaseModel):
    index: int
    message: OpenAIMessageBlock
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
    delta: OpenAIMessageBlock
    finish_reason: str | None = None


class OpenAIChunkResponse(BaseModel):
    """using stream"""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChunkChoice]
    usage: OpenAIUsage | None = None
    model_config = {"extra": "allow"}

class OpenAIMessageResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIResponseBlock]
    usage: OpenAIUsage
    model_config = {"extra": "allow"}

class OpenAIModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class OpenAIModelList(BaseModel):
    data: list[OpenAIModelInfo]
    object: str = "list"
    success: bool = True


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIError(BaseModel):
    error: OpenAIErrorDetail
