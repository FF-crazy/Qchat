from typing import Any, Literal
from pydantic import BaseModel


class GeminiThinkingConfig(BaseModel):
    includeThoughts: bool = True
    thinkingLevel: Literal["low", "high"]


class GeminiConfig(BaseModel):
    maxOutputTokens: int = 65536
    temperature: float = 1
    topP: float = 1
    thinkingConfig: GeminiThinkingConfig | None = None


class GeminiMessagePart(BaseModel):
    text: str | None = None
    inline_data: dict[str, str] | None = None


class GeminiMessageBlock(BaseModel):
    role: str
    parts: list[GeminiMessagePart]


class GeminiRequest(BaseModel):
    generationConfig: GeminiConfig
    contents: list[GeminiMessageBlock]
    systemInstruction: list[GeminiMessagePart] | None = None


class GeminiRequestWithModel(BaseModel):
    """Google's style is to concatenate the model name into the URL. In order to unify the interface, I had to add a new class."""

    model: str
    request_body: GeminiRequest


class GeminiUsage(BaseModel):
    promptTokenCount: int
    candidatesTokenCount: int
    totalTokenCount: int
    thoughtsTokenCount: int | None = None
    promptTokensDetails: None | dict[str, Any] = None


class GeminiResponseBlock(BaseModel):
    content: GeminiMessageBlock
    finishReason: str | None = "STOP"
    index: int
    safetyRatings: list[str]


class GeminiResponse(BaseModel):
    candidates: list[GeminiResponseBlock]
    usageMetadata: GeminiUsage
    modelVersion: str | None = None
    responseId: str | None = None


class GeminiModelInfo(BaseModel):
    name: str
    baseModelId: str | None = None
    version: str | None = None
    displayName: str | None = None
    description: str | None = None
    inputTokenLimit: int | None = None
    outputTokenLimit: int | None = None
    supportedGenerationMethods: list[str] | None = None
    thinking: bool | None = None
    temperature: float | None = None
    maxTemperature: float | None = None
    topP: float | None = None
    topK: int | None = None


class GeminiModelList(BaseModel):
    models: list[GeminiModelInfo]
    nextPageToken: str | None = None
