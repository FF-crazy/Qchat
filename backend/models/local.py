from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from backend.service.context import ContextManager


Role = Literal["system", "user", "assistant"]
PartType = Literal["text", "thinking"]


class ContentPart(BaseModel):
    type: PartType
    text: str
    signature: str | None = None


class CanonicalMessage(BaseModel):
    role: Role
    content: list[ContentPart]


class Provider(BaseModel):
    provider_name: str
    provider_type: Literal["openai", "gemini", "anthropic"]
    base_url: str
    api_key: str

    model_config = {
        "populate_by_name": True,
    }


class Prompt(BaseModel):
    prompt_name: str
    description: str = Field(default="")
    content: str


class Agent(BaseModel):
    prompt: Prompt | None = None
    sessions: list[ContextManager] | None = None

    @staticmethod
    def make_sessions(prompt: Prompt | None) -> list[ContextManager]:
        from backend.service.context import ContextManager

        context: list[CanonicalMessage] = []
        if prompt is not None:
            context.append(
                CanonicalMessage(
                    role="system",
                    content=[ContentPart(type="text", text=prompt.content)],
                )
            )
        return [ContextManager(context=context)]

    @model_validator(mode="after")
    def init_sessions(self) -> "Agent":
        if self.sessions is None:
            self.sessions = self.make_sessions(self.prompt)
        return self


type Context_type = Literal["text", "image"]


class QchatModelInfo(BaseModel):
    model_name: str
    support_context: list[Context_type]


class QchatModelList(BaseModel):
    model_list: list[QchatModelInfo]

class CurrentLocalConfig(BaseModel):
    provider: str | None = None
    agent: Agent | None = None
    stream: bool = False
    max_tokens: int = 65536
    temperature: float = 1.0
