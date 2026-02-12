from pydantic import BaseModel, Field, model_validator
from typing import Literal

from backend.service.context import CanonicalMessage, ContentPart, ContextManager

class Provider(BaseModel):
    provider_name: str
    provider_type: Literal["openai", "gemini", "anthropic"]
    base_url: str
    api_key: str

    model_config = {
        "populate_by_name": True,
    }

class Prompt(BaseModel):
    prompt_id: int = Field(default=0)
    prompt_name: str
    description: str = Field(default="")
    content: str

class Agent(BaseModel):
    prompt: Prompt | None = None
    sessions: list[ContextManager] | None = None

    @staticmethod
    def make_sessions(prompt: Prompt | None) -> list[ContextManager]:
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
