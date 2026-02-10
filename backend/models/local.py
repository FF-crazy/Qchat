from pydantic import BaseModel, Field
from typing import Literal
from typing import Any

class Provider(BaseModel):
    provider_id: int = Field(default=0)
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
