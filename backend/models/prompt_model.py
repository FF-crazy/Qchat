from pydantic import BaseModel, Field


class Prompt(BaseModel):
    prompt_id: int = Field(default=0)
    prompt_name: str
    description: str = Field(default="")
    content: str
