

from pydantic import BaseModel

from backend.models.openai import OpenAIMessage, OpenAIMessageRequest, OpenAIUsage


class ContextManager(BaseModel):
  context: list[OpenAIMessage]
  usage: OpenAIUsage | None
  MAX_TOKEN: int

  def add_message(self, message: OpenAIMessage) -> None:
    self.context.append(message)
  
  def switch_context(self, req: OpenAIMessageRequest) -> None:
    self.context = req.messages

  def update_context(self, usage: OpenAIUsage) -> None:
    self.usage = usage
  
  def check_context_approach(self) -> bool:
    if self.usage is None:
      return False
    return self.MAX_TOKEN * 0.8 <= self.usage.total_tokens