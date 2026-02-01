import httpx

from backend.models.local import ModelList, Provider
from backend.models.openai import (
  OpenAIError,
  OpenAIMessageRequest,
  OpenAIMessageResponse,
)
import certifi

OPENAI_V1_CHAT = "/v1/chat/completions"
OPENAI_V1_MODEL = "/v1/models"


class MessagePoster:
  def __init__(self, provider: Provider) -> None:
    self.HTTP_CLIENT = httpx.AsyncClient(timeout=None, verify=certifi.where())
    self.headers: dict[str, str] = {
      "Authorization": f"Bearer {provider.api_key}",
      "Content-Type": "application/json",
      "User-Agent": "Qchat/1.0",
  }
    self.provider = provider

  def _raise_openai_error(self, exc: httpx.HTTPStatusError) -> None:
    try:
      err = OpenAIError.model_validate(exc.response.json())
      raise RuntimeError(f"OpenAI API error: {err.error.message}") from exc
    except Exception:
      raise exc

  async def openai_post(self, payload: OpenAIMessageRequest) -> OpenAIMessageResponse:
    try:
      response = await self.HTTP_CLIENT.post(
        url=f"{self.provider.base_url}{OPENAI_V1_CHAT}",
        headers=self.headers,
        json=payload.model_dump(),
      )
      response.raise_for_status()
      data = response.json()
      return OpenAIMessageResponse.model_validate(data)
    except httpx.HTTPStatusError as e:
      self._raise_openai_error(e)
      raise RuntimeError() # never run this line
  async def openai_get_model_list(self) -> ModelList:
    try:
      response = await self.HTTP_CLIENT.get(
        url=f"{self.provider.base_url}{OPENAI_V1_MODEL}",
        headers=self.headers,
      )
      response.raise_for_status()
      return ModelList.model_validate(response.json())
    except httpx.HTTPStatusError as e:
      self._raise_openai_error(e)
      raise RuntimeError() # never run this line
