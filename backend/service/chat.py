import httpx

from backend.models.local import Provider
from backend.models.openai import OpenAIMessageRequest, OpenAIMessageResponse
from ..models import *
import certifi

OPENAI_V1 = "/v1/chat/completions"

class MessagePoster:
  def __init__(self) -> None:
    self.HTTP_CLIENT = httpx.AsyncClient(timeout=None, verify=certifi.where())

  async def openai_post(self, provider: Provider, payload: OpenAIMessageRequest) -> OpenAIMessageResponse:
    headers = {
      "Authorization": f"Bearer {provider.api_key}",
      "Content-Type": "application/json",
      "User-Agent": "Qchat/1.0",
  }
    try:
      response = await self.HTTP_CLIENT.post(
        url=f"{provider.base_url}{OPENAI_V1}",
        headers=headers,
        json=payload.model_dump(),
      )
      response.raise_for_status()
      data = response.json()
      return OpenAIMessageResponse.model_validate(data)
    except httpx.HTTPStatusError as e:
      raise e
