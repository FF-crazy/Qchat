from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.deps import get_config_manager
from backend.models.local import CurrentLocalConfig, Provider, QchatModelList, QchatRequest, QchatResponse
from backend.service.local import ConfigManager, ProviderProcessor
from backend.service.model_converter import ModelListConverter
from backend.service.post import MessagePoster

chat_router = APIRouter()


@chat_router.get("/models")
async def get_models(
    config_manager: ConfigManager = Depends(get_config_manager)) -> QchatModelList:
    config: CurrentLocalConfig = config_manager.get_config()
    if config.provider is None:
        return QchatModelList(model_list=[])
    poster = MessagePoster(provider=ProviderProcessor.get_provider(config.provider))
    raw_model_list = await poster.get_model_list()
    #TODO: cache model list in provider processor to avoid repeated API calls
    return ModelListConverter.to_qchat_model_list(raw_model_list)



@chat_router.post("/messages")
async def post_messages(
    request: QchatRequest,
    config_manager: ConfigManager = Depends(get_config_manager),
) -> QchatResponse | StreamingResponse:
    config: CurrentLocalConfig = config_manager.get_config()
    if config.provider is None:
        raise HTTPException(status_code=400, detail="No provider selected in config")

    provider: Provider = ProviderProcessor.get_provider(config.provider)
    poster = MessagePoster(provider=provider)

    if request.stream:
        return StreamingResponse(
            poster.post_message_stream(request),
            media_type="text/event-stream",
        )

    return await poster.post_message(request)
