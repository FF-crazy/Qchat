from fastapi import APIRouter, Depends

from backend.api.deps import get_config_manager
from backend.models.local import CurrentLocalConfig, QchatModelList
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
    return ModelListConverter.to_qchat_model_list(raw_model_list)


@chat_router.get("/config/current")
async def get_current_config(
    config_manager: ConfigManager = Depends(get_config_manager),
) -> CurrentLocalConfig:
    return config_manager.get_config()

@chat_router.post("/messages")
async def post_messages():
    pass
