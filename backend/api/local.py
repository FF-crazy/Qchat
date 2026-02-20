from fastapi import APIRouter, Depends

from backend.api.deps import get_config_manager
from backend.models.local import CurrentLocalConfig
from backend.service.local import ConfigManager, ProviderProcessor, FileProcessor

local_router = APIRouter(prefix="/local")

@local_router.get("/config/current")
async def get_current_config(
    config_manager = Depends(get_config_manager),
) -> dict:
    return config_manager.get_config_dict()

@local_router.post("/config/update")
async def update_config(
    config: CurrentLocalConfig,
    config_manager = Depends(get_config_manager)
) -> dict:
    config_manager.update_config(config)
    return config_manager.get_config_dict()


@local_router.post("/providers")
async def get_all_providers() -> list[str]:
    return list(ProviderProcessor.get_all_providers().keys())