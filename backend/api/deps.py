from fastapi import Request

from backend.service.local import ConfigManager


def get_config_manager(request: Request) -> ConfigManager:
    config_manager = getattr(request.app.state, "config_manager", None)
    if config_manager is None:
        raise RuntimeError("ConfigManager is not initialized")
    return config_manager
