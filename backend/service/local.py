from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Any
import tomllib
import logging

from backend.models.local import Provider

logger: logging.Logger = logging.getLogger(__name__)


class FileLoader:
    QCHAT_DIR: Path = Path.home() / ".config" / "Qchat"
    PROVIDER_FILE: Path = QCHAT_DIR / "provider.toml"

    @classmethod
    def init(cls):
        cls.QCHAT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROVIDER_FILE.touch(exist_ok=True)

    @classmethod
    def load_provider(cls) -> list[Provider]:
        with open(cls.PROVIDER_FILE, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
            providers_raw = cls._require_list(data.get("provider", []), "provider")
            try:
                providers = list(map(Provider.model_validate, providers_raw))
            except ValidationError as e:
                logger.error("Provider Format error, please check your Provider file")
                raise e
            if not providers:
                logger.warning("Provider File is empty. Please add new Provider")
            return providers

    @staticmethod
    def _require_list(value: Any, key: str) -> list[Any]:
        if isinstance(value, list):
            return value
        raise ValueError(f"Invalid provider config: {key} must be a list")
    

class ProviderProcessor:
    providers: list[Provider]
    current: int = 0

    @classmethod
    def set_providers(cls, providers: list[Provider]) -> None:
        cls.providers = providers
        cls.current = 0
    
    @classmethod
    def get_current(cls) -> tuple[int, Provider]:
      if not hasattr(cls, "providers") or not cls.providers:
        raise IndexError("No providers available. Call set_providers() first.")
      return (cls.current, cls.providers[cls.current])
    
    @classmethod
    def get_provider(cls, index: int) -> Provider:
        if index >= len(cls.providers) or index < 0:
            logger.error("Index out of Provider or invalid index")
            raise IndexError
        cls.current = index
        return cls.providers[index]


