from pathlib import Path
from pydantic import ValidationError
from typing import Any
import tomllib
import shutil
import logging

from backend.models.local import Prompt, Provider, QchatModelList, CurrentLocalConfig, Agent

logger: logging.Logger = logging.getLogger(__name__)

INIT_DIR: Path = Path(__file__).resolve().parent.parent / "init"


class FileProcessor:
    """
    FileProcessor processes providers, config and prompt and all file, instead of db.
    """
    QCHAT_DIR: Path = Path.home() / ".qchat"
    PROVIDER_FILE: Path = QCHAT_DIR / "provider.toml"
    CONFIG_FILE: Path = QCHAT_DIR / "config.toml"
    PROMPTS_DIR: Path = QCHAT_DIR / "prompts"

    @classmethod
    def init(cls):
        cls.QCHAT_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

        if not cls.CONFIG_FILE.exists():
            src = INIT_DIR / "config.toml"
            shutil.copy(src, cls.CONFIG_FILE)
            logger.info(f"Created default config: {cls.CONFIG_FILE}")

        if not cls.PROVIDER_FILE.exists():
            src = INIT_DIR / "provider.example.toml"
            shutil.copy(src, cls.PROVIDER_FILE)
            logger.info(f"Created example provider config: {cls.PROVIDER_FILE}")

        prompt_files = list(cls.PROMPTS_DIR.glob("*.toml"))
        if not prompt_files:
            src = INIT_DIR / "prompt.example.toml"
            shutil.copy(src, cls.PROMPTS_DIR / "prompt.example.toml")
            logger.info(f"Created example prompt: {cls.PROMPTS_DIR / 'prompt.example.toml'}")

    @classmethod
    def load_providers(cls) -> dict[str, Provider]:
        with open(cls.PROVIDER_FILE, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
            providers_raw = cls._require_list(data.get("provider", []), "provider")
            try:
                providers_list: list[Provider] = list(map(Provider.model_validate, providers_raw))
            except ValidationError as e:
                logger.error("Provider Format error, please check your Provider file")
                raise e
            if not providers_list:
                logger.warning("Provider File is empty. Please add new Provider")
            return {provider.provider_name: provider for provider in providers_list}

    @staticmethod
    def _require_list(value: Any, key: str) -> list[Any]:
        if isinstance(value, list):
            return value
        raise ValueError(f"Invalid config: {key} must be a list")

    @classmethod
    def load_prompts(cls) -> dict[str, Prompt]:
        prompts: dict[str, Prompt] = {}
        cls.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        for toml_path in sorted(cls.PROMPTS_DIR.glob("*.toml")):
            if not toml_path.is_file():
                continue
            with open(toml_path, "rb") as f:
                data: dict[str, Any] = tomllib.load(f)
            prompts_raw = cls._require_list(data.get("prompt", []), "prompt")
            try:
                prompt_list = list(map(Prompt.model_validate, prompts_raw))
            except ValidationError as e:
                logger.error(f"Prompt format error in file: {toml_path}")
                raise e
            for prompt in prompt_list:
                if prompt.prompt_name in prompts:
                    logger.warning(
                        f"Duplicate prompt_name {prompt.prompt_name} found in {toml_path}; overwriting"
                    )
                prompts[prompt.prompt_name] = prompt
        return prompts


class ProviderProcessor:
    providers: dict[str, Provider] = {}
    current: str | None = None
    _model_cache: dict[str, QchatModelList] = {}

    @classmethod
    def set_providers(cls, providers: dict[str, Provider]) -> None:
        cls.providers = providers
        cls._model_cache = {}
        cls.current = next(iter(providers), None)

    @classmethod
    def get_current(cls) -> tuple[str, Provider]:
        if not cls.providers or cls.current is None:
            raise KeyError("No providers available. Call set_providers() first.")
        return (cls.current, cls.providers[cls.current])

    @classmethod
    def get_provider(cls, name: str) -> Provider:
        provider = cls.providers.get(name)
        if provider is None:
            logger.error(f"Provider not found: {name}")
            raise KeyError(f"Provider not found: {name}")
        cls.current = name
        return provider

    @classmethod
    def get_all_providers(cls) -> dict[str, Provider]:
        return cls.providers

    @classmethod
    def set_current_provider(cls, name: str) -> Provider:
        if name not in cls.providers:
            logger.error(f"Provider not found: {name}")
            raise KeyError(f"Provider not found: {name}")
        cls.current = name
        return cls.providers[name]

    @classmethod
    def get_cached_model_list(cls, provider_name: str) -> QchatModelList | None:
        return cls._model_cache.get(provider_name)

    @classmethod
    def set_cached_model_list(cls, provider_name: str, model_list: QchatModelList) -> None:
        cls._model_cache[provider_name] = model_list

    @classmethod
    def invalidate_model_cache(cls, provider_name: str | None = None) -> None:
        if provider_name is None:
            cls._model_cache.clear()
        else:
            cls._model_cache.pop(provider_name, None)


class ConfigManager:
    def __init__(self, config: CurrentLocalConfig) -> None:
        self._config: CurrentLocalConfig = config

    def get_config(self) -> CurrentLocalConfig:
        return self._config

    def update_config(self, new_config: CurrentLocalConfig) -> None:
        self._config = new_config

    def update_provider(self, provider: str) -> None:
        self._config.provider = provider
        self._config.provider_type = ProviderProcessor.get_provider(provider).provider_type

    def get_config_dict(self) -> dict:
        return self._config.model_dump()
