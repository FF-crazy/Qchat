from fastapi import FastAPI
import uvicorn
from random import randint
import sys
import signal
import socket
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from backend.api.chat import chat_router
from backend.api.local import local_router
from backend.models.local import CurrentLocalConfig, Provider
from backend.service.local import ConfigManager, FileProcessor, ProviderProcessor

API_VERSION: str = "v1"
PORT_READY_PREFIX: str = "QCHAT_PORT:"


@asynccontextmanager
async def init_api(app: FastAPI) -> AsyncIterator[None]:
    # Resolve forward references: inject ContextManager into models namespace
    from backend.service.context import ContextManager
    import backend.models.local as _models
    _models.ContextManager = ContextManager
    _models.Agent.model_rebuild()
    _models.CurrentLocalConfig.model_rebuild()
    _models.QchatRequest.model_rebuild()

    FileProcessor.init()
    providers: dict[str, Provider] = FileProcessor.load_providers()
    ProviderProcessor.set_providers(providers)
    current_provider: str | None = next(iter(providers), None)
    app.state.config_manager = ConfigManager(CurrentLocalConfig(provider=current_provider))
    app.include_router(chat_router)
    app.include_router(local_router)
    # Print port after server is fully ready so the launcher can connect
    print(f"{PORT_READY_PREFIX}{app.state.port}", flush=True)
    yield


app: FastAPI = FastAPI(title="Qchat backend API", version=API_VERSION, lifespan=init_api)


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def find_available_port(
    start: int = 10000, end: int = 65535, max_attempts: int = 100
) -> int:
    for _ in range(max_attempts):
        port = randint(start, end)
        if is_port_available(port):
            return port
    raise RuntimeError("Could not find an available port after multiple attempts.")


def signal_handler(signum, frame):
    print("\nexiting...")
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        port = find_available_port()
        app.state.port = port
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nexiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
