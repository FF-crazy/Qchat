from fastapi import FastAPI
from fastapi.security import HTTPBearer
import uvicorn
from random import randint
import sys
import signal
import socket
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from backend.api.chat import chat_router
from backend.models.local import CurrentLocalConfig, Provider
from backend.service.local import ConfigManager, FileProcessor, ProviderProcessor

security = HTTPBearer(auto_error=False)
API_VERSION: str = "v1"

@asynccontextmanager
async def init_api(app: FastAPI) -> AsyncIterator[None]:
    FileProcessor.init()
    providers: dict[str, Provider] = FileProcessor.load_providers()
    ProviderProcessor.set_providers(providers)
    current_provider: str | None = next(iter(providers), None)
    app.state.config_manager = ConfigManager(CurrentLocalConfig(provider=current_provider))
    app.include_router(chat_router)
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
    raise RuntimeError(f"无法在 {max_attempts} 次尝试内找到可用端口")


def signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    print("\n正在退出...")
    sys.exit(0)


def main() -> None:
    # 注册信号处理器，支持 Ctrl+C 退出
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        port = find_available_port()
        print(f"服务启动在端口: {port}")
        uvicorn.run(app, host="127.0.0.1", port=port)
    except RuntimeError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n正在退出...")
        sys.exit(0)


if __name__ == "__main__":
    main()
