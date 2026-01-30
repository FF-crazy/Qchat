from fastapi import FastAPI
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from random import randint
import sys
import signal
import socket

app = FastAPI(title="Qchat backend API")
security = HTTPBearer(auto_error=False)


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
