"""Unified launcher: starts backend, waits for port, then runs TUI."""

import subprocess
import sys
import os
import signal
import atexit

BACKEND_PORT_PREFIX = "QCHAT_PORT:"


def start_backend() -> tuple[subprocess.Popen, int]:
    """Start the backend subprocess and return (process, port)."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    # Read lines until we get the port
    for line in proc.stdout:
        line = line.strip()
        if line.startswith(BACKEND_PORT_PREFIX):
            port = int(line[len(BACKEND_PORT_PREFIX):])
            return proc, port

    # If we get here, backend failed to start
    stderr = proc.stderr.read()
    proc.kill()
    print(f"Backend failed to start:\n{stderr}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    proc, port = start_backend()

    # Ensure backend is killed when frontend exits
    def cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    backend_url = f"http://127.0.0.1:{port}"

    from frontend.CLI.app import QchatApp

    app = QchatApp(backend_url=backend_url)
    app.run()


if __name__ == "__main__":
    main()
