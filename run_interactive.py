#!/usr/bin/env python3
"""Serve the real model outputs in a local hover-to-inspect viewer."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import socket
import socketserver
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass


def free_port(preferred: int) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the interactive perception result viewer")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="run the three perception models again before opening the viewer",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.rerun:
        subprocess.run([sys.executable, str(ROOT / "run_demo.py")], cwd=ROOT, check=True)

    required = [
        ROOT / "outputs" / "object_positions.json",
        ROOT / "outputs" / "depth_colormap.png",
        ROOT / "interactive_viewer.html",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise SystemExit(f"Missing {names}. Run: uv run python run_demo.py")

    port = free_port(args.port)
    handler = lambda *h_args, **h_kwargs: QuietHandler(
        *h_args, directory=str(ROOT), **h_kwargs
    )
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as server:
        server.daemon_threads = True
        url = f"http://127.0.0.1:{port}/interactive_viewer.html"
        print(f"Interactive perception viewer: {url}")
        print("Press Control+C in this terminal to stop it.")
        if not args.no_browser:
            threading.Thread(
                target=lambda: (time.sleep(0.4), webbrowser.open(url)), daemon=True
            ).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer stopped.")


if __name__ == "__main__":
    main()
