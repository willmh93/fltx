#!/usr/bin/env python3
"""Serve the browser wasm benchmark harness and save generated outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "res" / "bench"


def resolve_output_path(raw_path: str) -> Path:
    pure = PurePosixPath(unquote(raw_path))
    if pure.is_absolute():
        pure = PurePosixPath(*pure.parts[1:])

    if ".." in pure.parts:
        raise ValueError("parent directory segments are not allowed")
    if len(pure.parts) < 3 or pure.parts[0] != "res" or pure.parts[1] != "bench":
        raise ValueError("path must be under res/bench")
    if pure.suffix.lower() not in {".csv", ".svg"}:
        raise ValueError("only csv and svg benchmark outputs can be saved")

    target = (ROOT / Path(*pure.parts)).resolve()
    if not target.is_relative_to(BENCH_DIR.resolve()):
        raise ValueError("path resolves outside res/bench")
    return target


class BrowserBenchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[browser-bench] {format % args}\n")

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/res/bench/browser_bench.html")
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/save-benchmark-file":
            self.save_benchmark_file()
            return
        if self.path == "/collect-table":
            self.collect_table()
            return
        self.send_error(404, "unknown endpoint")

    def read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value

    def save_benchmark_file(self) -> None:
        try:
            payload = self.read_json_body()
            path_value = payload.get("path")
            content_value = payload.get("content")
            if not isinstance(path_value, str) or not isinstance(content_value, str):
                raise ValueError("path and content must be strings")

            target = resolve_output_path(path_value)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content_value, encoding="utf-8", newline="")
            self.send_json({"ok": True, "path": str(target.relative_to(ROOT))})
        except Exception as exc:
            self.send_error(400, str(exc))

    def collect_table(self) -> None:
        try:
            result = subprocess.run(
                [sys.executable, str(BENCH_DIR / "collect_table.py"), "--format", "both"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.send_json({"ok": True, "stdout": result.stdout, "stderr": result.stderr})
        except subprocess.CalledProcessError as exc:
            self.send_error(500, (exc.stdout or "") + (exc.stderr or ""))

    def send_json(self, value: dict[str, object]) -> None:
        data = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), BrowserBenchHandler)
    print(f"Serving fltx browser benchmarks at http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
