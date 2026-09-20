#!/usr/bin/env python3
"""Serve a dependency-free local UI for inspecting method-run JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = PROJECT_ROOT / "method_viewer"
RUNS_ROOT = PROJECT_ROOT / "outputs" / "method_runs"


def available_runs() -> list[dict]:
    rows = []
    if not RUNS_ROOT.exists():
        return rows
    for path in RUNS_ROOT.glob("*.jsonl"):
        if path.name.endswith(".events.jsonl"):
            continue
        stat = path.stat()
        rows.append({"name": path.name, "size": stat.st_size, "modified": stat.st_mtime})
    return sorted(rows, key=lambda row: row["modified"], reverse=True)


def resolve_run(name: str) -> Path:
    if Path(name).name != name or not name.endswith(".jsonl") or name.endswith(".events.jsonl"):
        raise ValueError("Invalid run filename")
    path = RUNS_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(name)
    return path


class ViewerHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"viewer: {format % args}")

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self.send_json(available_runs())
            return
        if parsed.path == "/api/run":
            try:
                name = parse_qs(parsed.query).get("file", [""])[0]
                path = resolve_run(name)
                samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.send_json({"name": name, "samples": samples})
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)
            return

        relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        path = (VIEWER_ROOT / relative).resolve()
        if VIEWER_ROOT.resolve() not in path.parents and path != VIEWER_ROOT.resolve():
            self.send_error(403)
            return
        if not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse method-run JSONL files in a local web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Method Run Viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Reading runs from: {RUNS_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
