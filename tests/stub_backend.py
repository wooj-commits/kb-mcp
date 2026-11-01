"""Minimal OpenAI-compatible stub for embeddings + chat (no GPU)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DIM = 64


def _vec(text: str) -> list[float]:
    """Deterministic pseudo-embedding from text tokens (bag-of-words hash)."""
    v = [0.0] * DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode()).digest()
        for i in range(DIM):
            # signed byte contribution
            b = digest[i % len(digest)]
            v[i] += (b / 127.5) - 1.0
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Keep test logs quiet; parent captures stdout
        sys_stderr_write = __import__("sys").stderr.write
        sys_stderr_write("[stub] " + (fmt % args) + "\n")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _send(self, code: int, body: dict[str, Any]):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "stub-embed", "object": "model"},
                        {"id": "stub-chat", "object": "model"},
                    ],
                },
            )
        self._send(404, {"error": "not_found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_json()
        if path.endswith("/embeddings"):
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            data = [
                {"object": "embedding", "index": i, "embedding": _vec(t)}
                for i, t in enumerate(inputs)
            ]
            return self._send(
                200,
                {"object": "list", "data": data, "model": body.get("model", "stub-embed")},
            )
        if path.endswith("/chat/completions"):
            messages = body.get("messages") or []
            user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            # Pull passage numbers present in the prompt and cite [1]
            answer = (
                "Based on the knowledge base, refunds take 5 to 7 business days [1]. "
                "I used only the provided passages."
            )
            if "could not find" in user.lower() or "(no passages" in user:
                answer = "I could not find it in the knowledge base."
            return self._send(
                200,
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "model": body.get("model", "stub-chat"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"<think>planning</think>\n{answer}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        self._send(404, {"error": "not_found", "path": path})


def serve(host: str = "127.0.0.1", port: int = 18080) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    httpd = serve(port=port)
    print(f"stub listening on http://127.0.0.1:{port}", flush=True)
    httpd.serve_forever()
