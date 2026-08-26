from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class LocalEmbeddingService:
    """CPU-only fastembed embedding service.

    Exposed two ways: in-process (``embed``) and, for backends that only accept an
    OpenAI-compatible embeddings URL, over a loopback HTTP server bound to
    127.0.0.1 on an ephemeral port. No weights are downloaded at request time and
    no request ever leaves the host.
    """

    def __init__(self, model_name: str, expected_dimensions: int, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.expected_dimensions = int(expected_dimensions)
        self._embedder = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self._lock = threading.Lock()
        self.request_count = 0
        self.text_count = 0
        probe = self.embed(["dimension probe"])
        self.dimensions = len(probe[0])
        if self.dimensions != self.expected_dimensions:
            raise RuntimeError(
                f"embedding dimension mismatch for {model_name}: "
                f"config declares {self.expected_dimensions}, model produced {self.dimensions}"
            )
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.request_count += 1
            self.text_count += len(texts)
            return [[float(value) for value in vector] for vector in self._embedder.embed(list(texts))]

    def start_http(self) -> str:
        if self._server is not None:
            return self.base_url
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence per-request logging
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": service.model_name, "object": "model", "owned_by": "fastembed-local"}],
                        },
                    )
                    return
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})

            def do_POST(self) -> None:
                if not self.path.rstrip("/").endswith("/embeddings"):
                    self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                supplied = request.get("input", [])
                texts = [supplied] if isinstance(supplied, str) else list(supplied)
                try:
                    vectors = service.embed([str(text) for text in texts])
                except Exception as exc:  # surface as an OpenAI-shaped error
                    self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                    return
                self._send(
                    200,
                    {
                        "object": "list",
                        "model": service.model_name,
                        "data": [
                            {"object": "embedding", "index": index, "embedding": vector}
                            for index, vector in enumerate(vectors)
                        ],
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    },
                )

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="local-embeddings")
        self._thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("start_http must be called before base_url")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def stop_http(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def disclosure(self) -> dict[str, Any]:
        return {
            "provider": "fastembed (ONNX Runtime, CPU)",
            "model": self.model_name,
            "dimensions": self.dimensions,
            "transport": "in-process plus loopback OpenAI-compatible /v1/embeddings",
            "requests": self.request_count,
            "texts": self.text_count,
        }
