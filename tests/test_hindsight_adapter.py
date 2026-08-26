from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ax0s_bench.adapters import HindsightAdapter


class _Client:
    def chat(self, messages, *, seed_offset=0):
        return {"content": "Alice moved to Paris", "usage": {}, "response_model": "fake"}


class _Handler(BaseHTTPRequestHandler):
    retained = []
    api_version = "0.9.1"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "healthy", "database": "connected"})
        elif self.path == "/version":
            self._json(200, {"api_version": type(self).api_version, "features": {"bank_llm_health": True}})
        elif self.path.endswith("/config"):
            self._json(200, {"llm_provider": "lmstudio", "embeddings_provider": "local"})
        elif self.path.endswith("/health/llm"):
            self._json(200, {"status": "healthy", "provider": "lmstudio"})
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self):
        type(self).retained = []
        self._json(200, {"success": True})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size)) if size else {}
        if self.path.endswith("/health/llm"):
            self._json(200, {"status": "healthy", "provider": "lmstudio"})
        elif self.path.endswith("/memories/recall"):
            self._json(200, {"results": [{"id": "fact-1", "text": "Alice moved to Paris in May 2024", "type": "world"}]})
        elif self.path.endswith("/memories"):
            type(self).retained.extend(body["items"])
            self._json(200, {"success": True, "bank_id": "test", "items_count": len(body["items"]), "async": False})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class HindsightAdapterTest(unittest.TestCase):
    def test_ingest_and_nonempty_recall(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                condition = {
                    "api_endpoint": f"http://127.0.0.1:{server.server_port}",
                    "timeout_seconds": 5,
                    "bank_prefix": "test-bank",
                    "reset_bank": True,
                    "retain_batch_size": 2,
                    "expected_package_version": "0.9.1",
                    "require_bank_llm_health": True,
                    "recall_budget": "low",
                    "recall_max_tokens": 512,
                    "ingestion_log_path": "logs/hindsight.jsonl",
                    "hindsight_repository": "https://github.com/vectorize-io/hindsight",
                    "hindsight_commit": "e5b49eb6729512bd9b103058daa93f701da25644",
                    "hindsight_package": "hindsight-api==0.9.1",
                    "server_freeze_sha256": "b214a3027fd2e8858bcf10a8a7815accf13f3ec7ac6cc7f7a13f52f97e2e309a",
                    "fully_local_claim": "test loopback server",
                }
                adapter = HindsightAdapter(_Client(), condition, Path(root))
                adapter.ingest_conversation(
                    {
                        "session_1_date_time": "1:00 pm on 10 May, 2024",
                        "session_1": [
                            {"speaker": "Alice", "text": "I moved to Paris."},
                            {"speaker": "Bob", "text": "Congratulations!"},
                            {"speaker": "Alice", "text": "My new office is near the river."},
                        ],
                    }
                )
                result = adapter.answer("Where did Alice move?", question_index=0)
                self.assertEqual(3, len(_Handler.retained))
                self.assertEqual("2024-05-10T13:00:00+00:00", _Handler.retained[0]["timestamp"])
                self.assertEqual("Alice moved to Paris", result["content"])
                self.assertEqual("Alice moved to Paris in May 2024", result["retrieval"]["results"][0]["text"])
                self.assertEqual(3, result["retrieval"]["ingestion_turns"])
                self.assertEqual(0, result["retrieval"]["ingestion_failures"])
                metadata = adapter.receipt_metadata()
                self.assertEqual(3, metadata["ingestion_turns"])
                self.assertEqual("lmstudio", metadata["server_bank_config"]["llm_provider"])
                _Handler.api_version = "9.9.9"
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    HindsightAdapter(_Client(), condition, Path(root))
                _Handler.api_version = "0.9.1"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
