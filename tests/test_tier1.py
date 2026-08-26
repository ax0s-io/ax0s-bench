from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ax0s_bench.adapters import AX0S_RECALL_ENV_BINDINGS, bind_ax0s_recall_environment
from ax0s_bench.egress_guard import EgressGuard
from ax0s_bench.runner import canonical_hash, verify_hashed_json, verify_tier3_attestation, write_hashed_json


class Tier1UnitTest(unittest.TestCase):
    def test_canonical_hash_ignores_key_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_verifier_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "receipt.json"
            expected = write_hashed_json(path, {"schema_version": "test", "questions": []})
            self.assertEqual(expected, verify_hashed_json(path))
            payload = json.loads(path.read_text())
            payload["schema_version"] = "tampered"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                verify_hashed_json(path)


    def test_ax0s_recall_environment_is_exact(self):
        original = {env: os.environ.get(env) for env in AX0S_RECALL_ENV_BINDINGS.values()}
        condition = {key: "<unset>" for key in AX0S_RECALL_ENV_BINDINGS}
        condition["recall_env_ax0s_reranker"] = "cross-encoder"
        try:
            observed = bind_ax0s_recall_environment(condition)
            self.assertEqual("cross-encoder", observed["AX0S_RERANKER"])
            self.assertNotIn("AX0S_RERANKER_BATCH", os.environ)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                bind_ax0s_recall_environment({})
        finally:
            for env, value in original.items():
                if value is None:
                    os.environ.pop(env, None)
                else:
                    os.environ[env] = value

    def test_tier3_attestation_hashes_artifacts_and_hardware(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = Path(root) / "model.bin"
            artifact.write_bytes(b"frozen-model")
            env_name = "AX0S_BENCH_TEST_ARTIFACT"
            revision_name = "AX0S_BENCH_TEST_REVISION"
            old_artifact = os.environ.get(env_name)
            old_revision = os.environ.get(revision_name)
            os.environ[env_name] = str(artifact)
            os.environ[revision_name] = "revision-1"
            config = SimpleNamespace(
                attestation={
                    "required": True,
                    "reproduction_tier": 3,
                    "model_id": "local-model",
                    "llama_cpp_revision": "revision-1",
                    "llama_cpp_revision_env": revision_name,
                    "gpu_name": "Test GPU",
                    "gpu_memory_mib": 123,
                    "gpu_driver": "1.0",
                    "artifacts": {
                        "main": {
                            "env_var": env_name,
                            "bytes": artifact.stat().st_size,
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        }
                    },
                }
            )
            try:
                with patch("ax0s_bench.runner.subprocess.check_output", return_value="Test GPU, 123, 1.0\n"):
                    result = verify_tier3_attestation(config, {"data": [{"id": "local-model"}]})
                self.assertTrue(result["required"])
                self.assertEqual("redacted", result["artifacts"]["main"]["path"])
            finally:
                if old_artifact is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = old_artifact
                if old_revision is None:
                    os.environ.pop(revision_name, None)
                else:
                    os.environ[revision_name] = old_revision

    def test_loopback_policy(self):
        self.assertTrue(EgressGuard._is_loopback("127.0.0.1"))
        self.assertTrue(EgressGuard._is_loopback("::1"))
        self.assertFalse(EgressGuard._is_loopback("192.0.2.1"))


if __name__ == "__main__":
    unittest.main()
