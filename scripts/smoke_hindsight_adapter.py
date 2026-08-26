from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ax0s_bench.adapters import HindsightAdapter
from ax0s_bench.runner import canonical_hash, git_revision


class RetrievalOnlyClient:
    def chat(self, messages, *, seed_offset=0):
        return {"content": "retrieval-only smoke; common answer model intentionally not called", "usage": {}}


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    adapter = HindsightAdapter(
        RetrievalOnlyClient(),
        {
            "api_endpoint": "http://127.0.0.1:8889",
            "timeout_seconds": 3600,
            "bank_prefix": "hindsight-seeded-smoke-20260820",
            "reset_bank": True,
            "retain_batch_size": 4,
            "expected_package_version": "0.9.1",
            "require_bank_llm_health": True,
            "recall_budget": "low",
            "recall_max_tokens": 1024,
            "ingestion_log_path": "logs/hindsight-adapter-seeded-smoke-20260820.jsonl",
            "hindsight_repository": "https://github.com/vectorize-io/hindsight",
            "hindsight_commit": "e5b49eb6729512bd9b103058daa93f701da25644",
            "hindsight_package": "hindsight-api==0.9.1",
            "server_freeze_sha256": "b214a3027fd2e8858bcf10a8a7815accf13f3ec7ac6cc7f7a13f52f97e2e309a",
            "fully_local_claim": "self-hosted loopback API, embedded pg0, loopback llama.cpp, CPU embedding/reranker",
        },
        repo,
    )
    adapter.ingest_conversation(
        {
            "session_1_date_time": "1:00 pm on 10 May, 2024",
            "session_1": [
                {"speaker": "Alice", "text": "I moved to Paris today to start a job at Atelier Nine."},
                {"speaker": "Bob", "text": "Congratulations on your move to Paris and the new job!"},
            ],
        }
    )
    result = adapter.answer("Where did Alice move?", question_index=0)
    retrieved = result["retrieval"]["results"]
    if not retrieved:
        raise RuntimeError("Hindsight seeded smoke returned no memories")
    if not any("paris" in str(item.get("text", "")).lower() for item in retrieved):
        raise RuntimeError(f"Hindsight seeded smoke returned no Paris fact: {retrieved}")
    payload = {
        "schema_version": "hindsight-adapter-seeded-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_git_revision": git_revision(repo),
        "question": "Where did Alice move?",
        "retrieval": result["retrieval"],
        "adapter_metadata": adapter.receipt_metadata(),
        "answer_model_called": False,
        "judge_model_called": False,
    }
    payload["content_hash"] = {
        "algorithm": "sha256-canonical-json-excluding-content_hash",
        "value": canonical_hash(payload),
    }
    output = repo / "results/hindsight-adapter-seeded-smoke-20260820.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
