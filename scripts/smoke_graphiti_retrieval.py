from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ax0s_bench.adapters import GraphitiAdapter
from ax0s_bench.config import load_config
from ax0s_bench.runner import canonical_hash, git_revision


class RetrievalOnlyClient:
    def chat(self, messages, *, seed_offset=0):
        return {"content": "retrieval-only smoke; common answer model intentionally not called", "usage": {}}


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    condition = dict(load_config(repo / "configs/locomo-graphiti-smoke.toml").condition)
    condition.update(
        {
            "reset_store": True,
            "store_path": ".bench/graphiti-seeded-smoke-20260820/falkordb.rdb",
            "group_prefix": "graphiti-seeded-smoke-20260820",
            "ingestion_log_path": "logs/graphiti-seeded-smoke-20260820.jsonl",
            "max_sessions": 1,
            "max_turns_per_session": 1,
        }
    )
    adapter = GraphitiAdapter(RetrievalOnlyClient(), condition, repo)
    adapter.ingest_conversation(
        {
            "session_1_date_time": "1:00 pm on 10 May, 2024",
            "session_1": [
                {"speaker": "Alice", "text": "I moved to Paris today to start a job at Atelier Nine."},
            ],
        }
    )
    result = adapter.answer("Where did Alice move and why?", question_index=0)
    retrieval = result["retrieval"]
    if not retrieval["edges"] and not retrieval["nodes"]:
        raise RuntimeError("Graphiti seeded smoke returned no graph facts or entities")
    facts = [str(edge.get("fact", "")) for edge in retrieval["edges"]]
    if not any("paris" in fact.lower() for fact in facts):
        raise RuntimeError(f"Graphiti seeded smoke returned no Paris fact: {facts}")
    payload = {
        "schema_version": "graphiti-adapter-seeded-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_git_revision": git_revision(repo),
        "question": "Where did Alice move and why?",
        "retrieval": retrieval,
        "adapter_metadata": adapter.receipt_metadata(),
        "answer_model_called": False,
        "judge_model_called": False,
    }
    payload["content_hash"] = {
        "algorithm": "sha256-canonical-json-excluding-content_hash",
        "value": canonical_hash(payload),
    }
    output = repo / "results/graphiti-adapter-seeded-smoke-20260820.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
