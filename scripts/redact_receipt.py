#!/usr/bin/env python3
"""Create a dataset-text-free public receipt from a private AX0S Bench receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DROP_KEYS = {
    "question", "gold_answer", "evidence", "text", "content", "raw_response",
    "rationale", "query", "prompt", "source_path", "store_path", "local_path",
    "path", "hostname", "host", "repo", "repository_path", "endpoint",
}
KEEP_HINTS = (
    "id", "score", "distance", "rank", "count", "total", "duration", "latency",
    "elapsed", "usage", "finish_reason", "model", "dimension", "top_k", "threshold",
    "ingestion", "failure", "rerank", "fusion", "revision", "version", "package",
    "adapter", "snapshot", "sha256", "temperature", "seed", "thinking", "max_tokens",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_metadata(value: Any, parent: str = "") -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lower = key.lower()
            if lower in DROP_KEYS or lower.endswith("_path") or "secret" in lower or "password" in lower or "token_value" in lower:
                continue
            cleaned = safe_metadata(item, lower)
            if cleaned not in ({}, [], None) and (any(hint in lower for hint in KEEP_HINTS) or isinstance(cleaned, (dict, list))):
                result[key] = cleaned
        return result
    if isinstance(value, list):
        cleaned = [safe_metadata(item, parent) for item in value]
        return [item for item in cleaned if item not in ({}, [], None)]
    if isinstance(value, (str, int, float, bool)) and any(hint in parent for hint in KEEP_HINTS):
        text = str(value)
        if "/home/" in text or "/Users/" in text or text.startswith("100."):
            return None
        return value
    return None


def redact(payload: dict[str, Any], *, label: str, full_raw_sha256: str, tier: int) -> dict[str, Any]:
    rows = []
    for row in payload.get("questions", []):
        metadata = safe_metadata(row.get("answer_metadata", {}))
        retrieval = metadata.get("retrieval", {}) if isinstance(metadata, dict) else {}
        for key in ("results", "hits", "memories"):
            values = retrieval.get(key)
            if isinstance(values, list) and len(values) > 20:
                retrieval[f"{key}_count_before_public_limit"] = len(values)
                retrieval[key] = values[:20]
        rows.append({
            "question_index": row["question_index"],
            "conversation_id": row["conversation_id"],
            "conversation_question_index": row.get("conversation_question_index"),
            "category": row.get("category"),
            "question_sha256": value_sha256(row.get("question")),
            "gold_answer_sha256": value_sha256(row.get("gold_answer")),
            "prediction": row.get("prediction"),
            "retrieval_and_timing_metadata": metadata,
            "judge": {"score": row["judge"]["score"], "verdict": row["judge"].get("verdict")},
        })

    full_content_hash = payload.get("content_hash", {}).get("value")
    public = {
        "schema_version": "ax0s-bench-redacted-receipt-v1",
        "label": label,
        "reproduction_tier": tier,
        "run_id": payload.get("run_id"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "harness_git_revision": payload.get("harness_git_revision"),
        "full_receipt_linkage": {
            "raw_file_sha256": full_raw_sha256,
            "canonical_content_sha256": full_content_hash,
            "availability": "Private full receipt available to auditors on request, subject to dataset licence review.",
        },
        "redaction": {
            "reason": "LoCoMo is CC BY-NC 4.0 and the full receipt reproduces dataset text.",
            "removed": ["question text", "gold answer text", "evidence text", "retrieved text", "judge rationale and raw response", "hostnames and paths", "retrieval entries after rank 20"],
            "value_hash_algorithm": "sha256 of canonical UTF-8 JSON value",
        },
        "dataset": {
            "name": payload.get("dataset", {}).get("name", "LoCoMo-10"),
            "revision": payload.get("dataset", {}).get("revision"),
            "sha256": payload.get("dataset", {}).get("sha256"),
            "selected_conversations": payload.get("dataset", {}).get("selected_conversations"),
        },
        "condition": safe_metadata(payload.get("condition", {})),
        "adapter_metadata": safe_metadata(payload.get("adapter_metadata", {})),
        "determinism": safe_metadata(payload.get("determinism", {})),
        "answer_model": safe_metadata(payload.get("answer_model", {})),
        "judge_model": safe_metadata(payload.get("judge_model", {})),
        "judge_prompt": safe_metadata(payload.get("judge_prompt", {})),
        "egress_guard": safe_metadata(payload.get("egress_guard", {})),
        "hardware": {"disclosure": "One NVIDIA GeForce RTX 5090 with 32,607 MiB; Linux x86_64. Host identity redacted."},
        "aggregate": payload.get("aggregate"),
        "questions": rows,
    }
    public["content_hash"] = {
        "algorithm": "sha256-canonical-json-excluding-content_hash",
        "value": value_sha256(public),
    }
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="private receipt path, or - for stdin")
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tier", type=int, default=3)
    parser.add_argument("--full-raw-sha256")
    args = parser.parse_args()

    if args.input == "-":
        payload = json.load(sys.stdin)
        if not args.full_raw_sha256:
            raise SystemExit("--full-raw-sha256 is required for stdin")
        full_raw_sha256 = args.full_raw_sha256
    else:
        input_path = Path(args.input)
        payload = json.loads(input_path.read_text())
        full_raw_sha256 = args.full_raw_sha256 or file_sha256(input_path)
    public = redact(payload, label=args.label, full_raw_sha256=full_raw_sha256, tier=args.tier)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
