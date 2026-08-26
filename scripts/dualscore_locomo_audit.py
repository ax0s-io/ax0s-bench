from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ax0s_bench.api import OpenAIChatClient
from ax0s_bench.config import ModelConfig
from ax0s_bench.judge import LLMJudge
from ax0s_bench.runner import canonical_hash

AUDIT_COMMIT = "9493fb4b4af4256ed17a18e8fd0b3cfdeec29539"
AUDIT_URL = f"https://raw.githubusercontent.com/dial481/locomo-audit/{AUDIT_COMMIT}/errors.json"
AUDIT_SHA256 = "f298ef46263fada20688a1d672be37d4f488c6da3b19343eea55ae5c1b5fe55e"
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
SCORE_CORRUPTING = {"HALLUCINATION", "TEMPORAL_ERROR", "ATTRIBUTION_ERROR", "AMBIGUOUS", "INCOMPLETE"}
SOURCE_HASHES = {
    "ax0s": "819ac1af1588baae43cdcb7b7a9a57d5f67d5434194d8f4bae43b23295895d91",
    "mem0": "94d479b883a03c208714404ef6c2c22a363e18e03ea3a43f5ba2073ed81bec5a",
}
ID_RE = re.compile(r"^locomo_(\d+)_qa(\d+)$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_receipt(path: Path, expected_hash: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    recorded = payload.pop("content_hash")
    actual = canonical_hash(payload)
    if recorded != {"algorithm": "sha256-canonical-json-excluding-content_hash", "value": expected_hash}:
        raise RuntimeError(f"unexpected recorded content hash for {path}: {recorded}")
    if actual != expected_hash:
        raise RuntimeError(f"canonical content hash mismatch for {path}: {actual}")
    payload["content_hash"] = recorded
    return payload


def aggregate(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    by_category: dict[str, list[int]] = defaultdict(list)
    values: list[int] = []
    for row in rows:
        score = int(row[score_key])
        values.append(score)
        by_category[str(row["category"])].append(score)
    return {
        "questions": len(values),
        "correct": sum(values),
        "accuracy": sum(values) / len(values),
        "by_category": {
            key: {"questions": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
            for key, v in sorted(by_category.items())
        },
        "categories_1_4": {
            "questions": sum(len(v) for k, v in by_category.items() if k in {"1", "2", "3", "4"}),
            "correct": sum(sum(v) for k, v in by_category.items() if k in {"1", "2", "3", "4"}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ax0s", type=Path, required=True)
    parser.add_argument("--mem0", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--judge-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    dataset_bytes = args.dataset.read_bytes()
    if sha256(dataset_bytes) != DATASET_SHA256:
        raise RuntimeError("frozen LoCoMo dataset hash mismatch")
    dataset = json.loads(dataset_bytes)

    with urllib.request.urlopen(AUDIT_URL, timeout=30) as response:
        audit_bytes = response.read()
    if sha256(audit_bytes) != AUDIT_SHA256:
        raise RuntimeError("locomo-audit errors.json hash mismatch")
    corrections_all = json.loads(audit_bytes)
    corrections = [row for row in corrections_all if row["error_type"] in SCORE_CORRUPTING]
    if len(corrections) != 99 or len({row["question_id"] for row in corrections}) != 99:
        raise RuntimeError(f"expected 99 unique score-corrupting corrections, got {len(corrections)}")

    sources = {
        "ax0s": verify_receipt(args.ax0s, SOURCE_HASHES["ax0s"]),
        "mem0": verify_receipt(args.mem0, SOURCE_HASHES["mem0"]),
    }
    prompt_hash = sha256(args.judge_prompt.read_bytes())
    if prompt_hash != "d09585046cb8892ab86fdfae4b711829998500c2be98fc9e8845fdab110ca720":
        raise RuntimeError("judge prompt hash mismatch")

    first = sources["ax0s"]
    cfg = first["config"]["values"]["judge_model"]
    for system, receipt in sources.items():
        if receipt["config"]["values"]["judge_model"] != cfg:
            raise RuntimeError(f"{system}: judge configuration differs from AX0S")
        if receipt["judge_prompt"]["sha256"] != prompt_hash:
            raise RuntimeError(f"{system}: receipt judge prompt differs from frozen prompt")
        if receipt["judge_model"] != first["judge_model"]:
            raise RuntimeError(f"{system}: observed judge disclosure differs from AX0S")
    judge = LLMJudge(OpenAIChatClient(ModelConfig(**cfg)), args.judge_prompt)
    output_systems: dict[str, Any] = {}
    total_calls = 0

    for system, receipt in sources.items():
        rows = receipt["questions"]
        if len(rows) != 1986:
            raise RuntimeError(f"{system}: expected 1986 questions, got {len(rows)}")
        mapped: list[dict[str, Any]] = []
        corrected_by_global: dict[int, dict[str, Any]] = {}
        seen: set[int] = set()
        for correction in corrections:
            match = ID_RE.fullmatch(correction["question_id"])
            if not match:
                raise RuntimeError(f"bad audit ID: {correction['question_id']}")
            conversation_index, local_index = map(int, match.groups())
            sample = dataset[conversation_index]
            qa = sample["qa"][local_index]
            candidates = [
                row for row in rows
                if row["conversation_id"] == sample["sample_id"]
                and int(row["conversation_question_index"]) == local_index
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"{system}: {correction['question_id']} maps to {len(candidates)} receipt rows")
            row = candidates[0]
            global_index = int(row["question_index"])
            if global_index in seen:
                raise RuntimeError(f"{system}: duplicate mapped global index {global_index}")
            seen.add(global_index)
            if qa["question"] != correction["question"] or row["question"] != correction["question"]:
                raise RuntimeError(f"{system}: question mismatch for {correction['question_id']}")
            if int(row["category"]) != int(correction["category"]):
                raise RuntimeError(f"{system}: category mismatch for {correction['question_id']}")
            if str(row["gold_answer"]) != str(correction["golden_answer"]):
                raise RuntimeError(f"{system}: original gold mismatch for {correction['question_id']}")
            corrected_judge = judge.grade(
                row["question"], str(correction["correct_answer"]), row["prediction"], question_index=global_index
            )
            total_calls += 1
            item = {
                "audit_question_id": correction["question_id"],
                "question_index": global_index,
                "conversation_id": row["conversation_id"],
                "conversation_question_index": local_index,
                "category": row["category"],
                "question": row["question"],
                "prediction": row["prediction"],
                "gold_answer_original": row["gold_answer"],
                "gold_answer_corrected": correction["correct_answer"],
                "error_type": correction["error_type"],
                "reasoning": correction["reasoning"],
                "judge_original": row["judge"],
                "judge_corrected": corrected_judge,
            }
            mapped.append(item)
            corrected_by_global[global_index] = item
            print(f"[{total_calls}/198] {system} {correction['question_id']} {row['judge']['score']}->{corrected_judge['score']}", flush=True)
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        score_rows = []
        for row in rows:
            replacement = corrected_by_global.get(int(row["question_index"]))
            score_rows.append({
                "category": row["category"],
                "original": int(row["judge"]["score"]),
                "corrected": int(replacement["judge_corrected"]["score"]) if replacement else int(row["judge"]["score"]),
            })
        original = aggregate(score_rows, "original")
        corrected = aggregate(score_rows, "corrected")
        affected_original = aggregate(
            [{"category": item["category"], "score": item["judge_original"]["score"]} for item in mapped], "score"
        )
        affected_corrected = aggregate(
            [{"category": item["category"], "score": item["judge_corrected"]["score"]} for item in mapped], "score"
        )
        output_systems[system] = {
            "source_receipt": str({"ax0s": args.ax0s, "mem0": args.mem0}[system]),
            "source_receipt_content_hash": SOURCE_HASHES[system],
            "join": {"corrections": 99, "matched_receipt_rows": len(mapped), "unmatched": 0},
            "original": original,
            "corrected": corrected,
            "delta_accuracy": corrected["accuracy"] - original["accuracy"],
            "affected_99": {"original": affected_original, "corrected": affected_corrected},
            "corrections": mapped,
        }

    payload = {
        "schema_version": "locomo-audit-dualscore-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Judge-only replay of the 99 score-corrupting public audit corrections against frozen stored predictions; all other labels retained.",
        "system_rerun": False,
        "judge_calls": total_calls,
        "audit": {
            "repository": "https://github.com/dial481/locomo-audit",
            "commit": AUDIT_COMMIT,
            "errors_url": AUDIT_URL,
            "errors_sha256": AUDIT_SHA256,
            "all_issues": len(corrections_all),
            "score_corrupting_issues": len(corrections),
            "included_error_types": sorted(SCORE_CORRUPTING),
            "provenance_limitation": "Public independent audit linked by Penfield; not a Penfield-signed release. AMBIGUOUS cases remain rubric-dependent.",
        },
        "dataset": {"sha256": DATASET_SHA256, "questions": 1986},
        "judge": {
            "prompt_sha256": prompt_hash,
            "model": cfg,
            "seed_policy": "original global question index + 100000 + retry attempt",
            "serial": True,
            "inter_call_sleep_seconds": args.sleep_seconds,
        },
        "systems": output_systems,
    }
    payload["content_hash"] = {
        "algorithm": "sha256-canonical-json-excluding-content_hash",
        "value": canonical_hash(payload),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
