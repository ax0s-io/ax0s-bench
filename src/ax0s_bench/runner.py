from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import make_adapter
from .api import OpenAIChatClient, probe_models
from .config import Config
from .dataset import load_dataset
from .egress_guard import EgressGuard
from .judge import LLMJudge


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tier3_attestation(config: Config, server_disclosure: dict[str, Any]) -> dict[str, Any]:
    expected = config.attestation
    if not expected.get("required", False):
        return {"required": False, "reproduction_tier": expected.get("reproduction_tier")}
    if int(expected.get("reproduction_tier", 0)) != 3:
        raise RuntimeError("required attestation is only supported for reproduction tier 3")

    observed_ids = {str(item.get("id")) for item in server_disclosure.get("data", []) if isinstance(item, dict)}
    required_model_id = str(expected["model_id"])
    if required_model_id not in observed_ids:
        raise RuntimeError(f"tier-3 model alias mismatch: expected {required_model_id}, observed {sorted(observed_ids)}")

    revision_env = str(expected["llama_cpp_revision_env"])
    observed_revision = os.environ.get(revision_env)
    if observed_revision != expected["llama_cpp_revision"]:
        raise RuntimeError(
            f"tier-3 llama.cpp revision mismatch: {revision_env}={observed_revision!r}, "
            f"expected {expected['llama_cpp_revision']!r}"
        )

    artifacts = {}
    for name, specification in expected["artifacts"].items():
        env_var = str(specification["env_var"])
        raw_path = os.environ.get(env_var)
        if not raw_path:
            raise RuntimeError(f"tier-3 attestation requires {env_var}")
        path = Path(raw_path)
        if not path.is_file():
            raise RuntimeError(f"tier-3 artifact from {env_var} is not a file")
        observed_size = path.stat().st_size
        observed_sha256 = sha256_file(path)
        expected_size = int(specification["bytes"]) if "bytes" in specification else None
        expected_sha256 = str(specification["sha256"])
        if (expected_size is not None and observed_size != expected_size) or observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"tier-3 artifact mismatch for {name}: size={observed_size}, sha256={observed_sha256}; "
                f"expected size={expected_size}, sha256={expected_sha256}"
            )
        artifacts[name] = {"bytes": observed_size, "sha256": observed_sha256, "path": "redacted"}

    gpu_line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    gpu_rows = [[part.strip() for part in row.split(",")] for row in gpu_line.splitlines() if row.strip()]
    expected_gpu = [str(expected["gpu_name"]), str(expected["gpu_memory_mib"]), str(expected["gpu_driver"])]
    if gpu_rows != [expected_gpu]:
        raise RuntimeError(f"tier-3 GPU mismatch: expected {[expected_gpu]}, observed {gpu_rows}")

    return {
        "required": True,
        "reproduction_tier": 3,
        "model_id": required_model_id,
        "llama_cpp_revision": observed_revision,
        "artifacts": artifacts,
        "gpu": {"name": gpu_rows[0][0], "memory_mib": int(gpu_rows[0][1]), "driver": gpu_rows[0][2]},
    }


def git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def hardware_disclosure() -> dict[str, Any]:
    gpu = "unavailable"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["judge"]["score"] for row in rows]
    by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row["judge"]["score"])
    return {
        "questions": len(rows),
        "correct": sum(scores),
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "by_category": {
            category: {"questions": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
            for category, values in sorted(by_category.items())
        },
    }


def write_hashed_json(path: Path, payload: dict[str, Any]) -> str:
    payload["content_hash"] = {"algorithm": "sha256-canonical-json-excluding-content_hash", "value": canonical_hash(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return payload["content_hash"]["value"]


def run(config: Config, repo: Path) -> tuple[Path, Path]:
    started = now_utc()
    random.seed(int(config.run["seed"]))
    dataset_path = repo / config.dataset["path"]
    samples = load_dataset(dataset_path, config.dataset["sha256"])
    selected_samples = samples[: int(config.run["max_conversations"])]

    egress_guard = EgressGuard()
    if config.run.get("local_only_egress", False):
        egress_guard.install()

    answer_client = OpenAIChatClient(config.answer_model)
    judge_client = OpenAIChatClient(config.judge_model)
    judge_prompt = repo / config.run["judge_prompt"]
    judge = LLMJudge(judge_client, judge_prompt)
    server_disclosure = probe_models(config.answer_model.endpoint)
    tier3_attestation = verify_tier3_attestation(config, server_disclosure)
    adapter = make_adapter(config.condition["adapter"], answer_client, config.condition, repo)

    rows: list[dict[str, Any]] = []
    global_index = 0
    for sample in selected_samples:
        sample_row_start = len(rows)
        adapter.ingest_conversation(sample["conversation"])
        questions = sample["qa"][: int(config.run["questions_per_conversation"])]
        for local_index, qa in enumerate(questions):
            prediction = adapter.answer(qa["question"], question_index=global_index)
            grade = judge.grade(
                qa["question"],
                str(qa.get("answer", qa.get("adversarial_answer"))),
                prediction["content"],
                question_index=global_index,
            )
            rows.append(
                {
                    "question_index": global_index,
                    "conversation_id": sample["sample_id"],
                    "conversation_question_index": local_index,
                    "category": qa.get("category"),
                    "question": qa["question"],
                    "gold_answer": qa.get("answer", qa.get("adversarial_answer")),
                    "evidence": qa.get("evidence", []),
                    "prediction": prediction["content"],
                    "answer_metadata": {key: value for key, value in prediction.items() if key != "content"},
                    "judge": grade,
                }
            )
            global_index += 1
            print(f'[{global_index}/{sum(min(len(s["qa"]), int(config.run["questions_per_conversation"])) for s in selected_samples)}] {grade["verdict"]}: {qa["question"]}', flush=True)

        if config.run.get("checkpoint_dir"):
            checkpoint_path = repo / config.run["checkpoint_dir"] / f'{sample["sample_id"]}.json'
            checkpoint_rows = rows[sample_row_start:]
            write_hashed_json(
                checkpoint_path,
                {
                    "schema_version": "ax0s-memory-bench-conversation-checkpoint-v1",
                    "run_id": config.run["id"],
                    "conversation_id": sample["sample_id"],
                    "finished_at": now_utc(),
                    "harness_git_revision": git_revision(repo),
                    "condition": config.condition,
                    "adapter_metadata": adapter.receipt_metadata(),
                    "aggregate": aggregate(checkpoint_rows),
                    "questions": checkpoint_rows,
                },
            )
            print(f"checkpoint: {checkpoint_path}", flush=True)

    receipt: dict[str, Any] = {
        "schema_version": "ax0s-memory-bench-receipt-v0",
        "run_id": config.run["id"],
        "started_at": started,
        "finished_at": now_utc(),
        "harness_git_revision": git_revision(repo),
        "config": {"path": str(config.path.relative_to(repo)), "sha256": config.raw_sha256, "values": config.raw},
        "dataset": {
            "name": config.dataset["name"],
            "source_url": config.dataset["source_url"],
            "revision": config.dataset["revision"],
            "sha256": config.dataset["sha256"],
            "local_path": config.dataset["path"],
            "selected_conversations": [sample["sample_id"] for sample in selected_samples],
        },
        "condition": config.condition,
        "adapter_metadata": adapter.receipt_metadata(),
        "determinism": {
            "runner_seed": config.run["seed"],
            "policy": "temperature zero plus per-question seed; exact GPU inference may still vary across builds/hardware",
        },
        "answer_model": answer_client.disclosure(),
        "judge_model": judge_client.disclosure(),
        "judge_prompt": {"path": config.run["judge_prompt"], "sha256": hashlib.sha256(judge_prompt.read_bytes()).hexdigest()},
        "server_models_probe": server_disclosure,
        "tier3_attestation": tier3_attestation,
        "egress_guard": egress_guard.disclosure(),
        "hardware": hardware_disclosure(),
        "aggregate": aggregate(rows),
        "questions": rows,
    }
    receipt_path = repo / config.run["result_path"]
    receipt_hash = write_hashed_json(receipt_path, receipt)

    sample_size = min(int(config.run["calibration_sample_size"]), len(rows))
    indices = sorted(random.Random(int(config.run["seed"])).sample(range(len(rows)), sample_size))
    calibration: dict[str, Any] = {
        "schema_version": "ax0s-memory-bench-judge-calibration-v0",
        "purpose": "Deterministic sample for later blind cross-check by a secondary judge; no secondary vendor was called in v0.",
        "source_receipt": str(receipt_path.relative_to(repo)),
        "source_receipt_content_hash": receipt_hash,
        "sampling": {"method": "python-random-sample-with-sorted-output", "seed": config.run["seed"], "size": sample_size},
        "judge_model": judge_client.disclosure(),
        "decisions": [
            {
                "question_index": rows[index]["question_index"],
                "conversation_id": rows[index]["conversation_id"],
                "question": rows[index]["question"],
                "gold_answer": rows[index]["gold_answer"],
                "prediction": rows[index]["prediction"],
                "judge": rows[index]["judge"],
                "secondary_judge": None,
            }
            for index in indices
        ],
    }
    calibration_path = repo / config.run["calibration_path"]
    write_hashed_json(calibration_path, calibration)
    return receipt_path, calibration_path



def verify_hashed_json(path: Path) -> str:
    payload = json.loads(path.read_text())
    recorded = payload.pop("content_hash")
    actual = canonical_hash(payload)
    if recorded.get("algorithm") != "sha256-canonical-json-excluding-content_hash":
        raise RuntimeError(f"unsupported content hash algorithm in {path}")
    if actual != recorded.get("value"):
        raise RuntimeError(f"content hash mismatch in {path}: expected {recorded.get('value')}, got {actual}")
    return actual
