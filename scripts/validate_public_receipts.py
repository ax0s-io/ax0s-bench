#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ax0s_bench.runner import verify_hashed_json

FORBIDDEN_KEYS = {"question", "gold_answer", "evidence", "text", "content", "raw_response", "rationale", "hostname"}
EXPECTED_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def walk(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise AssertionError(f"forbidden public receipt key {path}.{key}")
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if "/home/" in value or "/Users/" in value:
            raise AssertionError(f"private path at {path}")
    yield path, value


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    result_schema = json.loads((repo / "schemas/redacted-receipt-v1.schema.json").read_text())
    calibration_schema = json.loads((repo / "schemas/redacted-calibration-v1.schema.json").read_text())
    failure_schema = json.loads((repo / "schemas/redacted-failure-receipt-v1.schema.json").read_text())
    for schema in (result_schema, calibration_schema, failure_schema):
        Draft202012Validator.check_schema(schema)
    receipts = sorted((repo / "receipts/redacted").glob("*.json"))
    assert {path.stem for path in receipts} == {"ax0s-a2700dd", "ax0s-bcd8710", "hindsight-0.9.1", "mem0-2.0.18"}
    for path in receipts:
        payload = json.loads(path.read_text())
        Draft202012Validator(result_schema).validate(payload)
        assert payload["schema_version"] == "ax0s-bench-redacted-receipt-v1"
        assert payload["reproduction_tier"] == 3
        assert payload["dataset"]["sha256"] == EXPECTED_DATASET_SHA256
        assert len(payload["questions"]) == payload["aggregate"]["questions"] == 1986
        assert all(row["judge"]["score"] in (0, 1) for row in payload["questions"])
        assert all(len(row["question_sha256"]) == len(row["gold_answer_sha256"]) == 64 for row in payload["questions"])
        linkage = payload["full_receipt_linkage"]
        assert len(linkage["raw_file_sha256"]) == 64
        assert len(linkage["canonical_content_sha256"]) == 64
        list(walk(payload))
        verify_hashed_json(path)
    failures = sorted((repo / "receipts/failures").glob("*.json"))
    assert len(failures) == 2
    for path in failures:
        payload = json.loads(path.read_text())
        Draft202012Validator(failure_schema).validate(payload)
        assert payload["schema_version"] == "ax0s-bench-redacted-failure-receipt-v1"
        assert payload["score_emitted"] is False
        assert len(payload["full_receipt_raw_sha256"]) == 64
        list(walk(payload))
        verify_hashed_json(path)
    calibrations = sorted((repo / "receipts/calibration").glob("*.json"))
    assert len(calibrations) == 4
    for path in calibrations:
        payload = json.loads(path.read_text())
        Draft202012Validator(calibration_schema).validate(payload)
        assert len(payload["decisions"]) == 40
        assert all(decision["secondary_judge"] is None for decision in payload["decisions"])
        list(walk(payload))
        verify_hashed_json(path)
    print(f"public receipts: OK ({len(receipts)} results, {len(calibrations)} calibration samples, {len(failures)} failures)")


if __name__ == "__main__":
    main()
