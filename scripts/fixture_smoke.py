#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ax0s_bench.adapters import chronological_sessions, locomo_turn_text, parse_locomo_datetime
from ax0s_bench.egress_guard import EgressGuard
from ax0s_bench.runner import aggregate, verify_hashed_json, write_hashed_json


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    fixture = json.loads((repo / "tests/fixtures/tier1_smoke.json").read_text())
    assert fixture["notice"].startswith("Synthetic fixture")
    assert len(fixture["conversation"]) == 3
    assert len(fixture["questions"]) == 8

    rows = [
        {
            "question_index": item["id"],
            "conversation_id": "synthetic-1",
            "category": item["category"],
            "prediction": item["prediction"],
            "judge": {"score": item["score"], "verdict": "correct" if item["score"] else "incorrect"},
        }
        for item in fixture["questions"]
    ]
    assert all(row["judge"]["score"] in (0, 1) for row in rows)
    assert EgressGuard._is_loopback("127.0.0.1")
    assert EgressGuard._is_loopback("::1")
    assert not EgressGuard._is_loopback("192.0.2.1")
    assert parse_locomo_datetime("1:56 pm on 8 May, 2023").isoformat() == "2023-05-08T13:56:00+00:00"
    assert locomo_turn_text({"text": "plain"}) == "plain"
    conversation = {
        "session_1": [{"speaker": "A", "text": "one"}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_2": [{"speaker": "B", "text": "two"}],
        "session_2_date_time": "9:00 am on 1 May, 2023",
    }
    assert [number for number, _, _ in chronological_sessions(conversation)] == [2, 1]

    receipt = {
        "schema_version": "ax0s-bench-tier1-smoke-receipt-v1",
        "reproduction_tier": 1,
        "dataset": "synthetic fixture only",
        "models": "stub answerer and stub judge",
        "egress_guard": {"installed": True, "loopback_endpoints": ["127.0.0.1:1"], "blocked_endpoints": []},
        "aggregate": aggregate(rows),
        "questions": rows,
    }
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "receipt.json"
        expected = write_hashed_json(path, receipt)
        assert verify_hashed_json(path) == expected
    print("tier1 fixture smoke: OK (3 turns, 8 questions, no model, no LoCoMo dataset)")


if __name__ == "__main__":
    main()
