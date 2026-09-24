from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api import OpenAIChatClient


def _balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = 0
        while True:
            idx = text.find("{", start)
            if idx == -1:
                raise exc
            end = _balanced_object_end(text, idx)
            if end is not None:
                try:
                    return json.loads(text[idx:end])
                except json.JSONDecodeError:
                    pass
            start = idx + 1


class LLMJudge:
    def __init__(self, client: OpenAIChatClient, prompt_path: Path):
        self.client = client
        self.prompt_path = prompt_path
        self.rubric = prompt_path.read_text()

    def grade(self, question: str, gold: str, prediction: str, *, question_index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        response: dict[str, Any] = {}
        decision: dict[str, Any] = {}
        for attempt in range(3):
            response = self.client.chat(
                [
                    {"role": "system", "content": self.rubric},
                    {
                        "role": "user",
                        "content": (
                            f"QUESTION: {question}\n"
                            f"REFERENCE ANSWER: {gold}\n"
                            f"CANDIDATE ANSWER: {prediction}\n"
                            "Return the JSON decision now. Keep the rationale under 20 words."
                        ),
                    },
                ],
                seed_offset=100000 + question_index + attempt,
            )
            try:
                decision = _parse_json(response["content"])
                score = int(decision["score"])
                if score not in (0, 1):
                    raise ValueError(f"judge score must be 0 or 1, got {score}")
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(f"judge returned no valid decision after 3 attempts: {last_error}")
        return {
            "score": score,
            "verdict": "correct" if score == 1 else "incorrect",
            "rationale": str(decision.get("rationale", "")),
            "raw_response": response["content"],
            "usage": response.get("usage", {}),
            "response_model": response.get("response_model"),
            "system_fingerprint": response.get("system_fingerprint"),
        }
