from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from .config import ModelConfig


class OpenAIChatClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat(self, messages: list[dict[str, str]], *, seed_offset: int = 0) -> dict[str, Any]:
        cfg = self.config
        payload = {
            "model": cfg.model_id,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            "seed": cfg.seed + seed_offset,
            "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
        }
        request = urllib.request.Request(
            cfg.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                return {
                    "content": choice["message"].get("content", "").strip(),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": body.get("usage", {}),
                    "response_model": body.get("model"),
                    "system_fingerprint": body.get("system_fingerprint"),
                }
            except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        raise RuntimeError(f"chat completion failed after 3 attempts: {last_error}")

    def disclosure(self) -> dict[str, Any]:
        return asdict(self.config)


def probe_models(endpoint: str, timeout: int = 10) -> dict[str, Any]:
    request = urllib.request.Request(endpoint.rstrip("/") + "/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
