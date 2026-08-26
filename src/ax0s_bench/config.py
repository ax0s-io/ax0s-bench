from __future__ import annotations

import dataclasses
import hashlib
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    endpoint: str
    model_id: str
    disclosed_model: str
    provider: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    timeout_seconds: int
    enable_thinking: bool


@dataclasses.dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    raw_sha256: str
    run: dict[str, Any]
    dataset: dict[str, Any]
    answer_model: ModelConfig
    judge_model: ModelConfig
    condition: dict[str, Any]
    attestation: dict[str, Any]


def load_config(path: Path) -> Config:
    raw_bytes = path.read_bytes()
    raw = tomllib.loads(raw_bytes.decode())
    return Config(
        path=path,
        raw=raw,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        run=raw["run"],
        dataset=raw["dataset"],
        answer_model=ModelConfig(**raw["answer_model"]),
        judge_model=ModelConfig(**raw["judge_model"]),
        condition=raw["condition"],
        attestation=raw.get("attestation", {}),
    )
