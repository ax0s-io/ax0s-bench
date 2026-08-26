from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dataset(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256_file(path) == expected_sha256:
        print(f"already verified: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256_file(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    temporary.replace(path)
    print(f"fetched and verified: {path}")


def load_dataset(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return json.loads(path.read_text())
