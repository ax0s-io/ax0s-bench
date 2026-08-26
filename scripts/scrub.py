#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = {
    "hostnames": [re.compile(rb"chris[-_](?:5090|ms|macbook|laptop)", re.I)],
    "home paths": [re.compile(rb"/(?:home|Users)/chris(?:/|\b)", re.I), re.compile(b".prime" + b"/orchestration-runs", re.I)],
    "internal IPs": [
        re.compile(rb"\b100(?:\.\d{1,3}){3}\b"),
        re.compile(rb"\b10(?:\.\d{1,3}){3}\b"),
        re.compile(rb"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"),
        re.compile(rb"\b192\.168(?:\.\d{1,3}){2}\b"),
    ],
    "emails": [re.compile(rb"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)],
    "tokens": [
        re.compile(b"-----BE" + b"GIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"\bBea" + rb"rer\s+[A-Za-z0-9._~+/=-]{12,}", re.I),
        re.compile(rb"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*['\"][A-Za-z0-9._~+/=-]{24,}", re.I),
    ],
    "ssh aliases": [
        re.compile(rb"\bssh\s+(?:-\S+\s+)*5090\b", re.I),
        re.compile(rb"\bHost\s+5090\b", re.I),
        re.compile(rb"\bchris" + rb"@", re.I),
    ],
    "tailnet names": [re.compile(rb"\.ts\.net\b", re.I), re.compile(b"egd" + b"atahub", re.I)],
}
ALLOWED_VERSION_LINE = b"nvidia-curand==" + b"10." + b"4.0.35"


def indexed_blobs() -> list[tuple[str, str, bytes]]:
    listed = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    blobs = []
    for raw_entry in listed.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode().split()
        if stage != "0":
            raise RuntimeError(f"unmerged index entry: {raw_path.decode(errors='replace')}")
        data = subprocess.run(["git", "cat-file", "blob", object_id], cwd=ROOT, check=True, capture_output=True).stdout
        blobs.append((mode, raw_path.decode(), data))
    return blobs


def line_at(data: bytes, offset: int) -> bytes:
    start = data.rfind(b"\n", 0, offset) + 1
    end = data.find(b"\n", offset)
    return data[start : len(data) if end == -1 else end]


def main() -> None:
    blobs = indexed_blobs()
    if not blobs:
        raise SystemExit("scrub requires a populated Git index")
    findings = []
    total = 0
    for mode, path, data in blobs:
        if mode == "120000":
            findings.append(("symlink", path, "symlink not allowed"))
            continue
        total += len(data)
        for category, patterns in CATEGORIES.items():
            for pattern in patterns:
                matches = list(pattern.finditer(data))
                if category == "internal IPs" and path == "configs/hindsight-091-freeze.txt":
                    matches = [match for match in matches if line_at(data, match.start()) != ALLOWED_VERSION_LINE]
                if matches:
                    findings.append((category, path, pattern.pattern.decode(errors="replace")))
                    break
    print(f"checked {len(blobs)} indexed files, {total} bytes")
    for category in CATEGORIES:
        count = sum(1 for item in findings if item[0] == category)
        print(f"{category}: {'PASS' if count == 0 else f'FAIL ({count})'}")
    symlinks = sum(1 for item in findings if item[0] == "symlink")
    print(f"symlinks: {'PASS' if symlinks == 0 else f'FAIL ({symlinks})'}")
    if findings:
        for category, path, pattern in findings:
            print(f"{category}: {path}: {pattern}")
        raise SystemExit(1)
    print("scrub verdict: PASS")


if __name__ == "__main__":
    main()
