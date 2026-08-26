from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dataset import fetch_dataset
from .runner import run, verify_hashed_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="ax0s-bench")
    subparsers = result.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="fetch and verify the configured dataset")
    fetch.add_argument("--config", type=Path, required=True)
    execute = subparsers.add_parser("run", help="execute a benchmark condition")
    execute.add_argument("--config", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify a result/calibration content hash")
    verify.add_argument("--receipt", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "verify":
        digest = verify_hashed_json(args.receipt)
        print(f"verified sha256: {digest}  {args.receipt}")
        return
    config_path = args.config.resolve()
    config = load_config(config_path)
    repo = config_path.parent.parent
    if args.command == "fetch":
        fetch_dataset(config.dataset["source_url"], repo / config.dataset["path"], config.dataset["sha256"])
    else:
        receipt, calibration = run(config, repo)
        print(f"receipt: {receipt}")
        print(f"calibration: {calibration}")


if __name__ == "__main__":
    main()
