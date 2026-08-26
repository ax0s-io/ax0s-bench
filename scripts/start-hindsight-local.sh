#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
venv="$repo/.bench/hindsight-091-venv"
freeze="$repo/configs/hindsight-091-freeze.txt"
actual_freeze=$(uv pip freeze --python "$venv/bin/python" | LC_ALL=C sort)
expected_freeze=$(LC_ALL=C sort "$freeze")
if [[ "$actual_freeze" != "$expected_freeze" ]]; then
  echo "Hindsight sidecar environment does not match $freeze" >&2
  exit 1
fi
if [[ $(printf '%s\n' "$actual_freeze" | shasum -a 256 | awk '{print $1}') != b214a3027fd2e8858bcf10a8a7815accf13f3ec7ac6cc7f7a13f52f97e2e309a ]]; then
  echo "Hindsight sidecar freeze hash mismatch" >&2
  exit 1
fi
run_name=${HINDSIGHT_RUN_NAME:-locomo-hindsight-full-v1}
port=${HINDSIGHT_API_PORT:-8889}
home="$repo/.bench/$run_name/home"
mkdir -p "$home"

export HOME="$home"
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export CUDA_VISIBLE_DEVICES=""
export HINDSIGHT_API_HOST=127.0.0.1
export HINDSIGHT_API_PORT="$port"
export HINDSIGHT_API_ENABLE_BANK_LLM_HEALTH=true
export HINDSIGHT_API_LLM_PROVIDER=lmstudio
export HINDSIGHT_API_LLM_API_KEY=local-no-cloud
export HINDSIGHT_API_LLM_MODEL=qwen38-local-vl
export HINDSIGHT_API_LLM_BASE_URL=http://127.0.0.1:8081/v1
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
export HINDSIGHT_API_EMBEDDINGS_LOCAL_FORCE_CPU=true
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=true
export OTEL_SDK_DISABLED=true
if [[ ${HINDSIGHT_OFFLINE:-0} == 1 ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

cd "$repo"
exec "$venv/bin/hindsight-api"
