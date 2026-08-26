#!/usr/bin/env bash
# One-time local services and offline assets for the MemMachine and Graphiti conditions.
# Idempotent: safe to re-run. Starts nothing on the GPU and never touches the llama.cpp server.
set -euo pipefail

NEO4J_CONTAINER="${NEO4J_CONTAINER:-bench-memmachine-neo4j}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7688}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7475}"
NEO4J_DATA_DIR="${NEO4J_DATA_DIR:-$HOME/bench-services/neo4j}"
: "${MEMMACHINE_NEO4J_PASSWORD:?set MEMMACHINE_NEO4J_PASSWORD to the local Neo4j password}"

if podman ps --format '{{.Names}}' | grep -qx "$NEO4J_CONTAINER"; then
  echo "neo4j: $NEO4J_CONTAINER already running"
else
  mkdir -p "$NEO4J_DATA_DIR/data" "$NEO4J_DATA_DIR/logs"
  podman rm -f "$NEO4J_CONTAINER" >/dev/null 2>&1 || true
  podman run -d --name "$NEO4J_CONTAINER" \
    -p "127.0.0.1:${NEO4J_BOLT_PORT}:7687" -p "127.0.0.1:${NEO4J_HTTP_PORT}:7474" \
    -e NEO4J_AUTH="neo4j/${MEMMACHINE_NEO4J_PASSWORD}" \
    -e NEO4J_server_memory_heap_initial__size=1G -e NEO4J_server_memory_heap_max__size=4G \
    -v "$NEO4J_DATA_DIR/data:/data:Z" -v "$NEO4J_DATA_DIR/logs:/logs:Z" \
    docker.io/library/neo4j:5.23-community
  echo "neo4j: started $NEO4J_CONTAINER on 127.0.0.1:${NEO4J_BOLT_PORT}"
fi

# MemMachine's BM25 reranker (tokenizer: default) needs these NLTK corpora on disk.
# Downloading them once keeps every benchmark run free of outbound requests.
uv run python - <<'PY'
import nltk
for package in ("stopwords", "punkt", "punkt_tab"):
    print(package, nltk.download(package, quiet=True))
PY

# fastembed weights for BAAI/bge-small-en-v1.5 are cached the same way for every condition.
uv run python - <<'PY'
from fastembed import TextEmbedding
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("fastembed dims", len(next(iter(model.embed(["cache probe"])))))
PY

echo "setup complete"
