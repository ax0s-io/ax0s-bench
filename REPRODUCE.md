# Reproduce the benchmark

Do not compare receipts across tiers. Each tier answers a different question and labels its output with a tier number.

## Tier 1: CI smoke

Tier 1 runs anywhere with Python 3.12 or newer.

```bash
./scripts/check.sh
```

It uses a synthetic three-turn, eight-question fixture, a stub answerer, and a stub judge. It checks source compilation, unit tests, receipt structure, binary scores, canonical hashing, verifier round trips, loopback-only egress policy, and adapter helpers.

Tier 1 does not download a model. It does not fetch LoCoMo. It does not measure memory quality or reproduce a published score.

## Tier 2: reduced local run

Tier 2 uses one small model exposed through a local OpenAI-compatible endpoint. It fetches the pinned LoCoMo dataset and runs eight questions from the first conversation through the full-context adapter.

1. Start a local model server at `http://127.0.0.1:8081/v1`.
2. Record the exact model name, file SHA-256, server build, and hardware. Replace the placeholder disclosure in `configs/tier2-local-subset.toml`.
3. Install the harness and fetch the pinned dataset.
4. Run the reduced configuration.

```bash
uv sync --frozen
uv run ax0s-bench fetch --config configs/tier2-local-subset.toml
uv run ax0s-bench run --config configs/tier2-local-subset.toml
uv run ax0s-bench verify --receipt results/tier2-local-subset.json
```

Tier 2 proves that the dataset, client, judge, receipt, and verifier work together with a real local model. It does not reproduce the reference model, hardware, full frame, or system scores. A different quantisation is a different protocol.

## Tier 3: full frozen protocol

Tier 3 is the only tier whose results may be placed beside the published rows.

### Required frame

- LoCoMo-10 at `snap-research/locomo@3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
- Dataset SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- All 1,986 questions over 10 conversations.
- Harness protocol revision `08ddce9607a9ae5e1f37d260844e5401f4d30404`.
- One NVIDIA GeForce RTX 5090 with 32,607 MiB in the reference execution.
- Linux x86_64, driver 595.84, Python 3.13.13 in the reference execution.

### Answer and judge model

The same resident Qwen3.8-27B Q6_K model answered and judged.

- Main GGUF: 22,884,407,264 bytes, SHA-256 `ade6d6615688aafa1dcfc9f5b4e1a66447bc02f8e099a006a10f9263f22f8ccc`.
- Draft GGUF: 3,164,006,688 bytes, SHA-256 `cbf60a0c48b431bb61f1d49b8948dc88ac29c398d6dbdbb2e6e89ef77eacc9a`.
- Multimodal projector: 931,146,432 bytes, SHA-256 `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`.
- llama.cpp revision `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`.
- llama-server binary SHA-256 `e3bd1a58373f4ba6c35e2b6eac9392366dd11092c98898e6763020d7842745a7`.
- Answer sampling: temperature 0, top-p 1, maximum 128 tokens, seed `20260818 + question_index`, thinking off.
- Judge sampling: temperature 0, top-p 1, maximum 256 tokens, seed `20260818 + 100000 + question_index`, thinking off.
- Judge prompt SHA-256 `d09585046cb8892ab86fdfae4b711829998500c2be98fc9e8845fdab110ca720`.

### AX0S production condition

Clone ax0s-memory separately to `vendor/ax0s-memory-bcd8710` and check out `bcd871066173ca10fcbc7d0a6f1091152e472b3b`. The source is not included here.

- Embedder: `Qwen/Qwen3-Embedding-0.6B` at snapshot `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1,024 dimensions, 2,048-token window.
- Cross-encoder: `BAAI/bge-reranker-v2-m3` at snapshot `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.
- Cross-encoder main blob SHA-256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.
- Recall: weighted hybrid, cross-encoder enabled, batch 8, RRF fusion, RRF weight 2.0, narrow filters on, top 20.
- Embeddings and reranking ran on CPU. The resident answer/judge server was the only GPU process.

Before any tier-3 run, point the fail-closed attestation guard at the exact local artifacts:

```bash
export AX0S_BENCH_MAIN_MODEL_PATH=/path/to/main.gguf
export AX0S_BENCH_DRAFT_MODEL_PATH=/path/to/draft.gguf
export AX0S_BENCH_MMPROJ_PATH=/path/to/mmproj.gguf
export AX0S_BENCH_LLAMA_SERVER_PATH=/path/to/llama-server
export AX0S_BENCH_LLAMA_CPP_REVISION=4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

The runner hashes each file, checks its byte size when recorded, validates the llama.cpp revision, confirms the model alias from `/models`, and requires the exact GPU name, memory, and driver. It aborts before constructing an adapter on any mismatch.

Run `configs/locomo-ax0s-bcd8710.toml` only after its source guard resolves to that clean checkout. The AX0S adapter also binds every recall environment value from the config before importing ax0s-memory and before each query. The older AX0S config explicitly unsets those values. The other full configurations document the frozen competitor conditions and use the same tier-3 model and hardware guard.

Tier 3 proves a same-protocol condition result on the full frame. One run does not prove variance, managed-product quality, production latency, or a vendor headline.
