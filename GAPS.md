# Gaps and fairness limits

## Comparison limits

- These are same-protocol local-model condition comparisons. They do not replicate or refute vendor-published numbers.
- One local Qwen model answered and judged. There was no condition-specific independent secondary judge.
- There is one run per condition. Temperature zero and fixed seeds reduce variance but do not guarantee byte-identical GPU inference.
- Category 5 is adversarial and is reported separately from categories 1 to 4.
- CPU-forced embeddings and reranking make these retrieval-quality results, not production-latency measurements.
- A third-party audit has alleged LoCoMo answer-key defects. AX0S did not reproduce that audit or change any key. Each score means agreement with the pinned key under the frozen rubric, not ground-truth factual accuracy.

## AX0S conditions

The two AX0S rows have different exact conditions.

- `a2700dd3230f603d2150796329cf2f65deba63e5` measured the pre-fix default: Qwen3-Embedding-0.6B, weighted-hybrid retrieval, top 20, no minimum-score filter, no cross-encoder, and no incidence expansion.
- `bcd871066173ca10fcbc7d0a6f1091152e472b3b` measured the shipped production recall configuration: Qwen3-Embedding-0.6B, `BAAI/bge-reranker-v2-m3` cross-encoder on, batch 8, RRF fusion, RRF weight 2.0, narrow filters on, and top 20.

The cross-encoder is on by default in production and the `bcd8710` run measured it. The prior row remains because it records the pre-fix default. The `bcd8710` result is a revision-plus-configuration condition delta, not a code-only reranker ablation.

Open AX0S questions include deeper retrieval, minimum-score thresholds, incidence expansion, repeat-run variance, and production GPU latency.

## Mem0 condition

- The run used Mem0 OSS 2.0.18, not managed Mem0 Platform v3.
- It used local Qdrant, `BAAI/bge-small-en-v1.5`, 384 dimensions, top 20, threshold 0.1, and local Qwen extraction.
- Direct OSS rejected the managed API's `timestamp` parameter. The adapter carried time in metadata and an authoritative content prefix.
- Reasonable untried conditions include Platform v3, top 200, a stronger extractor or embedder, reranking, threshold changes, and the vendor rubric.
- The completed Mem0 receipt names harness revision `e5b64787de51fe3d6bf9a0fa3ef213a94a0bd1fc`. The pinned `08ddce9` rerun failed closed after a malformed turn and is not a Mem0 score.
- The completed Mem0 run lacks the later audit-hook egress receipt. Its configured model, embedder, and vector store were local, but the receipt cannot independently prove that no library telemetry attempted external egress.

## Hindsight condition

- The run used Hindsight OSS 0.9.1 with its frozen sidecar dependencies.
- Hindsight leads both AX0S rows on categories 2 and 3.
- Open questions include vendor-tuned settings, repeat runs, and independent judging.

## Graphiti condition

Graphiti OSS 0.29.3 was attempted but not scored. The local Qwen extraction path failed on 256 of 419 ingested turns. The runner aborted rather than skip failures. Prompt and structured-output work remains open.

## Asymmetry and updates

Mem0, Hindsight, and Graphiti were not rerun for the AX0S `bcd8710` update. The original same-frame head-to-head uses the `a2700dd` AX0S row. A `bcd8710` comparison against frozen competitor rows mixes an updated AX0S revision and recall configuration with their earlier conditions.

Any vendor may submit a tuned configuration for its own system. See `CONTRIBUTING.md`.
