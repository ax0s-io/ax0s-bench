# Results

## Frozen frame

- Dataset: LoCoMo-10 at `snap-research/locomo@3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`.
- Dataset SHA-256: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Frame: 1,986 questions over 10 conversations.
- Categories 1 to 4: 1,540 questions.
- Category 5: 446 questions, reported separately.
- Answerer and judge: the same local Qwen3.8-27B Q6_K model.
- Judge prompt SHA-256: `d09585046cb8892ab86fdfae4b711829998500c2be98fc9e8845fdab110ca720`.

## Scored conditions

- AX0S `bcd871066173ca10fcbc7d0a6f1091152e472b3b`, production recall configuration: 871/1,986, 43.86%. Categories 1 to 4: 861/1,540, 55.91%. Category 5: 10/446, 2.24%. Per category: 93/282, 160/321, 17/96, 591/841, and 10/446.
- AX0S `a2700dd3230f603d2150796329cf2f65deba63e5`, pre-fix default without the cross-encoder: 792/1,986, 39.88%. Categories 1 to 4: 783/1,540, 50.84%. Category 5: 9/446, 2.02%.
- Hindsight OSS 0.9.1: 721/1,986, 36.30%. Categories 1 to 4: 716/1,540, 46.49%. Category 5: 5/446, 1.12%. Per category: 73/282, 204/321, 24/96, 415/841, and 5/446.
- Mem0 OSS 2.0.18: 388/1,986, 19.54%. Categories 1 to 4: 382/1,540, 24.81%. Category 5: 6/446, 1.35%. Per category: 45/282, 104/321, 11/96, 222/841, and 6/446.
- Graphiti OSS 0.29.3: attempted but not scored. Ingestion failed closed after 256 of 419 turns failed. See `receipts/failures/graphiti-ingestion-failure.json`.

## Cross-encoder condition change

The `bcd8710` run measured AX0S with the production recall configuration: `BAAI/bge-reranker-v2-m3`, batch 8, RRF fusion, RRF weight 2.0, and narrow filters on. The reranker ran on CPU. The prior `a2700dd` row measured the pre-fix default without the cross-encoder.

The change from `a2700dd` to `bcd8710` is +79 correct and +3.98 percentage points overall. It is a shipped-condition delta for revision plus recall configuration, not a code-only ablation.

Mem0, Hindsight, and Graphiti were not rerun for the `bcd8710` update. The clean original head-to-head is the `a2700dd` AX0S row against the frozen competitor rows. Comparing `bcd8710` against those frozen rows mixes AX0S's newer revision and recall configuration with the prior competitor conditions.

## Receipt anchors

- AX0S `bcd8710` full raw result SHA-256: `1f087800025f712d7de97545d97300f49a6ac285d9314891ab15e0ea5c23002b`.
- AX0S `bcd8710` canonical full receipt hash: `4bbbc8c416df138609106b4b5523ddb604397c5ac7fda63dc933056150785c74`.
- AX0S `a2700dd` canonical full receipt hash: `819ac1af1588baae43cdcb7b7a9a57d5f67d5434194d8f4bae43b23295895d91`.
- Mem0 canonical full receipt hash: `94d479b883a03c208714404ef6c2c22a363e18e03ea3a43f5ba2073ed81bec5a`.
- Hindsight canonical full receipt hash: `0ff1b87e1bb62d2ae458a10fe85044a3a04926a0e30d17c523d9aa40566f79ca`.

Every redacted receipt also carries its full raw-file and canonical-content SHA-256 values.
