# AX0S Bench

AX0S wrote this benchmark harness, and it benchmarks AX0S's own product alongside open-source memory systems.

This repository contains the code, frozen configurations, redacted receipts, and fairness notes for local LoCoMo condition comparisons. It is not an independent benchmark and does not claim to rank managed products.

## Framing rules

These rules apply to every result in this repository:

- same-protocol local-model comparisons only
- never replication/refutation of vendor published numbers
- cat-5 reported separately
- condition comparison not managed-product ranking

These are historical results from August 2026, not measurements of today's systems. AX0S scored 871/1,986, or 43.86%, on August 25 at `bcd8710` with the cross-encoder configuration used in production at the time. The earlier AX0S condition scored 792/1,986, or 39.88%, on August 20 at `a2700dd` without the cross-encoder. Hindsight OSS 0.9.1 scored 721/1,986, or 36.30%, on August 22. Mem0 OSS 2.0.18 scored 388/1,986, or 19.54%, on August 19. Dates are completion dates in US Central time; full timestamps and harness revisions are in [RESULTS.md](RESULTS.md).

AX0S scores lower than Hindsight on categories 2 and 3 in both documented AX0S conditions. Category 5 is adversarial and appears separately from categories 1 to 4. See [RESULTS.md](RESULTS.md) and [GAPS.md](GAPS.md) before quoting a number.

## What is here

- `src/ax0s_bench/`: the runner, adapters, judge, dataset fetcher, egress guard, and CLI.
- `configs/`: frozen and reduced-run TOML configurations.
- `prompts/`: the binary judge prompt. Its SHA-256 is `d09585046cb8892ab86fdfae4b711829998500c2be98fc9e8845fdab110ca720`.
- `receipts/redacted/`: one dataset-text-free result receipt for each scored system or AX0S condition.
- `receipts/calibration/`: redacted 40-item judge samples. `secondary_judge` is empty by design.
- `receipts/failures/`: redacted Graphiti and pinned-Mem0 failure receipts.
- `schemas/`: the public receipt schema.
- `REPRODUCE.md`: three reproduction tiers and their limits.
- `GAPS.md`, `DIAGNOSIS.md`, and `JUDGE_NOTES.md`: fairness and failure disclosures.
- `SCRUB.md`: the named privacy scrub and rerunnable commands.

The original private repository history, raw LoCoMo text, model stores, checkpoints, host logs, and full receipts are not present.

## Quick check

Tier 1 needs Python 3.12 or newer and `uv` (including `uvx`). It does not install project dependencies, fetch LoCoMo, or download a model.

```bash
./scripts/check.sh
```

The check compiles the source, runs unit tests, runs the synthetic three-turn/eight-question fixture, verifies every public receipt hash, and checks the redaction contract.

## Fetching LoCoMo

LoCoMo is not included. To fetch the pinned file for tier 2 or tier 3:

```bash
uv sync --frozen
uv run ax0s-bench fetch --config configs/tier2-local-subset.toml
```

The fetch command downloads `snap-research/locomo@3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` and rejects any file whose SHA-256 is not `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

## Dataset licence

LoCoMo is distributed by its authors under CC BY-NC 4.0. The NonCommercial restriction applies to the dataset, even though this harness code uses Apache-2.0. This repository does not vendor the dataset or its verbatim question and answer text.

- Source: https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
- Licence: https://creativecommons.org/licenses/by-nc/4.0/
- Upstream licence file: https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt
- Changes made by this project: the harness fetches the pinned upstream file, selects benchmark subsets, sends questions through configured conditions, and publishes hashes and scores. It does not change the dataset file.
- Disclaimer: LoCoMo is provided under its upstream licence and without warranties. AX0S does not grant rights to the dataset.

Commercial users must assess whether their use complies with CC BY-NC 4.0. Publishing scores derived from NonCommercial material may also warrant legal review.

## Receipt policy

Public receipts remove LoCoMo question text, gold answers, evidence, retrieved text, raw judge output, judge rationales, hostnames, and private paths. They retain per-question hashes, predictions, scores, top retrieval identifiers and scores, aggregate results, model disclosures, and the SHA-256 anchors for the full private receipt.

Full receipts remain private. Auditors can request them, subject to dataset-licence review. Use `ax0s-bench verify --receipt <path>` to verify any public receipt's canonical content hash.

## Licence

The harness code is Apache-2.0. LoCoMo and third-party dependencies keep their own licences. See [LICENSES.md](LICENSES.md) and [NOTICE](NOTICE).
