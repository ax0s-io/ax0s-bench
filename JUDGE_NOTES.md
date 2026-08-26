# Judge notes

## Frozen rubric

- Judge prompt: `prompts/locomo_binary_judge_v1.txt`.
- Prompt SHA-256: `d09585046cb8892ab86fdfae4b711829998500c2be98fc9e8845fdab110ca720`.
- Judge: the same local Qwen3.8-27B Q6_K model used as the answerer.
- Sampling: temperature 0, top-p 1, maximum 256 tokens, thinking off, seed `20260818 + 100000 + question_index + retry_attempt`.
- Input: question, LoCoMo reference answer, and candidate answer. The judge did not see the source conversation or retrieved context.
- Output: binary 0 or 1 plus a short rationale. The runner allowed three parse attempts and aborted on a terminal parse failure.

## Public evidence

The public calibration files retain question and gold-answer hashes, the prediction, the binary decision, and an empty `secondary_judge` field. They remove the LoCoMo text and judge rationale. Each file links to the full private calibration by raw and canonical SHA-256.

Full receipts retain the raw judge response, parsed decision, usage, response model, and fingerprint. They remain private because they reproduce LoCoMo text.

## Independent review status

There was no independent secondary-judge pass for the scored memory conditions. No condition-specific Cohen's kappa or disagreement rate is claimable.

A separate prior 40-item audit of the full-context baseline reported 38/40 agreement with an Anthropic judge and found Qwen stricter on two multi-clause answers. That audit does not apply a correction to these scores.

## Answer-key caveat

A competitor report cites a third-party audit alleging score-corrupting LoCoMo key errors and permissive-judge acceptance of adjacent wrong answers. AX0S did not reproduce that audit. No row was removed, relabelled, or corrected.

Read each score as agreement with the pinned LoCoMo key under this rubric.
