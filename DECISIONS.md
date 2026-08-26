# Decisions made during public-repo preparation

The ratified publication report left a few details open. These conservative choices fill those gaps.

- The repository is named `ax0s-bench`. No Git remote or GitHub URL is configured because the exact AX0S organisation handle still needs Chris's confirmation.
- MemMachine ships as adapter code and frozen configuration only. It does not get a scored row because it was not part of the published comparison.
- Tier 2 uses the full-context adapter, one fetched conversation, eight questions, and a user-supplied small local model. No specific small model is endorsed because a model digest was not ratified.
- Public result receipts keep the first 20 retrieval identifiers and scores per question. They record the original result count when the private receipt returned more. This keeps the public files reviewable without publishing retrieved text.
- Four scored result receipts ship: current AX0S, prior AX0S, Hindsight, and Mem0. The prior AX0S row stays because it records the pre-fix default and is the clean original head-to-head condition.
- The two failure receipts are redacted summaries linked to the private originals by SHA-256. Raw failure logs stay private.
- GitHub Actions runs tier 1 only. It uses Python and the repository source directly, so it does not resolve the heavy benchmark dependencies in `uv.lock`.
- Tier 3 uses environment variables only to locate local model files. The frozen config owns every expected digest, byte size, server revision, and GPU value, and the runner fails closed on mismatch. Paths stay out of receipts.
