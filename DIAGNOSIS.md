# Rejected Mem0 whole-session condition

An early Mem0 condition grouped a complete LoCoMo session into one message and omitted reliable temporal transport. It scored 4.38%.

That 4.38% is not a Mem0 score. It measures a broken adapter condition and is excluded from every comparison.

The accepted condition ingests one chronological turn per `Memory.add()` call. It carries the source date in metadata and in an authoritative content prefix, retries malformed local extraction once, applies pinned deterministic JSON repair only on the retry, and aborts on any residual parse failure.

The accepted Mem0 OSS 2.0.18 condition completed all 5,882 turns and scored 388/1,986, or 19.54%. A later pinned-revision rerun failed closed on one malformed turn after 757 questions. That aborted run did not emit a score. See `receipts/failures/mem0-pinned-rerun-failure.json`.
