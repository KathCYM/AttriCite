# CiteAlign metadata release

This directory is the generated, publication-ready metadata layer for CiteAlign.
It deliberately excludes passages, bibliography text, local paths, cached PDFs,
and passage-derived discoverability queries.

## Contents

- `citealign_metadata.jsonl`: 7,750 metadata records (7,607 computer-science
  records and the deduplicated 143-record biomedical evaluation set).
- `splits/train_ids.txt`: 350 stable release IDs.
- `splits/validation_ids.txt`: 60 stable release IDs.
- `splits/test_ids.txt`: 299 stable release IDs.
- `splits/biomedical_test_ids.txt`: 143 stable release IDs.
- `validation_report.json`: counts and release-safety checks.
- `validation_report.json` also records the completed full-dataset audit counts.

Each record uses `release_id` as its globally unique stable identifier. Passages
are located either by a zero-based `body_sentence_index` or by scanning locally
extracted body sentences for the stored normalized SHA-256 fingerprint. Run the
repository's `training_data_collection/reconstruct_passages.py` utility to
reconstruct passages from the original publication hosts.

The complete 7,754-record construction pool was audited. Confirmed revisions
are recorded in the released metadata, two invalid extracted contexts are
excluded, and no unresolved review records remain.

CC BY 4.0 applies only to metadata and annotations created by the dataset
authors. Third-party publications and reconstructed passages remain governed by
their original copyright and license terms. See `../LICENSE_DATASET` and
`../DATASET_CARD.md`.
