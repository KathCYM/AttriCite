---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - en
tags:
  - citation-recovery
  - scholarly-retrieval
---

# CiteAlign

CiteAlign supports training and temporally separated evaluation of open-ended
citation recovery. The full collection has 7,607 eligible instances from 2024
and 2025 ACL, CVPR, ICLR, ICML, and NeurIPS source papers. The experimental
subset contains 350 training, 60 validation, and 299 test instances. A separate
143-instance 2025 biomedical set is provided for cross-domain evaluation.

## Public record format

The release intentionally does not redistribute passages or bibliography text
copied from third-party publications. Each JSONL record contains:

- source title, landing-page URL, PDF URL when available, venue, domain, and year;
- target Semantic Scholar identifier, canonical title, URL, and year;
- citation marker and deterministic zero-based body-sentence locator;
- SHA-256 of the whitespace-normalized unmasked sentence;
- resolution/discoverability annotations and experimental split membership.

Run `training_data_collection/reconstruct_passages.py` to download each source
from its original host, locate the sentence, verify the fingerprint, and replace
the recorded citation marker with `[CITATION]`. Upstream availability and PDF
revisions can cause explicit reconstruction failures.

## Split policy

Training and validation use 2024 source papers; testing uses 2025 source papers.
Every main-test target is absent from the complete eligible 2024 pool. The
experimental partitions have no normalized source-title, target-title, or exact
passage overlap. The biomedical set is evaluation-only and was not used for
training or checkpoint selection.

## License and responsible use

CC BY 4.0 applies only to metadata and annotations created by the authors. It
does not relicense third-party papers or reconstructed passages. Users are
responsible for access conditions and licenses at each original host. A recorded
citation is a proxy for author intent, not a guarantee that the source supports
the surrounding claim; inspect papers before using recovered citations.

The dataset and code are provided through the anonymous supplementary-material
links associated with the submission. Permanent URLs and the archival code
revision will be added after double-blind review.
