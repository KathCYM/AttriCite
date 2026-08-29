# AttriCite release guide

This repository is the code release for **AttriCite: Training an Open 4B Model
for Citation Recovery toward Faithful Attribution**. Do not create a public tag
until `python scripts/validate_release.py` succeeds.

## Anonymous review snapshot

Distribute a clean snapshot rather than the working Git directory. Do not
include `.git/`, commit metadata, local virtual environments, scheduler logs,
cached PDFs, experiment logs, API credentials, or filesystem paths. Upload the
code, metadata, cards, and checkpoint through accounts and links that do not
identify the authors. The anonymous cards deliberately omit permanent URLs,
author names, citation metadata, and archival revisions; restore these only for
the camera-ready release.

Run the release validator on the exact unpacked snapshot before submission:

```bash
python scripts/validate_release.py
```

## Paper-to-artifact map

| Paper commitment | Public artifact | Status before publishing |
| --- | --- | --- |
| Collection pipeline | `training_data_collection/` | Included |
| Passage reconstruction | `training_data_collection/reconstruct_passages.py` | Included |
| Dataset metadata and annotations | Metadata-only JSONL exported with `export_release_metadata.py` | Generate and upload |
| 350/60/299 temporal splits | Split membership in metadata (`experimental_split`) | Generate and verify |
| 143-item biomedical set | Metadata with `domain=Biomedical Research` | Generate and verify |
| Complete agent prompt and action protocol | `src/retriever/prompt_templates/`, `training/verl/` | Included |
| Full-parameter GRPO recipe | `training/verl/run_grpo_qwen3_4b.sh` | Included |
| Selected step-475 model | Hugging Face checkpoint referenced in `MODEL_CARD.md` | Upload and replace placeholders |
| Code under MIT | `LICENSE` | Included |
| Created metadata/annotations under CC BY 4.0 | `LICENSE_DATASET` | Included; third-party text excluded |

## Prepare dataset artifacts

The working CSV and audit files contain third-party text and are intentionally
gitignored. Export the public metadata from the repository root:

```bash
python -m training_data_collection.audit_full_dataset \
  --adjudications training_data_collection/full_audit_adjudications.jsonl

python -m training_data_collection.export_release_metadata \
  --audit-dir training_data_collection/audited \
  --manifest-dir training_data_collection/output \
  --split-dir training_data_collection/splits \
  --output release/citealign_metadata.jsonl

python -m training_data_collection.finalize_release_dataset \
  --metadata release/citealign_metadata.jsonl \
  --output-dir release \
  --manifest-dir training_data_collection/output \
  --audit-report training_data_collection/audited/full_audit_report.json
```

The exporter removes `raw_sentence`, `raw_reference`, `excerpt`, local paths, and
passage-derived discoverability queries.
It retains source identifiers, target annotations, a normalized-passage SHA-256
fingerprint, and a deterministic passage locator.

The audit step applies the complete adjudication file, excludes confirmed
construction failures, and must report zero unresolved review records before
export.

Users reconstruct text from the original host:

```bash
python -m training_data_collection.reconstruct_passages \
  --metadata release/citealign_metadata.jsonl \
  --output-dir training_data_collection/reconstructed
```

Reconstruction is best-effort because upstream PDFs can change or disappear.
The tool reports every unavailable or fingerprint-mismatched item and never
silently substitutes text.

## Publish the checkpoint

Merge the selected veRL FSDP actor checkpoint into Hugging Face format using
the veRL checkpoint merger used by the evaluation launchers, then upload the
complete tokenizer/config/weight directory. For anonymous review, distribute
the model and cards through anonymous supplementary-material links. Add
permanent repository URLs, revisions, citation metadata, and checksums only in
the camera-ready release. The checkpoint must be licensed Apache-2.0,
consistent with Qwen3-4B and the paper.

Do not commit multi-gigabyte weights to Git. Store them in the model host (or
Git LFS if explicitly desired), record immutable revisions and SHA-256 hashes in
the release manifest, and test loading the published revision in a clean
environment.

## Final checks

```bash
python scripts/validate_release.py
python -m pytest -q
```

Before tagging, also verify that no API keys, cached PDFs, raw excerpts, private
cluster paths, logs, or checkpoint shards are staged. Publish an immutable code
tag and record its commit SHA in the dataset and model cards.
