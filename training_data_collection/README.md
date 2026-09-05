# Training Data Collection

This directory contains utilities for building CiteGuard training data in the
same row format as `DATASET.csv` in a local working checkout.

## Current scope

The first-pass collector focuses on:

- published computer science papers provided through a manifest or crawled from
  proceedings pages
- PDF inputs
- numeric bracket citations such as `[12]` or `[12, 14]`
- author-year citations such as `(Zhang et al., 2024)` and `Zhang et al. (2024)`
- single-sentence excerpts with single or multiple citations
- oracle citations that can be resolved through the Semantic Scholar API
- oracle citations that are discoverable through a configurable Semantic Scholar field filter

This is intentionally narrower than the full long-term target. It is a good fit
for both numeric-citation venues such as CVF-style proceedings and author-year
venues such as ACL Anthology pages, as long as the proceedings page exposes
title links and direct PDF links near each other.

## Files

- [`collect_numeric_cs_candidates.py`](collect_numeric_cs_candidates.py):
  extract candidate rows and audit metadata from an existing manifest
- [`collect_cs_conference_candidates.py`](collect_cs_conference_candidates.py):
  crawl proceedings pages with direct PDF links, build manifest rows
  automatically, and then run the same candidate-extraction pipeline

## Manifest format

Create a CSV manifest with at least these columns:

- `source_paper_title`
- `source_paper_url`
- `year`

Optional columns:

- `pdf_url`
- `pdf_path`
- `venue`
- `domain`

If `pdf_path` is not provided, the collector downloads the PDF from `pdf_url`.
If `pdf_url` is also missing, it falls back to `source_paper_url` only when the
source URL itself ends with `.pdf`.

Example:

```csv
source_paper_title,source_paper_url,year,pdf_url,venue,domain
Example Paper,https://example.org/paper,2024,https://example.org/paper.pdf,CVPR,Computer Science
```

ICLR 2024:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://proceedings.iclr.cc/paper_files/paper/2024 \
  --year 2024 \
  --venue ICLR \
  --manifest_out training_data_collection/output/iclr2024_manifest.csv \
  --output_csv training_data_collection/output/iclr2024_candidates.csv \
  --audit_jsonl training_data_collection/output/iclr2024_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_iclr2024 \
  --citation_style author_year \
  --citation_mode single \
  --progress_every 100
```

## Usage

Set `S2_API_KEY` first, then run:

```bash
python -m training_data_collection.collect_numeric_cs_candidates \
  --manifest training_data_collection/example_manifest.csv \
  --output_csv training_data_collection/output/train_candidates.csv \
  --audit_jsonl training_data_collection/output/train_candidates.audit.jsonl \
  --citation_style numeric \
  --citation_mode any \
  --progress_every 100 \
  --split train \
  --starting_id 100000
```

To crawl papers directly from conference proceedings pages that expose paper PDF
links, use:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://openaccess.thecvf.com/CVPR2024 \
  --year 2024 \
  --venue CVPR \
  --manifest_out training_data_collection/output/cvpr2024_manifest.csv \
  --output_csv training_data_collection/output/cvpr2024_candidates.csv \
  --audit_jsonl training_data_collection/output/cvpr2024_candidates.audit.jsonl \
  --citation_style numeric \
  --progress_every 100 \
  --citation_mode any
```

For ACL 2024 author-year citations, use the ACL Anthology volume pages and set
`--citation_style author_year`:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://aclanthology.org/volumes/2024.acl-long/ \
  --proceedings_url https://aclanthology.org/volumes/2024.acl-short/ \
  --proceedings_url https://aclanthology.org/volumes/2024.acl-demos/ \
  --year 2024 \
  --venue ACL \
  --manifest_out training_data_collection/output/acl2024_manifest.csv \
  --output_csv training_data_collection/output/acl2024_candidates.csv \
  --audit_jsonl training_data_collection/output/acl2024_candidates.audit.jsonl \
  --citation_style author_year \
  --citation_mode single \
  --progress_every 100
```

For NeurIPS 2024 numeric citations, use the NeurIPS proceedings listing and a
separate PDF cache so the run stays isolated from any in-flight ACL crawl:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://papers.nips.cc/paper_files/paper/2024 \
  --year 2024 \
  --venue NeurIPS \
  --manifest_out training_data_collection/output/neurips2024_manifest.csv \
  --output_csv training_data_collection/output/neurips2024_candidates.csv \
  --audit_jsonl training_data_collection/output/neurips2024_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_neurips2024 \
  --citation_style numeric \
  --citation_mode single \
  --progress_every 100
```

For ICML 2024 author-year citations, use the PMLR proceedings page and a
separate PDF cache:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://proceedings.mlr.press/v235/ \
  --year 2024 \
  --venue ICML \
  --manifest_out training_data_collection/output/icml2024_manifest.csv \
  --output_csv training_data_collection/output/icml2024_candidates.csv \
  --audit_jsonl training_data_collection/output/icml2024_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_icml2024 \
  --citation_style author_year \
  --citation_mode single \
  --progress_every 100
```

## Additional 2025 curation

For a diverse multi-citation batch, use separate output and cache paths from the
existing single-citation runs. These collectors resume safely if interrupted.

NeurIPS 2025 (numeric citations):

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://papers.nips.cc/paper_files/paper/2025 \
  --year 2025 --venue NeurIPS \
  --manifest_out training_data_collection/output/neurips2025_multi_manifest.csv \
  --output_csv training_data_collection/output/neurips2025_multi_candidates.csv \
  --audit_jsonl training_data_collection/output/neurips2025_multi_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_neurips2025 \
  --citation_style numeric --citation_mode multi --progress_every 100
```

ICML 2025 (author-year citations):

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://proceedings.mlr.press/v267/ \
  --year 2025 --venue ICML \
  --manifest_out training_data_collection/output/icml2025_multi_manifest.csv \
  --output_csv training_data_collection/output/icml2025_multi_candidates.csv \
  --audit_jsonl training_data_collection/output/icml2025_multi_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_icml2025 \
  --citation_style author_year --citation_mode multi --progress_every 100
```

ACL 2025 (author-year citations):

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://aclanthology.org/volumes/2025.acl-long/ \
  --proceedings_url https://aclanthology.org/volumes/2025.acl-short/ \
  --year 2025 --venue ACL \
  --manifest_out training_data_collection/output/acl2025_multi_manifest.csv \
  --output_csv training_data_collection/output/acl2025_multi_candidates.csv \
  --audit_jsonl training_data_collection/output/acl2025_multi_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_acl2025 \
  --citation_style author_year --citation_mode multi --progress_every 100
```

## Biomedical single-citation curation

BioNLP 2025 is a convenient first biomedical venue because ACL Anthology exposes
its paper and PDF links in a layout the crawler already supports. The new
`--fields_of_study` option applies the corresponding Semantic Scholar filter to
both oracle-reference resolution and discoverability checks:

```bash
python -m training_data_collection.collect_cs_conference_candidates \
  --proceedings_url https://aclanthology.org/volumes/2025.bionlp-1/ \
  --year 2025 --venue BioNLP \
  --domain "Biomedical NLP" --fields_of_study Medicine \
  --manifest_out training_data_collection/output/bionlp2025_single_manifest.csv \
  --output_csv training_data_collection/output/bionlp2025_single_candidates.csv \
  --audit_jsonl training_data_collection/output/bionlp2025_single_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_bionlp2025 \
  --citation_style author_year --citation_mode single --progress_every 50
```

Use `--fields_of_study Biology` for a biology-heavy venue, or
`--fields_of_study ""` to disable field filtering. Changing this option on an
existing checkpoint requires a new output path or `--restart`, preventing a
mixed-domain resumed run.

### PubMed Central Open Access

For broader biomedical coverage, the PubMed collector searches PubMed/PMC
content indexed by Europe PMC and requires both `OPEN_ACCESS:Y` and `HAS_PDF:Y`
before sending article PDFs through the same citation pipeline. Set a contact
email for the programmatic client:

```bash
export NCBI_EMAIL="you@example.edu"
python -m training_data_collection.collect_pubmed_candidates \
  --year 2025 \
  --query "clinical trial OR medical imaging OR genomics" \
  --max_articles 200 \
  --manifest_out training_data_collection/output/pubmed2025_single_manifest.csv \
  --output_csv training_data_collection/output/pubmed2025_single_candidates.csv \
  --audit_jsonl training_data_collection/output/pubmed2025_single_candidates.audit.jsonl \
  --pdf_cache_dir training_data_collection/cache/pdfs_pubmed2025 \
  --citation_style numeric \
  --citation_mode single \
  --fields_of_study Medicine \
  --progress_every 25
```

`--query` accepts Europe PMC search syntax and is optional. The collector always
adds the publication year, open-access, and indexed-PDF constraints. For reproducible,
focused batches, use separate output paths for queries such as `cancer`,
`medical imaging`, `genomics`, or `clinical trial` rather than one very broad
search. `NCBI_API_KEY` is also read automatically when available.

Build the default PubMed imaging + cancer/genomics candidates into a
deterministic biomedical test set:

```bash
python -m training_data_collection.build_biomed_test_split
```

This writes `training_data_collection/splits/test_2025_biomed.csv` and a JSON
validation report. It removes exact duplicates, assigns fresh test IDs, restores
journal/domain metadata from the audit files, and excludes excerpt, source-title,
or target-title overlap with the existing training datasets. Repeat `--input`
or `--train_reference` to supply custom candidate and leakage-reference files.

Both commands now write output incrementally and create a checkpoint sidecar by
default at `<output_csv>.state.json`. If the run is interrupted, rerunning the
same command resumes from the next unprocessed source paper instead of starting
over.

To start over from scratch with the same output paths, add:

```bash
--restart
```

## Outputs

The collector writes:

- a CSV with the same columns as the local working `DATASET.csv`
- a JSONL audit file with the raw sentence, raw reference text, Semantic Scholar
  resolution details, and discoverability metadata

## Final training bundles

The release includes two ready-to-use, post-adjudication split bundles:

- `splits/final_small/`: 350 training, 60 validation, and 299 test examples
- `splits/final_1k/`: 1,000 training, 200 validation, and 299 test examples

Both bundles use the same finalized 299-example main test set. Every row carries
the corrected canonical `target_paper_title`, a stable `target_paper_id`, and a
`release_id`. The bundles have been checked for exact normalized overlap in
passages, source-paper titles, and target-paper titles across train, validation,
and test. The machine-readable results are in
`splits/final_training_bundles_validation.json`.

## Notes

- The collector defaults to `fieldsOfStudy="Computer Science"` and accepts
  `--fields_of_study` to select another Semantic Scholar field or disable the
  filter; it retains the current CiteGuard year-cutoff logic.
- The discoverability filter currently checks whether the resolved oracle paper
  appears in top-20 Semantic Scholar relevance results for at least one
  heuristic query derived from the sentence.
- The proceedings crawler is intentionally heuristic-based: it works best on
  pages that list a title link followed nearby by a direct paper PDF link, such
  as many CVF and ACL Anthology proceedings pages. It also supports NeurIPS
  title-only listings whose abstract-page URLs can be mapped directly to paper
  PDFs, and PMLR proceedings pages where the title is plain text inside each
  paper block.
- `--citation_style numeric` is the default. Use `--citation_style author_year`
  for venues like ACL.
- Both collectors print a source-paper count before citation extraction and can
  print one example paper every `N` processed source papers via
  `--progress_every` (default: `100`).
- Both collectors retry transient proceedings, PDF, and Semantic Scholar
  requests by default and save checkpoint state after each completed source
  paper, which makes long conference crawls resumable.
- Unreadable PDFs that trigger `PyPDF2` parse errors are skipped and marked as
  processed so one malformed source paper does not stop the whole run.
- This script is designed to bootstrap clean candidates. It is expected that
  you will still do manual review and filtering before treating the output as
  final training data.

To remove bibliography entries that leaked into existing candidate files while
keeping each CSV, audit JSONL, and checkpoint synchronized, first preview and
then apply the cleanup:

```bash
python -m training_data_collection.clean_candidate_outputs --dry_run
python -m training_data_collection.clean_candidate_outputs
```

The cleanup creates `*.pre_reference_cleanup.bak` backups before modifying any
file. Manifests are not changed.

## Reproducible year-based splits

Build venue-balanced splits with 2024 source papers for training and 2025
source papers for testing:

```bash
python -m training_data_collection.build_year_splits
```

The default output is written to `training_data_collection/splits/`:

- `train_2024_balanced.csv`: 82 rows per venue (410 total)
- `test_2025_balanced.csv`: initially 60 rows per venue; the completed audit excludes one invalid NeurIPS context (299 retained total)
- `split_validation.json`: input, deduplication, balance, and leakage checks

The default training quota is limited by the current ICLR 2024 pool, which has
82 eligible candidates. The test split only includes target papers absent from
the complete eligible 2024 pool. It also has no exact excerpt, source-title, or
target-title overlap with the training split or exact excerpt overlap with
`DATASET.csv`. Sampling is deterministic under `--seed` and favors source- and
target-paper diversity.

When more 2024 candidates are available, increase the balanced quota without
changing the split logic:

```bash
python -m training_data_collection.build_year_splits --train_per_venue 200
```

## Full-dataset audit and adjudication

Run the conservative audit over every private construction record before
exporting public metadata:

```bash
python -m training_data_collection.audit_full_dataset \
  --input-dir training_data_collection/output \
  --output-dir training_data_collection/audited
```

The command writes revised candidate CSVs and audit JSONL files without
modifying the source artifacts. Confirmed manual corrections are applied,
deterministic structural failures are rejected, and ambiguous heuristic cases
are quarantined in `full_audit_review_queue.jsonl`. A nonzero exit status means
that review cases remain and the output is not release-ready.

After reviewing queued records, provide JSONL adjudications and rerun:

```bash
python -m training_data_collection.audit_full_dataset \
  --input-dir training_data_collection/output \
  --output-dir training_data_collection/audited \
  --adjudications training_data_collection/full_audit_adjudications.jsonl
```

Each adjudication contains `source_file`, `id`, an `action` of `accept`,
`revise`, or `reject`, and reviewer `reasons`. A `revise` decision additionally
contains a `revision` object with verified replacement fields. The output is
release-ready only when every record has a terminal accept, revise, or reject
decision. Automated warning flags alone never invent a replacement paper.
