"""Export metadata-only CiteAlign records from private collection artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from training_data_collection.collect_numeric_cs_candidates import (
    extract_pdf_text,
    normalize_body_text,
    normalize_spaces,
    split_body_and_references,
    split_sentences,
)


PRIVATE_FIELDS = {
    "raw_sentence",
    "raw_reference",
    "excerpt",
    "pdf_path",
    "discoverability_query",
}


def infer_pdf_url(source_url: str, manifest_pdf_url: str | None) -> str | None:
    if manifest_pdf_url:
        return manifest_pdf_url
    if source_url.lower().endswith(".pdf"):
        return source_url
    if "openaccess.thecvf.com/" in source_url and "/html/" in source_url:
        return source_url.replace("/html/", "/papers/").removesuffix(".html") + ".pdf"
    if "proceedings.iclr.cc/" in source_url and source_url.endswith("-Abstract-Conference.html"):
        return source_url.replace("-Abstract-Conference.html", "-Paper-Conference.pdf")
    if "papers.nips.cc/" in source_url and "-Abstract-" in source_url and source_url.endswith(".html"):
        return source_url.replace("-Abstract-", "-Paper-").removesuffix(".html") + ".pdf"
    return None


def infer_year(source_url: str, manifest_year: str | int | None) -> int | None:
    if manifest_year:
        return int(manifest_year)
    match = re.search(r"(?:19|20)\d{2}", source_url)
    return int(match.group(0)) if match else None


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifests(audit_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for path in audit_dir.glob("*_manifest.csv"):
        for row in _csv_rows(path):
            key = (row.get("source_paper_url", ""), row.get("source_paper_title", ""))
            result[key] = row
    return result


def _split_membership(split_dir: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    canonical = {
        "train_2024_rl.csv": "train",
        "validation_2024_rl.csv": "validation",
        "test_2025_balanced.csv": "test",
        "test_2025_biomed.csv": "biomedical_test",
    }
    for filename, split in canonical.items():
        path = split_dir / filename
        if not path.exists():
            continue
        for row in _csv_rows(path):
            result[(row.get("source_file", ""), row.get("original_id", ""))] = split
    return result


def _locate(raw_sentence: str, pdf_path: str, cache: dict[str, list[str]]) -> int | None:
    if not pdf_path or not Path(pdf_path).exists():
        return None
    if pdf_path not in cache:
        body, _ = split_body_and_references(extract_pdf_text(Path(pdf_path)))
        cache[pdf_path] = split_sentences(normalize_body_text(body))
    needle = normalize_spaces(raw_sentence)
    return next((i for i, value in enumerate(cache[pdf_path]) if normalize_spaces(value) == needle), None)


def export(
    audit_dir: Path,
    split_dir: Path,
    output: Path,
    locate_missing: bool = False,
    manifest_dir: Path | None = None,
) -> int:
    manifests = _manifests(manifest_dir or audit_dir)
    membership = _split_membership(split_dir)
    sentence_cache: dict[str, list[str]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for audit_path in sorted(audit_dir.glob("*.audit.jsonl")):
            source_file = audit_path.name.removesuffix(".audit.jsonl") + ".csv"
            with audit_path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    # Keep the adjudication record in the private audit trail, but
                    # omit passages that the completed audit marked unreleasable.
                    if row.get("manual_exclusion"):
                        continue
                    raw_sentence = str(row["raw_sentence"])
                    locator = row.get("body_sentence_index")
                    if locator is None and locate_missing:
                        locator = _locate(raw_sentence, str(row.get("pdf_path", "")), sentence_cache)
                    manifest = manifests.get(
                        (str(row.get("source_paper_url", "")), str(row.get("source_paper_title", ""))),
                        {},
                    )
                    experimental_split = membership.get((source_file, str(row.get("id", ""))))
                    # The biomedical public set is the deduplicated, leakage-checked
                    # 143-record evaluation split rather than all 144 raw candidates.
                    if row.get("domain") == "Biomedical Research" and experimental_split is None:
                        continue
                    public = {key: value for key, value in row.items() if key not in PRIVATE_FIELDS}
                    public.update(
                        {
                            "schema_version": 1,
                            "release_id": hashlib.sha256(
                                f"{row.get('source_paper_url', '')}|{row.get('id', '')}".encode("utf-8")
                            ).hexdigest()[:20],
                            "year": infer_year(str(row.get("source_paper_url", "")), manifest.get("year")),
                            "source_pdf_url": infer_pdf_url(
                                str(row.get("source_paper_url", "")), manifest.get("pdf_url")
                            ),
                            "body_sentence_index": locator,
                            "passage_locator": (
                                {"type": "body_sentence_index", "value": locator}
                                if locator is not None
                                else {"type": "normalized_sha256_scan", "value": hashlib.sha256(
                                    normalize_spaces(raw_sentence).encode("utf-8")
                                ).hexdigest()}
                            ),
                            "passage_sha256": hashlib.sha256(
                                normalize_spaces(raw_sentence).encode("utf-8")
                            ).hexdigest(),
                            "experimental_split": experimental_split,
                        }
                    )
                    destination.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Directory containing source manifests; defaults to --audit-dir.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--locate-missing",
        action="store_true",
        help="Re-parse cached PDFs to backfill missing sentence indices (the default uses fingerprint scan locators)",
    )
    args = parser.parse_args()
    count = export(
        args.audit_dir,
        args.split_dir,
        args.output,
        args.locate_missing,
        args.manifest_dir,
    )
    print(f"Wrote {count} metadata-only records to {args.output}")


if __name__ == "__main__":
    main()
