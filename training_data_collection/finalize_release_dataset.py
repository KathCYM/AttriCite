"""Finalize an exported CiteAlign JSONL and write release validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PRIVATE_FIELDS = {
    "raw_sentence",
    "raw_reference",
    "excerpt",
    "pdf_path",
    "discoverability_query",
}
EXPECTED_SPLITS = {"train": 350, "validation": 60, "test": 299, "biomedical_test": 143}


def infer_pdf_url(source_url: str, current: str | None) -> str | None:
    if current:
        return current
    if source_url.lower().endswith(".pdf"):
        return source_url
    if "openaccess.thecvf.com/" in source_url and "/html/" in source_url:
        return source_url.replace("/html/", "/papers/").removesuffix(".html") + ".pdf"
    if "proceedings.iclr.cc/" in source_url and source_url.endswith("-Abstract-Conference.html"):
        return source_url.replace("-Abstract-Conference.html", "-Paper-Conference.pdf")
    if "papers.nips.cc/" in source_url and "-Abstract-" in source_url and source_url.endswith(".html"):
        return source_url.replace("-Abstract-", "-Paper-").removesuffix(".html") + ".pdf"
    return None


def infer_year(source_url: str, current: str | int | None) -> int | None:
    if current:
        return int(current)
    match = re.search(r"(?:19|20)\d{2}", source_url)
    return int(match.group(0)) if match else None


def load_manifests(manifest_dir: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    records: dict[tuple[str, str], dict[str, str]] = {}
    if manifest_dir is None:
        return records
    for path in manifest_dir.glob("*_manifest.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                records[(row.get("source_paper_url", ""), row.get("source_paper_title", ""))] = row
    return records


def finalize(
    metadata: Path,
    output_dir: Path,
    manifest_dir: Path | None = None,
    audit_report: Path | None = None,
) -> dict:
    with metadata.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    manifests = load_manifests(manifest_dir)
    split_ids: dict[str, list[str]] = {name: [] for name in EXPECTED_SPLITS}
    forbidden_found: set[str] = set()
    removed_private_fields: Counter[str] = Counter()
    seen_release_ids: set[str] = set()

    for row in rows:
        for field in PRIVATE_FIELDS:
            if field in row:
                row.pop(field)
                removed_private_fields[field] += 1
        forbidden_found.update(PRIVATE_FIELDS.intersection(row))
        source_url = str(row.get("source_paper_url") or "")
        manifest = manifests.get((source_url, str(row.get("source_paper_title") or "")), {})
        release_id = hashlib.sha256(f"{source_url}|{row.get('id', '')}".encode("utf-8")).hexdigest()[:20]
        if release_id in seen_release_ids:
            raise ValueError(f"duplicate stable release_id: {release_id}")
        seen_release_ids.add(release_id)
        row["release_id"] = release_id
        row["year"] = (
            int(manifest["year"])
            if manifest.get("year")
            else (2025 if row.get("domain") == "Biomedical Research" else infer_year(source_url, None))
        )
        row["source_pdf_url"] = infer_pdf_url(
            source_url, manifest.get("pdf_url") or row.get("source_pdf_url")
        )
        digest = row["passage_sha256"]
        index = row.get("body_sentence_index")
        row["passage_locator"] = (
            {"type": "body_sentence_index", "value": index}
            if index is not None
            else {"type": "normalized_sha256_scan", "value": digest}
        )
        split = row.get("experimental_split")
        if split in split_ids:
            split_ids[split].append(release_id)

    split_counts = {name: len(ids) for name, ids in split_ids.items()}
    report = {
        "schema_version": 1,
        "total_records": len(rows),
        "computer_science_records": sum(row.get("domain") == "Computer Science" for row in rows),
        "biomedical_records": sum(row.get("domain") == "Biomedical Research" for row in rows),
        "year_counts": dict(sorted(Counter(str(row.get("year")) for row in rows).items())),
        "split_counts": split_counts,
        "expected_split_counts": EXPECTED_SPLITS,
        "split_counts_match": split_counts == EXPECTED_SPLITS,
        "missing_source_pdf_url": sum(not row.get("source_pdf_url") for row in rows),
        "missing_source_pdf_url_in_experimental_splits": sum(
            bool(row.get("experimental_split")) and not row.get("source_pdf_url") for row in rows
        ),
        "index_locators": sum(row.get("body_sentence_index") is not None for row in rows),
        "fingerprint_scan_locators": sum(row.get("body_sentence_index") is None for row in rows),
        "missing_fingerprints": sum(not row.get("passage_sha256") for row in rows),
        "duplicate_release_ids": len(rows) - len(seen_release_ids),
        "forbidden_fields_found": sorted(forbidden_found),
        "removed_private_fields": dict(sorted(removed_private_fields.items())),
    }
    if audit_report is not None:
        completed_audit = json.loads(audit_report.read_text(encoding="utf-8"))
        report["full_dataset_audit"] = {
            "records_audited": completed_audit["records_audited"],
            "decision_counts": completed_audit["decision_counts"],
            "unresolved_review_records": completed_audit["unresolved_review_records"],
            "release_ready": completed_audit["release_ready"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "citealign_metadata.jsonl"
    with final_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for split, ids in split_ids.items():
        (splits_dir / f"{split}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    report = finalize(
        args.metadata,
        args.output_dir,
        args.manifest_dir,
        args.audit_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    failed = (
        report["total_records"] != 7750
        or report["computer_science_records"] != 7607
        or report["biomedical_records"] != 143
        or report["year_counts"] != {"2024": 2401, "2025": 5350}
        or not report["split_counts_match"]
        or report["missing_source_pdf_url_in_experimental_splits"]
        or report["missing_fingerprints"]
        or report["duplicate_release_ids"]
        or report["forbidden_fields_found"]
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
