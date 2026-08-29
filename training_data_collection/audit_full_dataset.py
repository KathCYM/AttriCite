"""Audit every private CiteAlign record and produce adjudicated dataset artifacts.

The pipeline is deliberately conservative. It automatically applies only
authoritative corrections, rejects deterministic construction failures, and
sends ambiguous heuristic findings to a review queue. Source artifacts are
never modified; revised CSV/JSONL files are written to a separate directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from training_data_collection.apply_audit_corrections import (
    CORRECTIONS,
    EXCLUSIONS,
    detect_common_failures,
    paper_url,
)


REQUIRED_FIELDS = (
    "id",
    "source_paper_title",
    "source_paper_url",
    "citation_marker",
    "raw_sentence",
    "raw_reference",
    "resolved_paper_id",
    "resolved_title",
)


@dataclass(frozen=True)
class Decision:
    action: str
    reasons: tuple[str, ...]
    revision: dict | None = None
    authority: str = "automatic_policy"


def _normalized(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_like_bibliography_entry(sentence: str) -> bool:
    """Detect only high-confidence bibliography fragments.

    This intentionally requires several independent signals. Truncated or
    noisy body sentences are review candidates, not automatic rejections.
    """
    text = _normalized(sentence)
    if not text:
        return False
    starts_like_authors = bool(
        re.match(r"^(?:\[[0-9]+\]\s*)?[A-Z][\w'’-]+(?:,|\s+[A-Z]\.)", text)
    )
    venue_or_identifier = bool(
        re.search(r"\b(?:arXiv|doi|Proceedings of|pp?\.|vol\.|volume)\b", text, re.I)
    )
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", text))
    sentence_like_citation = bool(re.search(r"\[[Cc][Ii][Tt][Aa][Tt][Ii][Oo][Nn]\]", text))
    return starts_like_authors and venue_or_identifier and has_year and sentence_like_citation


def structural_failures(row: dict) -> list[str]:
    failures = [f"missing_{field}" for field in REQUIRED_FIELDS if not _normalized(row.get(field))]
    sentence = _normalized(row.get("raw_sentence"))
    marker = _normalized(row.get("citation_marker"))
    if sentence and marker and marker not in sentence:
        failures.append("citation_marker_absent_from_raw_sentence")
    if row.get("sentence_citation_count") != 1:
        failures.append("not_single_citation_context")
    if _looks_like_bibliography_entry(sentence):
        failures.append("bibliography_entry_used_as_passage")
    return failures


def review_flags(row: dict) -> list[str]:
    flags = detect_common_failures(row)
    sentence = _normalized(row.get("raw_sentence"))
    reference = _normalized(row.get("raw_reference"))
    if len(sentence) < 40:
        flags.append("very_short_passage")
    if sentence and sentence[-1] not in ".!?)]}”’\"'":
        flags.append("possibly_truncated_passage")
    if len(reference) > 1200:
        flags.append("reference_span_likely_contains_multiple_entries")
    return sorted(set(flags))


def strong_acceptance_evidence(row: dict) -> list[str]:
    """Return evidence sufficient to clear noisy-reference warnings."""
    title = re.sub(r"[^a-z0-9]+", " ", str(row.get("resolved_title") or "").lower()).strip()
    source_title = re.sub(
        r"[^a-z0-9]+", " ", str(row.get("source_paper_title") or "").lower()
    ).strip()
    # The mapped entry occurs first in raw_reference; later text is commonly
    # spillover from PDF column extraction. Limit evidence to its prefix.
    reference_prefix = re.sub(
        r"[^a-z0-9]+", " ", str(row.get("raw_reference") or "")[:900].lower()
    ).strip()
    marker = f"{row.get('citation_marker', '')} {row.get('reference_key', '')}"
    marker_years = [int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", marker)]
    resolved_year = row.get("resolved_year")
    year_agrees = (
        not marker_years
        or not isinstance(resolved_year, int)
        or min(abs(resolved_year - year) for year in marker_years) <= 1
    )
    if title and title != source_title and title in reference_prefix and year_agrees:
        return ["exact_resolved_title_in_reference_prefix", "citation_year_consistent"]
    query = re.sub(
        r"[^a-z0-9]+", " ", str(row.get("resolution_query") or "").lower()
    ).strip()
    query_similarity = SequenceMatcher(a=query, b=title).ratio() if query and title else 0.0
    marker_words = [
        word
        for word in re.sub(r"[^a-z]+", " ", str(row.get("citation_marker") or "").lower()).split()
        if len(word) > 2 and word not in {"and", "etal"}
    ]
    marker_author_present = not marker_words or marker_words[0] in reference_prefix
    if (
        title
        and title != source_title
        and query_similarity >= 0.93
        and marker_author_present
        and year_agrees
    ):
        return ["resolution_query_matches_resolved_title", "citation_year_consistent"]
    return []


def load_adjudications(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None:
        return {}
    result: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["source_file"]), str(item["id"]))
            if key in result:
                raise ValueError(f"Duplicate adjudication at line {number}: {key}")
            if item.get("action") not in {"accept", "revise", "reject"}:
                raise ValueError(f"Invalid adjudication action at line {number}")
            if item["action"] == "revise" and not item.get("revision"):
                raise ValueError(f"Revision missing at line {number}")
            result[key] = item
    return result


def load_confirmed_accepts(
    annotations_path: Path | None, sample_path: Path | None, split_path: Path | None
) -> dict[tuple[str, str], dict]:
    paths = (annotations_path, sample_path, split_path)
    if any(path is None or not path.exists() for path in paths):
        return {}
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    audit_to_dataset = {str(row["audit_id"]): str(row["dataset_id"]) for row in sample}
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split = {str(row["id"]): row for row in csv.DictReader(handle)}
    result = {}
    for audit_id, annotation in annotations.items():
        if annotation.get("overall_label_validity") != "Yes":
            continue
        dataset_id = audit_to_dataset.get(str(audit_id))
        row = split.get(str(dataset_id))
        if row is None:
            continue
        key = (str(row["source_file"]), str(row["original_id"]))
        result[key] = {
            "action": "accept",
            "reasons": ["confirmed_overall_label_validity", str(audit_id)],
            "authority": "confirmed_manual_audit",
        }
    return result


def decide(row: dict, source_file: str, adjudications: dict[tuple[str, str], dict]) -> Decision:
    key = (source_file, str(row.get("id", "")))
    if key in adjudications:
        item = adjudications[key]
        return Decision(
            item["action"],
            tuple(item.get("reasons") or ("external_adjudication",)),
            item.get("revision"),
            str(item.get("authority") or "external_adjudication"),
        )
    if key in EXCLUSIONS or row.get("manual_exclusion"):
        return Decision("reject", ("confirmed_invalid_context",), authority="confirmed_manual_audit")
    if key in CORRECTIONS:
        correction = CORRECTIONS[key]
        return Decision(
            "revise",
            (str(correction["reason"]),),
            {
                "resolved_paper_id": correction["paper_id"],
                "resolved_title": correction["title"],
                "resolved_year": correction["year"],
                "resolution_query": correction["title"],
                "resolution_score": None,
                "discoverability_rank": None,
                "resolution_method": "confirmed_manual_audit",
                "discoverability_verification": "canonical_target_confirmed_by_manual_audit",
            },
            "confirmed_manual_audit",
        )
    failures = structural_failures(row)
    if failures:
        return Decision("reject", tuple(failures))
    flags = review_flags(row)
    if flags:
        evidence = strong_acceptance_evidence(row)
        unsafe = {"resolved_to_source_paper", "citation_year_disagreement"}.intersection(flags)
        if evidence and not unsafe:
            return Decision("accept", tuple(evidence), authority="deterministic_cross_check")
        return Decision("review", tuple(flags))
    return Decision("accept", ())


def apply_revision(row: dict, decision: Decision) -> dict:
    revised = dict(row)
    if decision.revision:
        revised.update(decision.revision)
        revised["full_dataset_audit_revision"] = {
            "authority": decision.authority,
            "reasons": list(decision.reasons),
        }
    return revised


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def audit_file(
    audit_path: Path,
    output_dir: Path,
    adjudications: dict[tuple[str, str], dict],
) -> tuple[list[dict], list[dict], Counter]:
    source_file = audit_path.name.removesuffix(".audit.jsonl") + ".csv"
    retained: list[dict] = []
    decisions: list[dict] = []
    counts: Counter = Counter()
    with audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decision = decide(row, source_file, adjudications)
            counts[decision.action] += 1
            decisions.append(
                {
                    "source_file": source_file,
                    "id": str(row.get("id", "")),
                    "action": decision.action,
                    "authority": decision.authority,
                    "reasons": list(decision.reasons),
                    "revision": decision.revision,
                }
            )
            # Review records are quarantined from candidate outputs until an
            # explicit adjudication accepts, revises, or rejects them.
            if decision.action in {"accept", "revise"}:
                retained.append(apply_revision(row, decision))
    write_jsonl(output_dir / audit_path.name, retained)
    return retained, decisions, counts


def revise_candidate_csv(input_dir: Path, output_dir: Path, source_file: str, retained: list[dict]) -> None:
    input_path = input_dir / source_file
    if not input_path.exists():
        return
    by_id = {str(row["id"]): row for row in retained}
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    output_rows = []
    for row in rows:
        audit = by_id.get(str(row.get("id", "")))
        if audit is None:
            continue
        row["target_paper_title"] = str(audit.get("resolved_title") or row.get("target_paper_title") or "")
        paper_id = str(audit.get("resolved_paper_id") or "")
        if paper_id:
            row["target_paper_url"] = paper_url(paper_id)
        output_rows.append(row)
    output_path = output_dir / source_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


def run(
    input_dir: Path,
    output_dir: Path,
    adjudication_path: Path | None,
    annotations_path: Path | None = None,
    sample_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from input directory")
    adjudications = load_confirmed_accepts(annotations_path, sample_path, split_path)
    # Explicit decisions override imported confirmed accepts.
    adjudications.update(load_adjudications(adjudication_path))
    all_decisions: list[dict] = []
    totals: Counter = Counter()
    files = 0
    for audit_path in sorted(input_dir.glob("*.audit.jsonl")):
        retained, decisions, counts = audit_file(audit_path, output_dir, adjudications)
        source_file = audit_path.name.removesuffix(".audit.jsonl") + ".csv"
        revise_candidate_csv(input_dir, output_dir, source_file, retained)
        all_decisions.extend(decisions)
        totals.update(counts)
        files += 1

    used = {(item["source_file"], item["id"]) for item in all_decisions}
    unused = sorted(set(adjudications) - used)
    if unused:
        raise ValueError(f"Adjudications do not match input records: {unused[:10]}")

    write_jsonl(output_dir / "full_audit_decisions.jsonl", all_decisions)
    write_jsonl(
        output_dir / "full_audit_review_queue.jsonl",
        (item for item in all_decisions if item["action"] == "review"),
    )
    report = {
        "schema_version": 1,
        "policy": (
            "Confirmed revisions are applied; deterministic structural failures are rejected; "
            "ambiguous heuristic findings require review."
        ),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "files_audited": files,
        "records_audited": sum(totals.values()),
        "decision_counts": dict(sorted(totals.items())),
        "release_candidate_records": totals["accept"] + totals["revise"],
        "quarantined_review_records": totals["review"],
        "rejected_records": totals["reject"],
        "unresolved_review_records": totals["review"],
        "release_ready": totals["review"] == 0,
    }
    (output_dir / "full_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("training_data_collection/output"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_data_collection/audited"))
    parser.add_argument(
        "--adjudications",
        type=Path,
        help="Optional JSONL decisions with source_file, id, action, reasons, and revision",
    )
    parser.add_argument(
        "--confirmed-annotations",
        type=Path,
        default=Path("outputs/citation_audit_100/confirmed_annotations.json"),
    )
    parser.add_argument(
        "--audit-sample",
        type=Path,
        default=Path("outputs/citation_audit_100/audit_sample.json"),
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=Path("training_data_collection/splits/test_2025_balanced.csv"),
    )
    args = parser.parse_args()
    report = run(
        args.input_dir,
        args.output_dir,
        args.adjudications,
        args.confirmed_annotations,
        args.audit_sample,
        args.test_split,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["release_ready"] else 2)


if __name__ == "__main__":
    main()
