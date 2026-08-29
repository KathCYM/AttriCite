"""Apply confirmed manual-audit corrections to CiteAlign data artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


CORRECTIONS = {
    ("iclr2025_candidates.csv", "100092"): {
        "audit_id": "A008",
        "title": "Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring",
        "paper_id": "03e008a4852de41f645546030ab081432b802922",
        "year": 2012,
        "reason": "entity_resolution",
    },
    ("icml2025_candidates.csv", "100045"): {
        "audit_id": "A012",
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "paper_id": "6c4b76232bb72897685d19b3d264c6ee3005bc2b",
        "year": 2020,
        "reason": "citation_mapping",
    },
    ("iclr2025_candidates.csv", "100077"): {
        "audit_id": "A018",
        "title": "Discovering and Exploring Cases of Educational Source Code Plagiarism with Dolos",
        "paper_id": "459722419437bca8da81e37725c981c74b4d2270",
        "year": 2024,
        "reason": "citation_mapping",
    },
    ("icml2025_candidates.csv", "100012"): {
        "audit_id": "A032",
        "title": "Vertex-Reinforced Random Walk",
        "paper_id": "ARXIV:math/0404041",
        "year": 2004,
        "reason": "entity_resolution",
    },
    ("icml2025_candidates.csv", "100063"): {
        "audit_id": "A034",
        "title": "Sharpness-Aware Minimization for Efficiently Improving Generalization",
        "paper_id": "a2cd073b57be744533152202989228cb4122270a",
        "year": 2020,
        "reason": "citation_mapping",
    },
    ("icml2025_candidates.csv", "100034"): {
        "audit_id": "A057",
        "title": "Partitioned Variational Inference: A Framework for Probabilistic Federated Learning",
        "paper_id": "06cfecfb904bb9d813ef9f1040ef784cdf151a54",
        "year": 2022,
        "reason": "entity_resolution",
    },
    ("icml2025_candidates.csv", "100073"): {
        "audit_id": "A058",
        "title": "Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers",
        "paper_id": "18fd98e23e6405a7f8977ecf9d6fd67731b0771d",
        "year": 2024,
        "reason": "entity_resolution",
    },
    ("icml2025_candidates.csv", "100001"): {
        "audit_id": "A065",
        "title": "Gradual Release of Sensitive Data under Differential Privacy",
        "paper_id": "ARXIV:1504.00429",
        "year": 2016,
        "reason": "entity_resolution",
    },
    ("icml2025_candidates.csv", "100024"): {
        "audit_id": "A071",
        "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        "paper_id": "66c10bf1f11bc1b2d92204d8f8391d087f6de1c4",
        "year": 2024,
        "reason": "entity_resolution",
    },
    ("iclr2025_candidates.csv", "100063"): {
        "audit_id": "A084",
        "title": "Learning-Based Frequency Estimation Algorithms",
        "paper_id": "OPENREVIEW:r1lohoCqY7",
        "year": 2019,
        "reason": "entity_resolution",
    },
}

EXCLUSIONS = {
    ("neurips2025_candidates.csv", "101068"): {
        "audit_id": "A083",
        "reason": "passage_extraction",
        "dataset_id": "2000252",
    },
}


def paper_url(paper_id: str) -> str:
    if paper_id.startswith("ARXIV:"):
        return f"https://arxiv.org/abs/{paper_id.removeprefix('ARXIV:')}"
    if paper_id.startswith("OPENREVIEW:"):
        return f"https://openreview.net/forum?id={paper_id.removeprefix('OPENREVIEW:')}"
    if paper_id.startswith("DOI:"):
        return f"https://doi.org/{paper_id.removeprefix('DOI:')}"
    if paper_id.startswith("REPEC:"):
        return f"https://ideas.repec.org/{paper_id.removeprefix('REPEC:')}"
    if paper_id.startswith("URL:"):
        return paper_id.removeprefix("URL:")
    return f"https://www.semanticscholar.org/paper/{paper_id}"


def _normalized_title(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def detect_common_failures(row: dict) -> list[str]:
    """Return conservative review flags; never infer a replacement target."""
    flags: list[str] = []
    source_title = _normalized_title(row.get("source_paper_title"))
    resolved_title = _normalized_title(row.get("resolved_title"))
    if source_title and source_title == resolved_title:
        flags.append("resolved_to_source_paper")

    marker = f"{row.get('citation_marker', '')} {row.get('reference_key', '')}"
    marker_years = [int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", marker)]
    resolved_year = row.get("resolved_year")
    if marker_years and isinstance(resolved_year, int) and min(abs(resolved_year - y) for y in marker_years) > 1:
        flags.append("citation_year_disagreement")

    score = row.get("resolution_score")
    if isinstance(score, (int, float)) and score < 0.65:
        flags.append("low_resolution_score")

    raw_reference = str(row.get("raw_reference") or "")
    reference_spillover = len(raw_reference) > 1200 or len(re.findall(r"\b(?:doi|url|arxiv)\b", raw_reference, re.I)) >= 4
    if reference_spillover:
        flags.append("bibliography_span_spillover")
    if reference_spillover and isinstance(score, (int, float)) and score < 0.80:
        flags.append("spillover_with_weak_match")
    return flags


def write_quality_report(root: Path, output_path: Path) -> dict:
    """Write a release-safe report containing identifiers and flags, not source text."""
    candidates: list[dict] = []
    counts: Counter[str] = Counter()
    audit_dir = root / "training_data_collection" / "output"
    for path in sorted(audit_dir.glob("*.audit.jsonl")):
        source_file = path.name.removesuffix(".audit.jsonl") + ".csv"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                flags = detect_common_failures(row)
                counts.update(flags)
                actionable_flags = [flag for flag in flags if flag != "bibliography_span_spillover"]
                if actionable_flags:
                    candidates.append(
                        {
                            "source_file": source_file,
                            "id": str(row.get("id", "")),
                            "source_paper_title": row.get("source_paper_title"),
                            "resolved_title": row.get("resolved_title"),
                            "resolved_year": row.get("resolved_year"),
                            "resolution_score": row.get("resolution_score"),
                            "flags": actionable_flags,
                            "confirmed_correction": CORRECTIONS.get((source_file, str(row.get("id", ""))), {}).get("audit_id"),
                        }
                    )
    report = {
        "schema_version": 1,
        "policy": "Heuristics nominate records for review; only human-confirmed entries are corrected.",
        "confirmed_corrections": len(CORRECTIONS),
        "flag_counts": dict(sorted(counts.items())),
        "review_candidate_count": len(candidates),
        "review_candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def update_validation_report(path: Path, quality_report: dict) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["confirmed_manual_corrections"] = len(CORRECTIONS)
    report["resolution_review_candidates"] = quality_report["review_candidate_count"]
    report["resolution_review_flag_counts"] = quality_report["flag_counts"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_csv(path: Path, source_filename: str | None = None) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    changed = 0
    retained = []
    for row in rows:
        source_file = source_filename or row.get("source_file", "")
        original_id = row.get("original_id") or row.get("id", "")
        if (source_file, str(original_id)) in EXCLUSIONS:
            changed += 1
            continue
        correction = CORRECTIONS.get((source_file, str(original_id)))
        if correction is not None:
            row["target_paper_title"] = correction["title"]
            row["target_paper_url"] = paper_url(correction["paper_id"])
            changed += 1
        retained.append(row)
    rows = retained
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return changed


def rewrite_audit_jsonl(path: Path, source_filename: str) -> int:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    changed = 0
    for row in rows:
        exclusion = EXCLUSIONS.get((source_filename, str(row.get("id", ""))))
        if exclusion is not None:
            row["manual_exclusion"] = {
                "audit_id": exclusion["audit_id"],
                "reason": exclusion["reason"],
                "release_status": "excluded_invalid_citation_context",
            }
            changed += 1
            continue
        correction = CORRECTIONS.get((source_filename, str(row.get("id", ""))))
        if correction is None:
            continue
        if "manual_correction" not in row:
            row["manual_correction"] = {
                "audit_id": correction["audit_id"],
                "reason": correction["reason"],
                "original_resolved_paper_id": row.get("resolved_paper_id"),
                "original_resolved_title": row.get("resolved_title"),
                "original_resolved_year": row.get("resolved_year"),
                "original_resolution_query": row.get("resolution_query"),
                "original_resolution_score": row.get("resolution_score"),
                "original_discoverability_rank": row.get("discoverability_rank"),
            }
        row.update(
            {
                "resolved_paper_id": correction["paper_id"],
                "resolved_title": correction["title"],
                "resolved_year": correction["year"],
                "resolution_query": correction["title"],
                "resolution_score": None,
                "resolution_method": "confirmed_manual_audit",
                "discoverability_rank": None,
                "discoverability_verification": "canonical_target_confirmed_by_manual_audit",
            }
        )
        changed += 1
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    return changed


def rewrite_release(path: Path, root: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    original_audits: dict[tuple[str, str, str], dict] = {}
    for audit_path in (root / "training_data_collection" / "output").glob("*.audit.jsonl"):
        year = "2024" if "2024" in audit_path.name or audit_path.name == "cvpr_single.audit.jsonl" else "2025"
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                audit_row = json.loads(line)
                original_audits[(year, str(audit_row.get("id")), str(audit_row.get("source_paper_title")))] = audit_row
    changed = 0
    retained = []
    for row in rows:
        source_file = {
            "ICLR": "iclr2025_candidates.csv",
            "ICML": "icml2025_candidates.csv",
            "NeurIPS": "neurips2025_candidates.csv",
        }.get(str(row.get("venue")))
        if (
            row.get("experimental_split") == "test"
            and (source_file or "", str(row.get("id", ""))) in EXCLUSIONS
        ):
            changed += 1
            continue
        correction_audit_ids = {item["audit_id"] for item in CORRECTIONS.values()}
        if row.get("experimental_split") != "test" and row.get("manual_correction", {}).get("audit_id") in correction_audit_ids:
            original = original_audits.get(
                (str(row.get("year")), str(row.get("id")), str(row.get("source_paper_title")))
            )
            if original is None:
                raise KeyError(f"Cannot restore non-test collision for release row {row.get('release_id')}")
            for field in (
                "resolved_paper_id", "resolved_title", "resolved_year", "resolution_query",
                "resolution_score", "discoverability_rank",
            ):
                row[field] = original.get(field)
            row.pop("manual_correction", None)
            row.pop("resolution_method", None)
            row.pop("discoverability_verification", None)
        correction = (
            CORRECTIONS.get((source_file or "", str(row.get("id", ""))))
            if row.get("experimental_split") == "test"
            else None
        )
        if correction is None:
            retained.append(row)
            continue
        if "manual_correction" not in row:
            row["manual_correction"] = {
                "audit_id": correction["audit_id"],
                "reason": correction["reason"],
                "original_resolved_paper_id": row.get("resolved_paper_id"),
                "original_resolved_title": row.get("resolved_title"),
                "original_resolved_year": row.get("resolved_year"),
                "original_resolution_score": row.get("resolution_score"),
                "original_discoverability_rank": row.get("discoverability_rank"),
            }
        row.update(
            {
                "resolved_paper_id": correction["paper_id"],
                "resolved_title": correction["title"],
                "resolved_year": correction["year"],
                "resolution_query": correction["title"],
                "resolution_score": None,
                "resolution_method": "confirmed_manual_audit",
                "discoverability_rank": None,
                "discoverability_verification": "canonical_target_confirmed_by_manual_audit",
            }
        )
        changed += 1
        retained.append(row)
    rows = retained
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--scan-only", action="store_true", help="Only refresh the review report")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("release/resolution_quality_report.json"),
        help="Path relative to --root for the release-safe review report",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    report_path = args.report if args.report.is_absolute() else root / args.report
    if args.scan_only:
        report = write_quality_report(root, report_path)
        validation_path = root / "release" / "validation_report.json"
        if validation_path.exists():
            update_validation_report(validation_path, report)
        print(json.dumps({"report": str(report_path), "review_candidates": report["review_candidate_count"]}, indent=2))
        return
    changes: dict[str, int] = {}
    for source_file in {key[0] for key in CORRECTIONS} | {key[0] for key in EXCLUSIONS}:
        csv_path = root / "training_data_collection" / "output" / source_file
        audit_path = csv_path.with_suffix(".audit.jsonl")
        changes[str(csv_path)] = rewrite_csv(csv_path, source_file)
        changes[str(audit_path)] = rewrite_audit_jsonl(audit_path, source_file)
    split_path = root / "training_data_collection" / "splits" / "test_2025_balanced.csv"
    changes[str(split_path)] = rewrite_csv(split_path)
    release_path = root / "release" / "citealign_metadata.jsonl"
    changes[str(release_path)] = rewrite_release(release_path, root)
    print(json.dumps(changes, indent=2, sort_keys=True))
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        remaining_ids = {str(row["id"]) for row in csv.DictReader(handle)}
    excluded_ids = {item["dataset_id"] for item in EXCLUSIONS.values()}
    if remaining_ids & excluded_ids:
        raise SystemExit(f"Excluded audit IDs remain in test split: {sorted(remaining_ids & excluded_ids)}")
    quality_report = write_quality_report(root, report_path)
    update_validation_report(root / "release" / "validation_report.json", quality_report)
    print(json.dumps({"quality_report": str(report_path), "review_candidates": quality_report["review_candidate_count"]}, indent=2))


if __name__ == "__main__":
    main()
