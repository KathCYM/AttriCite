"""Build a deterministic biomedical CiteGuard test set from PubMed candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_COLUMNS = (
    "id",
    "excerpt",
    "target_paper_title",
    "target_paper_url",
    "source_paper_title",
    "source_paper_url",
    "year",
    "split",
)
OUTPUT_COLUMNS = EXPECTED_COLUMNS + (
    "venue",
    "domain",
    "original_id",
    "source_file",
)
DEFAULT_INPUTS = (
    "training_data_collection/output/pubmed_imaging_2025_single.csv",
    "training_data_collection/output/pubmed_cancer_genomics_2025_single.csv",
)
DEFAULT_TRAIN_REFERENCES = (
    "DATASET.csv",
    "training_data_collection/splits/train_2024_balanced.csv",
    "training_data_collection/splits/train_2024_rl.csv",
    "training_data_collection/splits/train_2024_rl_1k.csv",
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        normalize_text(row["excerpt"]),
        normalize_title(row["source_paper_title"]),
        normalize_title(row["target_paper_title"]),
    )


def stable_score(row: dict[str, str], seed: int) -> str:
    payload = "\x1f".join((str(seed), *row_key(row), row["source_file"], row["original_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_audit_metadata(csv_path: Path) -> tuple[dict[str, dict[str, str]], dict]:
    audit_path = csv_path.with_name(csv_path.name.replace(".csv", ".audit.jsonl"))
    metadata: dict[str, dict[str, str]] = {}
    invalid_lines = 0
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                metadata[str(item.get("id", ""))] = {
                    "venue": str(item.get("venue") or ""),
                    "domain": str(item.get("domain") or "Biomedical Research"),
                }
    return metadata, {
        "file": audit_path.name,
        "exists": audit_path.exists(),
        "rows": len(metadata),
        "invalid_json_lines": invalid_lines,
    }


def read_candidates(path: Path) -> tuple[list[dict[str, str]], dict]:
    audit_metadata, audit_report = load_audit_metadata(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        raw_rows = list(reader)
    problems: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        if any(not (raw.get(key) or "").strip() for key in (
            "id", "excerpt", "target_paper_title", "source_paper_title", "year"
        )):
            problems["missing_required_value"] += 1
            continue
        if raw["excerpt"].count("[CITATION]") != 1:
            problems["citation_marker_count_not_one"] += 1
            continue
        if raw["year"] != "2025":
            problems["year_not_2025"] += 1
            continue
        metadata = audit_metadata.get(raw["id"], {})
        row = {column: raw.get(column, "") for column in EXPECTED_COLUMNS}
        row.update(
            {
                "venue": metadata.get("venue", ""),
                "domain": metadata.get("domain", "Biomedical Research"),
                "original_id": raw["id"],
                "source_file": path.name,
            }
        )
        if not row["venue"]:
            problems["missing_audit_venue"] += 1
        rows.append(row)
    return rows, {
        "file": path.name,
        "columns": list(columns),
        "missing_columns": sorted(set(EXPECTED_COLUMNS) - set(columns)),
        "raw_rows": len(raw_rows),
        "eligible_rows": len(rows),
        "problems": dict(sorted(problems.items())),
        "audit": audit_report,
    }


def load_training_keys(paths: list[Path]) -> tuple[set[str], set[str], set[str], list[dict]]:
    excerpts: set[str] = set()
    sources: set[str] = set()
    targets: set[str] = set()
    reports = []
    for path in paths:
        if not path.exists():
            reports.append({"file": str(path), "exists": False})
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        excerpts.update(normalize_text(row.get("excerpt", "")) for row in rows if row.get("excerpt"))
        sources.update(
            normalize_title(row.get("source_paper_title", ""))
            for row in rows
            if row.get("source_paper_title")
        )
        targets.update(
            normalize_title(row.get("target_paper_title", ""))
            for row in rows
            if row.get("target_paper_title")
        )
        reports.append({"file": str(path), "exists": True, "rows": len(rows)})
    return excerpts, sources, targets, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--train_reference", action="append", dest="train_references")
    parser.add_argument(
        "--output_csv", default="training_data_collection/splits/test_2025_biomed.csv"
    )
    parser.add_argument(
        "--validation_json",
        default="training_data_collection/splits/test_2025_biomed_validation.json",
    )
    parser.add_argument("--starting_id", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    input_paths = [Path(value) for value in (args.inputs or DEFAULT_INPUTS)]
    train_paths = [Path(value) for value in (args.train_references or DEFAULT_TRAIN_REFERENCES)]
    all_rows: list[dict[str, str]] = []
    input_reports = []
    for path in input_paths:
        rows, report = read_candidates(path)
        all_rows.extend(rows)
        input_reports.append(report)

    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    for row in all_rows:
        key = row_key(row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduplicated.append(row)

    train_excerpts, train_sources, train_targets, train_reports = load_training_keys(train_paths)
    leakage_counts: Counter[str] = Counter()
    clean_rows = []
    for row in deduplicated:
        leaked = False
        if normalize_text(row["excerpt"]) in train_excerpts:
            leakage_counts["excerpt"] += 1
            leaked = True
        if normalize_title(row["source_paper_title"]) in train_sources:
            leakage_counts["source_title"] += 1
            leaked = True
        if normalize_title(row["target_paper_title"]) in train_targets:
            leakage_counts["target_title"] += 1
            leaked = True
        if not leaked:
            clean_rows.append(row)

    clean_rows.sort(key=lambda row: stable_score(row, args.seed))
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for offset, row in enumerate(clean_rows):
            output = dict(row)
            output["id"] = str(args.starting_id + offset)
            output["split"] = "test"
            writer.writerow(output)

    source_counts = Counter(normalize_title(row["source_paper_title"]) for row in clean_rows)
    target_counts = Counter(normalize_title(row["target_paper_title"]) for row in clean_rows)
    validation = {
        "seed": args.seed,
        "starting_id": args.starting_id,
        "inputs": input_reports,
        "training_references": train_reports,
        "raw_eligible_rows": len(all_rows),
        "exact_duplicates_removed": duplicate_count,
        "rows_removed_for_training_leakage": sum(
            1
            for row in deduplicated
            if normalize_text(row["excerpt"]) in train_excerpts
            or normalize_title(row["source_paper_title"]) in train_sources
            or normalize_title(row["target_paper_title"]) in train_targets
        ),
        "training_leakage_matches_by_type": dict(sorted(leakage_counts.items())),
        "output": {
            "file": str(output_path),
            "rows": len(clean_rows),
            "unique_excerpts": len({normalize_text(row["excerpt"]) for row in clean_rows}),
            "unique_source_papers": len(source_counts),
            "unique_target_papers": len(target_counts),
            "max_rows_per_source": max(source_counts.values(), default=0),
            "max_rows_per_target": max(target_counts.values(), default=0),
            "by_source_file": dict(sorted(Counter(row["source_file"] for row in clean_rows).items())),
            "by_venue": dict(sorted(Counter(row["venue"] for row in clean_rows).items())),
            "citation_marker_count_valid": all(row["excerpt"].count("[CITATION]") == 1 for row in clean_rows),
            "all_split_test": True,
            "all_year_2025": all(row["year"] == "2025" for row in clean_rows),
        },
    }
    validation_path = Path(args.validation_json)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"Wrote {len(clean_rows)} biomedical test rows to {output_path}")
    print(f"Wrote validation report to {validation_path}")


if __name__ == "__main__":
    main()
