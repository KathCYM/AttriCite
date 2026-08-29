"""Build deterministic, venue-balanced CiteGuard train/test datasets.

The default policy uses 2024 source papers for training and 2025 source papers
for testing. Test targets are required to be absent from the complete eligible
2024 pool, making the test set stricter than a source-year-only split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VENUES = ("ACL", "CVPR", "ICLR", "ICML", "NeurIPS")
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
OUTPUT_COLUMNS = EXPECTED_COLUMNS + ("venue", "original_id", "source_file")

DEFAULT_INPUTS = {
    (2024, "ACL"): "acl2024_candidates.csv",
    (2024, "CVPR"): "cvpr_single.csv",
    (2024, "ICLR"): "iclr2024_candidates.csv",
    (2024, "ICML"): "icml2024_candidates.csv",
    (2024, "NeurIPS"): "neurips2024_candidates.csv",
    (2025, "ACL"): "acl2025_candidates.csv",
    (2025, "CVPR"): "cvpr2025_candidates.csv",
    (2025, "ICLR"): "iclr2025_candidates.csv",
    (2025, "ICML"): "icml2025_candidates.csv",
    (2025, "NeurIPS"): "neurips2025_candidates.csv",
}


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    venue: str
    year: int
    source_file: str

    @property
    def excerpt_key(self) -> str:
        return normalize_text(self.row["excerpt"])

    @property
    def source_key(self) -> str:
        return normalize_title(self.row["source_paper_title"])

    @property
    def target_key(self) -> str:
        return normalize_title(self.row["target_paper_title"])


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def stable_score(candidate: Candidate, seed: int) -> str:
    payload = "\x1f".join(
        (
            str(seed),
            candidate.venue,
            candidate.source_key,
            candidate.target_key,
            candidate.excerpt_key,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path: Path, venue: str, year: int) -> tuple[list[Candidate], dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing_columns = sorted(set(EXPECTED_COLUMNS) - set(columns))
        extra_columns = sorted(set(columns) - set(EXPECTED_COLUMNS))
        raw_rows = list(reader)

    problems: Counter[str] = Counter()
    seen_ids: set[str] = set()
    candidates: list[Candidate] = []
    for row in raw_rows:
        required = ("id", "excerpt", "target_paper_title", "source_paper_title", "year", "split")
        if any(not (row.get(column) or "").strip() for column in required):
            problems["missing_required_value"] += 1
            continue
        if row["id"] in seen_ids:
            problems["duplicate_id_within_file"] += 1
            continue
        seen_ids.add(row["id"])
        try:
            row_year = int(row["year"])
        except ValueError:
            problems["invalid_year"] += 1
            continue
        if row_year != year:
            problems["wrong_year"] += 1
            continue
        if row["excerpt"].count("[CITATION]") != 1:
            problems["citation_marker_count_not_one"] += 1
            continue
        candidates.append(Candidate(row=row, venue=venue, year=year, source_file=path.name))

    audit_path = path.with_name(path.name.replace(".csv", ".audit.jsonl"))
    audit_info = validate_audit(audit_path, candidates)
    return candidates, {
        "file": path.name,
        "year": year,
        "venue": venue,
        "columns": list(columns),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "raw_rows": len(raw_rows),
        "eligible_rows_before_dedup": len(candidates),
        "problems": dict(sorted(problems.items())),
        "audit": audit_info,
    }


def validate_audit(path: Path, candidates: Iterable[Candidate]) -> dict:
    if not path.exists():
        return {"file": path.name, "exists": False}
    audit_rows = []
    invalid_json_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                audit_rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_json_lines += 1
    csv_ids = {candidate.row["id"] for candidate in candidates}
    audit_ids = {str(row.get("id", "")) for row in audit_rows}
    return {
        "file": path.name,
        "exists": True,
        "rows": len(audit_rows),
        "invalid_json_lines": invalid_json_lines,
        "eligible_csv_ids_missing_from_audit": len(csv_ids - audit_ids),
        "audit_ids_absent_from_eligible_csv": len(audit_ids - csv_ids),
    }


def deduplicate(candidates: Iterable[Candidate]) -> tuple[list[Candidate], int]:
    result = []
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for candidate in candidates:
        key = (candidate.excerpt_key, candidate.source_key, candidate.target_key)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        result.append(candidate)
    return result, duplicates


def diverse_sample(
    candidates: Iterable[Candidate],
    quota: int,
    seed: int,
) -> list[Candidate]:
    remaining = list(candidates)
    if len(remaining) < quota:
        raise ValueError(f"Only {len(remaining)} candidates available for quota {quota}")
    selected: list[Candidate] = []
    source_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scores = {id(candidate): stable_score(candidate, seed) for candidate in remaining}

    # Greedily minimize source and target reuse. The stable hash is only the
    # deterministic tie-breaker, so prolific papers cannot dominate by chance.
    while len(selected) < quota:
        candidate = min(
            remaining,
            key=lambda row: (
                source_counts[row.source_key],
                target_counts[row.target_key],
                scores[id(row)],
            ),
        )
        selected.append(candidate)
        source_counts[candidate.source_key] += 1
        target_counts[candidate.target_key] += 1
        remaining.remove(candidate)
    return selected


def write_split(path: Path, rows: list[Candidate], split: str, id_start: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for offset, candidate in enumerate(rows):
            output = {column: candidate.row.get(column, "") for column in EXPECTED_COLUMNS}
            output.update(
                {
                    "id": str(id_start + offset),
                    "split": split,
                    "venue": candidate.venue,
                    "original_id": candidate.row["id"],
                    "source_file": candidate.source_file,
                }
            )
            writer.writerow(output)


def summarize(rows: list[Candidate]) -> dict:
    return {
        "rows": len(rows),
        "by_venue": dict(sorted(Counter(row.venue for row in rows).items())),
        "unique_excerpts": len({row.excerpt_key for row in rows}),
        "unique_source_papers": len({row.source_key for row in rows}),
        "unique_target_papers": len({row.target_key for row in rows}),
        "max_rows_per_source": max(Counter(row.source_key for row in rows).values(), default=0),
        "max_rows_per_target": max(Counter(row.target_key for row in rows).values(), default=0),
    }


def make_rl_train_validation_split(
    rows: list[Candidate], validation_per_venue: int, seed: int
) -> tuple[list[Candidate], list[Candidate]]:
    """Hold out a balanced validation set while minimizing paper overlap."""
    rl_train: list[Candidate] = []
    validation: list[Candidate] = []
    for venue in VENUES:
        venue_rows = [row for row in rows if row.venue == venue]
        if len(venue_rows) <= validation_per_venue:
            raise ValueError(
                f"{venue} has {len(venue_rows)} rows, not enough for "
                f"{validation_per_venue} validation rows plus RL training"
            )
        source_frequency = Counter(row.source_key for row in venue_rows)
        target_frequency = Counter(row.target_key for row in venue_rows)
        ordered = sorted(
            venue_rows,
            key=lambda row: (
                source_frequency[row.source_key],
                target_frequency[row.target_key],
                stable_score(row, seed),
            ),
        )
        held_out: list[Candidate] = []
        held_out_sources: set[str] = set()
        held_out_targets: set[str] = set()
        for row in ordered:
            if row.source_key in held_out_sources or row.target_key in held_out_targets:
                continue
            held_out.append(row)
            held_out_sources.add(row.source_key)
            held_out_targets.add(row.target_key)
            if len(held_out) == validation_per_venue:
                break
        if len(held_out) < validation_per_venue:
            for row in ordered:
                if row in held_out:
                    continue
                held_out.append(row)
                if len(held_out) == validation_per_venue:
                    break
        validation.extend(held_out)
        rl_train.extend(row for row in venue_rows if row not in held_out)
    rl_train.sort(key=lambda row: stable_score(row, seed + 1))
    validation.sort(key=lambda row: stable_score(row, seed + 2))
    return rl_train, validation


def build(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pools: dict[tuple[int, str], list[Candidate]] = {}
    input_reports = []
    duplicate_reports = {}

    for (year, venue), filename in DEFAULT_INPUTS.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
        rows, report = read_csv(path, venue, year)
        if report["missing_columns"]:
            raise ValueError(f"{filename} missing columns: {report['missing_columns']}")
        rows, duplicate_count = deduplicate(rows)
        pools[(year, venue)] = rows
        report["eligible_rows_after_dedup"] = len(rows)
        report["exact_duplicate_rows_removed"] = duplicate_count
        input_reports.append(report)
        duplicate_reports[f"{year}_{venue}"] = duplicate_count

    full_2024_targets = {
        row.target_key
        for venue in VENUES
        for row in pools[(2024, venue)]
    }

    test_rows: list[Candidate] = []
    test_eligible_counts = {}
    for venue in VENUES:
        eligible = [
            row for row in pools[(2025, venue)]
            if row.target_key not in full_2024_targets
        ]
        test_eligible_counts[venue] = len(eligible)
        test_rows.extend(diverse_sample(eligible, args.test_per_venue, args.seed + 2025))

    test_targets = {row.target_key for row in test_rows}
    test_excerpts = {row.excerpt_key for row in test_rows}
    test_sources = {row.source_key for row in test_rows}

    train_rows: list[Candidate] = []
    train_eligible_counts = {}
    for venue in VENUES:
        eligible = [
            row for row in pools[(2024, venue)]
            if row.target_key not in test_targets
            and row.excerpt_key not in test_excerpts
            and row.source_key not in test_sources
        ]
        train_eligible_counts[venue] = len(eligible)
        train_rows.extend(diverse_sample(eligible, args.train_per_venue, args.seed + 2024))

    # Stable global order makes IDs reproducible while retaining venue balance.
    train_rows.sort(key=lambda row: stable_score(row, args.seed + 1))
    test_rows.sort(key=lambda row: stable_score(row, args.seed + 2))

    train_path = output_dir / "train_2024_balanced.csv"
    test_path = output_dir / "test_2025_balanced.csv"
    write_split(train_path, train_rows, "train", 1_000_000)
    write_split(test_path, test_rows, "test", 2_000_000)

    rl_train_rows, validation_rows = make_rl_train_validation_split(
        train_rows, args.validation_per_venue, args.seed + 3000
    )
    rl_train_path = output_dir / "train_2024_rl.csv"
    validation_path = output_dir / "validation_2024_rl.csv"
    write_split(rl_train_path, rl_train_rows, "train", 1_100_000)
    write_split(validation_path, validation_rows, "validation", 1_500_000)

    benchmark_overlap = {"train": 0, "test": 0}
    benchmark_path = Path(args.benchmark)
    if benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8-sig", newline="") as handle:
            benchmark_excerpts = {
                normalize_text(row.get("excerpt", "")) for row in csv.DictReader(handle)
            }
        benchmark_overlap = {
            "train": len({row.excerpt_key for row in train_rows} & benchmark_excerpts),
            "test": len({row.excerpt_key for row in test_rows} & benchmark_excerpts),
        }

    overlap = {
        "exact_excerpts": len({row.excerpt_key for row in train_rows} & test_excerpts),
        "source_paper_titles": len({row.source_key for row in train_rows} & test_sources),
        "target_paper_titles": len({row.target_key for row in train_rows} & test_targets),
    }
    rl_validation_overlap = {
        "exact_excerpts": len(
            {row.excerpt_key for row in rl_train_rows}
            & {row.excerpt_key for row in validation_rows}
        ),
        "source_paper_titles": len(
            {row.source_key for row in rl_train_rows}
            & {row.source_key for row in validation_rows}
        ),
        "target_paper_titles": len(
            {row.target_key for row in rl_train_rows}
            & {row.target_key for row in validation_rows}
        ),
    }
    if any(overlap.values()) or any(benchmark_overlap.values()):
        raise AssertionError(
            f"Leakage check failed: split={overlap}, benchmark={benchmark_overlap}"
        )

    report = {
        "policy": {
            "seed": args.seed,
            "train_source_year": 2024,
            "test_source_year": 2025,
            "venues": list(VENUES),
            "train_per_venue": args.train_per_venue,
            "test_per_venue": args.test_per_venue,
            "validation_per_venue": args.validation_per_venue,
            "test_targets_must_be_absent_from_full_2024_pool": True,
            "sampling": "greedy minimum source/target reuse with stable hash tie-break",
        },
        "inputs": input_reports,
        "eligible_after_leakage_filters": {
            "train_2024": train_eligible_counts,
            "test_2025_target_novel": test_eligible_counts,
        },
        "outputs": {
            "train": {"file": str(train_path), **summarize(train_rows)},
            "rl_train": {"file": str(rl_train_path), **summarize(rl_train_rows)},
            "validation": {"file": str(validation_path), **summarize(validation_rows)},
            "test": {"file": str(test_path), **summarize(test_rows)},
        },
        "leakage": {
            "between_train_and_test": overlap,
            "between_rl_train_and_validation": rl_validation_overlap,
            "with_existing_benchmark": benchmark_overlap,
        },
    }
    report_path = output_dir / "split_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="training_data_collection/output")
    parser.add_argument("--output_dir", default="training_data_collection/splits")
    parser.add_argument("--benchmark", default="DATASET.csv")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--train_per_venue", type=int, default=82)
    parser.add_argument("--test_per_venue", type=int, default=60)
    parser.add_argument("--validation_per_venue", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["outputs"], indent=2))
