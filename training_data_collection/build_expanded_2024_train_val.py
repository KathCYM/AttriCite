"""Build decontaminated 1,000-train/200-validation 2024 splits.

Historical roles are immutable: every old training example stays in training,
every old validation example stays in validation, and the 2025 test set stays
unchanged. New rows are filtered at excerpt, source-title, and target-title
levels so no row can cross any historical or current split boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_data_collection.build_expanded_2024_train import (
    allocate_targets,
    candidate_key,
    fill_venue,
    read_keys,
)
from training_data_collection.build_year_splits import (
    DEFAULT_INPUTS,
    VENUES,
    Candidate,
    deduplicate,
    normalize_text,
    normalize_title,
    stable_score,
    summarize,
    read_csv,
    write_split,
)


def overlaps(rows: list[Candidate], excerpts: set[str], sources: set[str], targets: set[str]) -> dict[str, int]:
    return {
        "excerpts": len({row.excerpt_key for row in rows} & excerpts),
        "sources": len({row.source_key for row in rows} & sources),
        "targets": len({row.target_key for row in rows} & targets),
    }


def row_sets(rows: list[Candidate]) -> tuple[set[str], set[str], set[str]]:
    return (
        {row.excerpt_key for row in rows},
        {row.source_key for row in rows},
        {row.target_key for row in rows},
    )


def conflicts(row: Candidate, split_sets: tuple[set[str], set[str], set[str]]) -> bool:
    excerpts, sources, targets = split_sets
    return row.excerpt_key in excerpts or row.source_key in sources or row.target_key in targets


def build(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    old_train_path = Path(args.existing_train)
    old_validation_path = Path(args.existing_validation)
    test_path = Path(args.test)
    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    report_path = Path(args.report)

    old_train_rows, old_train_excerpts, old_train_sources, old_train_targets = read_keys(old_train_path)
    old_val_rows, old_val_excerpts, old_val_sources, old_val_targets = read_keys(old_validation_path)
    _, test_excerpts, test_sources, test_targets = read_keys(test_path)
    old_train_sets = (old_train_excerpts, old_train_sources, old_train_targets)
    old_val_sets = (old_val_excerpts, old_val_sources, old_val_targets)
    test_sets = (test_excerpts, test_sources, test_targets)

    old_train_keys = {
        (
            normalize_text(row["excerpt"]),
            normalize_title(row["source_paper_title"]),
            normalize_title(row["target_paper_title"]),
        )
        for row in old_train_rows
    }
    old_val_keys = {
        (
            normalize_text(row["excerpt"]),
            normalize_title(row["source_paper_title"]),
            normalize_title(row["target_paper_title"]),
        )
        for row in old_val_rows
    }

    pools: dict[str, list[Candidate]] = {venue: [] for venue in VENUES}
    seen_global: set[tuple[str, str, str]] = set()
    input_counts = {}
    duplicate_counts = {}
    for venue in VENUES:
        rows, _ = read_csv(input_dir / DEFAULT_INPUTS[(2024, venue)], venue, 2024)
        rows, exact_duplicates = deduplicate(rows)
        input_counts[venue] = len(rows)
        duplicate_counts[venue] = exact_duplicates
        for row in rows:
            key = candidate_key(row)
            if key in seen_global:
                duplicate_counts[venue] += 1
                continue
            seen_global.add(key)
            pools[venue].append(row)

    by_key = {candidate_key(row): row for venue in VENUES for row in pools[venue]}
    missing_train = old_train_keys - set(by_key)
    missing_val = old_val_keys - set(by_key)
    if missing_train or missing_val:
        raise AssertionError(
            f"Historical rows missing from candidate pool: train={len(missing_train)}, "
            f"validation={len(missing_val)}"
        )

    historical_train = [by_key[key] for key in old_train_keys]
    historical_validation = [by_key[key] for key in old_val_keys]
    historical_train_by_venue = {
        venue: [row for row in historical_train if row.venue == venue] for venue in VENUES
    }
    historical_val_by_venue = {
        venue: [row for row in historical_validation if row.venue == venue] for venue in VENUES
    }

    # Expand validation first. A new validation row may not overlap the old
    # training set or the fixed test set. Existing validation rows are retained.
    validation_pools: dict[str, list[Candidate]] = {}
    for venue in VENUES:
        retained = historical_val_by_venue[venue]
        retained_keys = {candidate_key(row) for row in retained}
        additions = [
            row for row in pools[venue]
            if candidate_key(row) not in retained_keys
            and not conflicts(row, old_train_sets)
            and not conflicts(row, test_sets)
        ]
        validation_pools[venue] = retained + additions

    val_capacities = {venue: len(validation_pools[venue]) for venue in VENUES}
    val_initial = {venue: len(historical_val_by_venue[venue]) for venue in VENUES}
    val_targets = allocate_targets(val_capacities, val_initial, args.validation_rows)
    validation: list[Candidate] = []
    for venue in VENUES:
        validation.extend(
            fill_venue(
                historical_val_by_venue[venue],
                validation_pools[venue],
                val_targets[venue],
                args.seed + 100 + VENUES.index(venue),
            )
        )
    validation.sort(key=lambda row: stable_score(row, args.seed + 200))
    validation_sets = row_sets(validation)

    # Expand training only after validation is fixed. No current training row
    # may overlap current validation or the fixed test set.
    train_pools: dict[str, list[Candidate]] = {}
    for venue in VENUES:
        train_pools[venue] = [
            row for row in pools[venue]
            if not conflicts(row, validation_sets) and not conflicts(row, test_sets)
        ]

    train_capacities = {venue: len(train_pools[venue]) for venue in VENUES}
    train_initial = {venue: len(historical_train_by_venue[venue]) for venue in VENUES}
    train_targets = allocate_targets(train_capacities, train_initial, args.train_rows)
    train: list[Candidate] = []
    for venue in VENUES:
        train.extend(
            fill_venue(
                historical_train_by_venue[venue],
                train_pools[venue],
                train_targets[venue],
                args.seed + 300 + VENUES.index(venue),
            )
        )
    train.sort(key=lambda row: stable_score(row, args.seed + 400))
    train_sets = row_sets(train)

    current_overlap = {
        "train_vs_validation": overlaps(train, *validation_sets),
        "train_vs_test": overlaps(train, *test_sets),
        "validation_vs_test": overlaps(validation, *test_sets),
    }
    historical_cross_role_overlap = {
        "current_train_vs_previous_validation": overlaps(train, *old_val_sets),
        "current_train_vs_previous_test": overlaps(train, *test_sets),
        "current_validation_vs_previous_train": overlaps(validation, *old_train_sets),
        "current_validation_vs_previous_test": overlaps(validation, *test_sets),
    }
    checks = [
        value
        for section in (current_overlap, historical_cross_role_overlap)
        for group in section.values()
        for value in group.values()
    ]
    if any(checks):
        raise AssertionError(
            f"Contamination detected: current={current_overlap}, historical={historical_cross_role_overlap}"
        )
    if not old_train_keys <= {candidate_key(row) for row in train}:
        raise AssertionError("Not every previous training row was preserved in current training")
    if not old_val_keys <= {candidate_key(row) for row in validation}:
        raise AssertionError("Not every previous validation row was preserved in current validation")

    write_split(train_output, train, "train", args.train_id_start)
    write_split(validation_output, validation, "validation", args.validation_id_start)
    report = {
        "policy": {
            "train_rows": args.train_rows,
            "validation_rows": args.validation_rows,
            "test_rows": 300,
            "year": 2024,
            "seed": args.seed,
            "historical_roles_preserved": True,
            "decontamination_keys": ["exact excerpt", "source paper title", "target paper title"],
        },
        "files": {
            "train": str(train_output),
            "validation": str(validation_output),
            "test_unchanged": str(test_path),
            "previous_train_unchanged": str(old_train_path),
            "previous_validation_unchanged": str(old_validation_path),
        },
        "input_candidates_after_dedup": input_counts,
        "duplicates_removed": duplicate_counts,
        "train_capacity_by_venue": train_capacities,
        "validation_capacity_by_venue": val_capacities,
        "train_target_by_venue": train_targets,
        "validation_target_by_venue": val_targets,
        "previous_train_rows_preserved": len(old_train_keys),
        "previous_validation_rows_preserved": len(old_val_keys),
        "train_summary": summarize(train),
        "validation_summary": summarize(validation),
        "current_overlap": current_overlap,
        "historical_cross_role_overlap": historical_cross_role_overlap,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="training_data_collection/output")
    parser.add_argument("--existing-train", default="training_data_collection/splits/train_2024_rl.csv")
    parser.add_argument("--existing-validation", default="training_data_collection/splits/validation_2024_rl.csv")
    parser.add_argument("--test", default="training_data_collection/splits/test_2025_balanced.csv")
    parser.add_argument("--train-output", default="training_data_collection/splits/train_2024_rl_1k.csv")
    parser.add_argument("--validation-output", default="training_data_collection/splits/validation_2024_rl_200.csv")
    parser.add_argument("--report", default="training_data_collection/splits/train_2024_rl_1k_val200_validation.json")
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--train-id-start", type=int, default=1_200_000)
    parser.add_argument("--validation-id-start", type=int, default=1_600_000)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({
        "train": result["train_summary"],
        "validation": result["validation_summary"],
        "current_overlap": result["current_overlap"],
        "historical_cross_role_overlap": result["historical_cross_role_overlap"],
    }, indent=2))
