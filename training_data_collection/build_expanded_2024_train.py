"""Build a larger 2024 RL training split without replacing existing splits.

The builder preserves every row in the current RL training set, excludes all
excerpt/source/target overlap with the fixed validation and 2025 test sets, and
then fills venue quotas as evenly as the collected candidate pools permit.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from training_data_collection.build_year_splits import (
    DEFAULT_INPUTS,
    VENUES,
    Candidate,
    deduplicate,
    normalize_text,
    normalize_title,
    read_csv,
    stable_score,
    summarize,
    write_split,
)


def read_keys(path: Path) -> tuple[list[dict[str, str]], set[str], set[str], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # A small number of PDF-derived excerpts contain embedded NUL bytes.
        # Python's csv parser rejects those bytes even though the surrounding
        # file is valid UTF-8. Remove only NULs in memory; never rewrite the
        # fixed historical split files.
        rows = list(csv.DictReader(line.replace("\0", "") for line in handle))
    return (
        rows,
        {normalize_text(row["excerpt"]) for row in rows},
        {normalize_title(row["source_paper_title"]) for row in rows},
        {normalize_title(row["target_paper_title"]) for row in rows},
    )


def candidate_key(row: Candidate) -> tuple[str, str, str]:
    return row.excerpt_key, row.source_key, row.target_key


def allocate_targets(capacities: dict[str, int], initial: dict[str, int], total: int) -> dict[str, int]:
    targets = dict(initial)
    if sum(targets.values()) > total:
        raise ValueError(f"Existing training split already has {sum(targets.values())} rows > target {total}")
    if sum(capacities.values()) < total:
        raise ValueError(f"Only {sum(capacities.values())} eligible candidates remain for target {total}")

    # Water filling: repeatedly grow the currently smallest venue that still
    # has capacity. This is the closest feasible allocation to equal balance.
    while sum(targets.values()) < total:
        eligible = [venue for venue in VENUES if targets[venue] < capacities[venue]]
        if not eligible:
            raise ValueError("Candidate capacity exhausted before reaching target size")
        venue = min(eligible, key=lambda name: (targets[name], VENUES.index(name)))
        targets[venue] += 1
    return targets


def fill_venue(
    existing: list[Candidate],
    candidates: list[Candidate],
    target: int,
    seed: int,
) -> list[Candidate]:
    selected = list(existing)
    selected_keys = {candidate_key(row) for row in selected}
    remaining = [row for row in candidates if candidate_key(row) not in selected_keys]
    source_counts = Counter(row.source_key for row in selected)
    target_counts = Counter(row.target_key for row in selected)
    scores = {id(row): stable_score(row, seed) for row in remaining}

    while len(selected) < target:
        if not remaining:
            raise ValueError(f"Unable to fill venue quota {target}; stopped at {len(selected)}")
        row = min(
            remaining,
            key=lambda item: (
                source_counts[item.source_key],
                target_counts[item.target_key],
                scores[id(item)],
            ),
        )
        selected.append(row)
        source_counts[row.source_key] += 1
        target_counts[row.target_key] += 1
        remaining.remove(row)
    return selected


def build(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    old_train_path = Path(args.existing_train)
    validation_path = Path(args.validation)
    test_path = Path(args.test)
    output_path = Path(args.output)
    report_path = Path(args.report)

    old_rows, old_excerpts, old_sources, old_targets = read_keys(old_train_path)
    _, val_excerpts, val_sources, val_targets = read_keys(validation_path)
    _, test_excerpts, test_sources, test_targets = read_keys(test_path)

    old_keys = {
        (
            normalize_text(row["excerpt"]),
            normalize_title(row["source_paper_title"]),
            normalize_title(row["target_paper_title"]),
        )
        for row in old_rows
    }

    pools: dict[str, list[Candidate]] = {}
    rejected = {}
    seen_global: set[tuple[str, str, str]] = set()
    for venue in VENUES:
        filename = DEFAULT_INPUTS[(2024, venue)]
        rows, _ = read_csv(input_dir / filename, venue, 2024)
        rows, exact_duplicates = deduplicate(rows)
        eligible = []
        reasons = Counter()
        for row in rows:
            key = candidate_key(row)
            if key in seen_global:
                reasons["cross_venue_duplicate"] += 1
                continue
            seen_global.add(key)
            if row.excerpt_key in val_excerpts:
                reasons["validation_excerpt"] += 1
            elif row.source_key in val_sources:
                reasons["validation_source"] += 1
            elif row.target_key in val_targets:
                reasons["validation_target"] += 1
            elif row.excerpt_key in test_excerpts:
                reasons["test_excerpt"] += 1
            elif row.source_key in test_sources:
                reasons["test_source"] += 1
            elif row.target_key in test_targets:
                reasons["test_target"] += 1
            else:
                eligible.append(row)
        pools[venue] = eligible
        rejected[venue] = {
            "exact_duplicates": exact_duplicates,
            **dict(sorted(reasons.items())),
        }

    candidate_by_key = {
        candidate_key(row): row
        for venue in VENUES
        for row in pools[venue]
    }
    missing_old = sorted(old_keys - set(candidate_by_key))
    if missing_old:
        raise AssertionError(
            f"{len(missing_old)} existing training rows are absent from the eligible 2024 pool"
        )

    existing_by_venue: dict[str, list[Candidate]] = {venue: [] for venue in VENUES}
    for key in old_keys:
        row = candidate_by_key[key]
        existing_by_venue[row.venue].append(row)

    capacities = {venue: len(pools[venue]) for venue in VENUES}
    initial = {venue: len(existing_by_venue[venue]) for venue in VENUES}
    targets = allocate_targets(capacities, initial, args.target_rows)

    selected = []
    for venue in VENUES:
        selected.extend(
            fill_venue(
                existing_by_venue[venue],
                pools[venue],
                targets[venue],
                args.seed + VENUES.index(venue),
            )
        )
    selected.sort(key=lambda row: stable_score(row, args.seed + 100))

    selected_keys = {candidate_key(row) for row in selected}
    if not old_keys <= selected_keys:
        raise AssertionError("Expanded split does not contain every existing RL training row")

    overlaps = {
        "validation_excerpts": len({row.excerpt_key for row in selected} & val_excerpts),
        "validation_sources": len({row.source_key for row in selected} & val_sources),
        "validation_targets": len({row.target_key for row in selected} & val_targets),
        "test_excerpts": len({row.excerpt_key for row in selected} & test_excerpts),
        "test_sources": len({row.source_key for row in selected} & test_sources),
        "test_targets": len({row.target_key for row in selected} & test_targets),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Leakage detected: {overlaps}")

    write_split(output_path, selected, "train", args.id_start)
    report = {
        "output": str(output_path),
        "target_rows": args.target_rows,
        "seed": args.seed,
        "existing_train": str(old_train_path),
        "existing_rows_preserved": len(old_keys),
        "validation_unchanged": str(validation_path),
        "test_unchanged": str(test_path),
        "eligible_capacity_by_venue": capacities,
        "existing_rows_by_venue": initial,
        "selected_target_by_venue": targets,
        "rejected_by_venue": rejected,
        "summary": summarize(selected),
        "overlap": overlaps,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="training_data_collection/output")
    parser.add_argument("--existing-train", default="training_data_collection/splits/train_2024_rl.csv")
    parser.add_argument("--validation", default="training_data_collection/splits/validation_2024_rl.csv")
    parser.add_argument("--test", default="training_data_collection/splits/test_2025_balanced.csv")
    parser.add_argument("--output", default="training_data_collection/splits/train_2024_rl_1k.csv")
    parser.add_argument("--report", default="training_data_collection/splits/train_2024_rl_1k_validation.json")
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--id-start", type=int, default=1_200_000)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"summary": result["summary"], "overlap": result["overlap"]}, indent=2))
