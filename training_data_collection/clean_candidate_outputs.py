from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from pathlib import Path

from training_data_collection.collect_numeric_cs_candidates import (
    DATASET_COLUMNS,
    looks_like_reference_entry_leakage,
    make_csv_dict_writer,
)


BACKUP_SUFFIX = ".pre_reference_cleanup.bak"


def load_audit_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.replace("\x00", "")
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def is_reference_leakage(row: dict[str, object]) -> bool:
    return looks_like_reference_entry_leakage(
        str(row.get("raw_sentence", "")),
        str(row.get("raw_reference", "")),
    )


def backup_once(path: Path) -> None:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    temporary_path = path.with_name(path.name + ".cleanup.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = make_csv_dict_writer(handle, fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    temporary_path = path.with_name(path.name + ".cleanup.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary_path.replace(path)


def clean_pair(
    csv_path: Path, audit_path: Path, dry_run: bool
) -> tuple[int, int, list[str], int]:
    audit_rows = load_audit_rows(audit_path)
    rejected_rows = [row for row in audit_rows if is_reference_leakage(row)]
    rejected_ids = {
        str(row["id"])
        for row in rejected_rows
        if "id" in row
    }
    examples = [str(row.get("raw_sentence", "")) for row in rejected_rows[:3]]

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        raw_csv = handle.read()
        nul_count = raw_csv.count("\x00")
        # Interrupted writes can leave NUL bytes that Python's CSV parser rejects.
        reader = csv.DictReader(io.StringIO(raw_csv.replace("\x00", ""), newline=""))
        fieldnames = list(reader.fieldnames or DATASET_COLUMNS)
        csv_rows = list(reader)

    kept_csv_rows = [row for row in csv_rows if str(row.get("id", "")) not in rejected_ids]
    kept_audit_rows = [
        row for row in audit_rows if str(row.get("id", "")) not in rejected_ids
    ]
    removed = len(csv_rows) - len(kept_csv_rows)
    if dry_run or (removed == 0 and nul_count == 0):
        return removed, len(kept_csv_rows), examples, nul_count

    backup_once(csv_path)
    backup_once(audit_path)
    write_csv(csv_path, kept_csv_rows, fieldnames)
    write_audit(audit_path, kept_audit_rows)

    state_path = csv_path.with_name(csv_path.name + ".state.json")
    if state_path.exists():
        backup_once(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["dataset_row_count"] = len(kept_csv_rows)
        state["audit_row_count"] = len(kept_audit_rows)
        temporary_path = state_path.with_name(state_path.name + ".cleanup.tmp")
        temporary_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(state_path)

    return removed, len(kept_csv_rows), examples, nul_count


def find_candidate_pairs(output_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for csv_path in sorted(output_dir.glob("*.csv")):
        audit_path = csv_path.with_suffix(".audit.jsonl")
        if audit_path.exists():
            pairs.append((csv_path, audit_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove bibliography/reference-entry leakage from candidate outputs."
    )
    parser.add_argument(
        "--output_dir", default="training_data_collection/output", type=Path
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    total_removed = 0
    pairs = find_candidate_pairs(args.output_dir)
    for csv_path, audit_path in pairs:
        removed, kept, examples, nul_count = clean_pair(
            csv_path, audit_path, args.dry_run
        )
        total_removed += removed
        action = "would remove" if args.dry_run else "removed"
        print(f"{csv_path.name}: {action} {removed}, kept {kept}")
        if nul_count:
            repair_action = "would strip" if args.dry_run else "stripped"
            print(f"  {repair_action} {nul_count} NUL byte(s)")
        for example in examples:
            print(f"  example: {example}")
    print(f"Processed {len(pairs)} candidate file(s); total removed: {total_removed}")


if __name__ == "__main__":
    main()
