from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from training_data_collection.collect_numeric_cs_candidates import make_csv_dict_writer


YEAR_IN_URL = re.compile(r"/paper/(\d{4})/")
BACKUP_SUFFIX = ".pre_year_repair.bak"


def url_year(url: object) -> int | None:
    match = YEAR_IN_URL.search(str(url))
    return int(match.group(1)) if match else None


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = make_csv_dict_writer(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)


def repair_year(output_dir: Path, year: int, dry_run: bool) -> tuple[int, int]:
    csv_path = output_dir / f"iclr{year}_candidates.csv"
    audit_path = output_dir / f"iclr{year}_candidates.audit.jsonl"
    state_path = output_dir / f"iclr{year}_candidates.csv.state.json"
    fields, rows = load_csv(csv_path)
    audits = load_jsonl(audit_path)

    kept_rows = [row for row in rows if url_year(row.get("source_paper_url")) == year]
    kept_ids = {str(row.get("id")) for row in kept_rows}
    kept_audits = [row for row in audits if str(row.get("id")) in kept_ids]
    removed = len(rows) - len(kept_rows)

    if dry_run:
        return removed, len(kept_rows)

    for path in (csv_path, audit_path, state_path):
        if path.exists():
            backup_once(path)
    write_csv(csv_path, fields, kept_rows)
    write_jsonl(audit_path, kept_audits)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        processed = state.get("processed_source_paper_keys", [])
        processed = [key for key in processed if url_year(key) in {None, year}]
        state["processed_source_paper_keys"] = processed
        state["processed_source_paper_count"] = len(processed)
        state["dataset_row_count"] = len(kept_rows)
        state["audit_row_count"] = len(kept_audits)
        state["next_id"] = max((int(row["id"]) for row in kept_rows), default=99999) + 1
        if url_year(state.get("last_processed_source_paper_url")) not in {None, year}:
            state["last_processed_source_paper_title"] = None
            state["last_processed_source_paper_url"] = None
        temporary = state_path.with_name(state_path.name + ".repair.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        temporary.replace(state_path)

    return removed, len(kept_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove ICLR candidate rows saved under the wrong conference year."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("training_data_collection/output")
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for year in args.years:
        removed, kept = repair_year(args.output_dir, year, args.dry_run)
        action = "would remove" if args.dry_run else "removed"
        print(f"ICLR {year}: {action} {removed} wrong-year row(s), kept {kept}")


if __name__ == "__main__":
    main()
