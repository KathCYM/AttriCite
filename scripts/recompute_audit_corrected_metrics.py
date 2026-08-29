"""Recompute test accuracy without mutating immutable evaluation logs."""

from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path


AUDITED_IDS = {
    "2000181": "A008",
    "2000283": "A012",
    "2000238": "A018",
    "2000033": "A032",
    "2000282": "A034",
    "2000115": "A057",
    "2000201": "A058",
    "2000124": "A065",
    "2000255": "A071",
    "2000180": "A084",
}
EXCLUDED_IDS = {"2000252": "A083"}


def is_correct(selected: dict | None, gold: str) -> bool:
    if not selected or not selected.get("title"):
        return False
    return SequenceMatcher(a=selected["title"].lower(), b=gold.lower()).ratio() > 0.8


def load_gold(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]): row["target_paper_title"] for row in csv.DictReader(handle)}


def analyze(path: Path, gold: dict[str, str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    all_results = payload["results"]
    results = [row for row in all_results if str(row["id"]) not in EXCLUDED_IDS]
    old_correct = sum(bool(row.get("is_correct")) for row in results)
    new_values = {str(row["id"]): is_correct(row.get("selected"), gold[str(row["id"])]) for row in results}
    effects = []
    for row in results:
        row_id = str(row["id"])
        if row_id not in AUDITED_IDS:
            continue
        effects.append(
            {
                "audit_id": AUDITED_IDS[row_id],
                "dataset_id": row_id,
                "selected_paper_id": (row.get("selected") or {}).get("paperId"),
                "selected_title": (row.get("selected") or {}).get("title"),
                "old_correct": bool(row.get("is_correct")),
                "corrected_correct": new_values[row_id],
                "corrected_gold": gold[row_id],
            }
        )
    new_correct = sum(new_values.values())
    return {
        "file": path.as_posix(),
        "instances": len(results),
        "old_correct": old_correct,
        "corrected_correct": new_correct,
        "old_accuracy": old_correct / len(results),
        "corrected_accuracy": new_correct / len(results),
        "accuracy_delta": (new_correct - old_correct) / len(results),
        "audited_record_effects": effects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    gold = load_gold(args.split)
    report = {"schema_version": 1, "runs": [analyze(path, gold) for path in args.files]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote corrected metrics for {len(report['runs'])} runs to {args.output}")


if __name__ == "__main__":
    main()
