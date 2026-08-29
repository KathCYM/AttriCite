"""Rescore saved validation rollouts against the adjudicated validation labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT_DIR = (
    ROOT
    / "training"
    / "verl"
    / "validation_rollouts"
    / "qwen3_4b_full_grpo_json_fewshot_kl_n8_max5"
)
DEFAULT_SPLIT = ROOT / "training_data_collection" / "splits" / "validation_2024_rl.csv"
DEFAULT_AUDITED_DIR = ROOT / "training_data_collection" / "audited"
DEFAULT_OUTPUT = ROOT / "latex" / "images" / "checkpoint_validation.csv"
SELECTION_RE = re.compile(r"Selection recorded:\s*(.*?)\.\s*Stop now\.", re.DOTALL)


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def title_match(left: str, right: str) -> bool:
    return SequenceMatcher(a=normalized(left), b=normalized(right)).ratio() > 0.8


def load_corrected_validation(split_path: Path, audited_dir: Path) -> list[dict[str, str]]:
    with split_path.open(encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))

    audited_cache: dict[str, dict[str, dict[str, str]]] = {}
    output: list[dict[str, str]] = []
    for row in split_rows:
        source_file = row["source_file"]
        if source_file not in audited_cache:
            with (audited_dir / source_file).open(encoding="utf-8-sig", newline="") as handle:
                audited_cache[source_file] = {
                    str(item["id"]): item for item in csv.DictReader(handle)
                }
        audited = audited_cache[source_file][str(row["original_id"])]
        output.append(
            {
                "id": str(row["id"]),
                "excerpt": row["excerpt"],
                "old_gold": row["target_paper_title"],
                "corrected_gold": audited["target_paper_title"],
            }
        )
    return output


def selected_title(trajectory: str) -> str | None:
    matches = SELECTION_RE.findall(trajectory)
    return matches[-1].strip() if matches else None


def match_record(input_text: str, records: list[dict[str, str]]) -> dict[str, str]:
    matches = [row for row in records if row["excerpt"] in input_text]
    if len(matches) != 1:
        raise ValueError(f"Expected one passage match, found {len(matches)}")
    return matches[0]


def rescore(path: Path, records: list[dict[str, str]]) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(records):
        raise ValueError(f"Expected {len(records)} rows in {path}, found {len(rows)}")

    old_correct = 0
    corrected_correct = 0
    changed_items: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        record = match_record(str(row["input"]), records)
        if record["id"] in seen:
            raise ValueError(f"Duplicate validation record {record['id']} in {path}")
        seen.add(record["id"])
        selected = selected_title(str(row.get("output") or ""))
        old_value = bool(selected and title_match(selected, record["old_gold"]))
        corrected_value = bool(selected and title_match(selected, record["corrected_gold"]))
        old_correct += old_value
        corrected_correct += corrected_value
        if old_value != corrected_value:
            changed_items.append(
                {
                    "id": record["id"],
                    "selected_title": selected,
                    "old_gold": record["old_gold"],
                    "corrected_gold": record["corrected_gold"],
                    "old_correct": old_value,
                    "corrected_correct": corrected_value,
                }
            )

    return {
        "step": int(path.stem),
        "instances": len(rows),
        "old_correct": old_correct,
        "corrected_correct": corrected_correct,
        "old_accuracy": 100.0 * old_correct / len(rows),
        "corrected_accuracy": 100.0 * corrected_correct / len(rows),
        "changed_items": changed_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--audited-dir", type=Path, default=DEFAULT_AUDITED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    records = load_corrected_validation(args.split, args.audited_dir)
    results = [
        rescore(path, records)
        for path in sorted(args.rollout_dir.glob("*.jsonl"), key=lambda item: int(item.stem))
    ]
    if not results:
        raise FileNotFoundError(f"No checkpoint JSONL files found in {args.rollout_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_accuracy"])
        writer.writeheader()
        writer.writerows(
            {"step": row["step"], "validation_accuracy": row["corrected_accuracy"]}
            for row in results
        )

    report_path = args.report or args.output.with_name("checkpoint_validation_rescore.json")
    report_path.write_text(
        json.dumps({"schema_version": 1, "checkpoints": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    best = max(results, key=lambda row: (float(row["corrected_accuracy"]), int(row["step"])))
    print(f"Rescored {len(results)} checkpoints over {len(records)} validation records")
    print(f"Best corrected checkpoint: step {best['step']} ({best['corrected_accuracy']:.2f}%)")
    print(f"Wrote {args.output} and {report_path}")


if __name__ == "__main__":
    main()
