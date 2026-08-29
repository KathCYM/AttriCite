"""Create a deterministic, threshold-focused audit sample of test title pairs."""

from __future__ import annotations

import csv
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "training_data_collection" / "splits" / "test_2025_balanced.csv"
OUT = ROOT / "outputs" / "title_match_audit_100.csv"
SEED = "title-match-audit-20260829"

FILES = [
    *(ROOT / "results_v2/results/temperature_sampling_validation_test").glob("base_test_temp0p7_run[1-3].json"),
    *(ROOT / "results_v2/results/temperature_sampling_validation_test").glob("step475_test_temp0p7_run[1-3].json"),
    *(ROOT / "results_v2/results/together_temperature_sampling").glob("gpt_oss_20b_test_temp0p7_run[1-3].json"),
    *(ROOT / "results_v2/results/together_temperature_sampling").glob("gemma4_31b_test_temp0p7_run[1-3].json"),
    *(ROOT / "results_v2/results/openai_temperature_sampling").glob("gpt54mini_test_temp0p7_run[1-3].json"),
    ROOT / "results_v2/results/anthropic_temperature_sampling/claude_haiku45_test2025_temp0p7_run1.json",
]


def stable_key(text: str) -> str:
    return hashlib.sha256(f"{SEED}|{text}".encode()).hexdigest()


def main() -> None:
    prior: dict[tuple[str, str], tuple[str, str]] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                prior[(row["selected_title"], row["target_title"])] = (
                    row.get("manual_equivalent", ""), row.get("notes", "")
                )
    with GOLD.open(encoding="utf-8-sig", newline="") as handle:
        gold = {str(row["id"]): row["target_paper_title"] for row in csv.DictReader(handle)}

    pairs: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(FILES):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["results"]:
            row_id = str(row["id"])
            if row_id not in gold:
                continue
            selected = str((row.get("selected") or {}).get("title") or "").strip()
            if not selected:
                continue
            target = gold[row_id]
            score = SequenceMatcher(None, selected.casefold(), target.casefold()).ratio()
            key = (selected, target)
            record = pairs.setdefault(key, {
                "selected_title": selected,
                "target_title": target,
                "similarity": score,
                "threshold_decision": score > 0.8,
                "occurrences": 0,
                "example_ids": set(),
                "files": set(),
            })
            record["occurrences"] = int(record["occurrences"]) + 1
            record["example_ids"].add(row_id)
            record["files"].add(path.name)

    records = list(pairs.values())
    accepted_nonexact = [r for r in records if 0.8 < float(r["similarity"]) < 1.0]
    near_negative = [r for r in records if 0.7 <= float(r["similarity"]) <= 0.8]
    # Put every borderline pair first, then fill both sides of the decision
    # boundary. Stable hashes make the audit exactly reproducible.
    bands = [
        [r for r in records if 0.70 <= float(r["similarity"]) <= 0.90],
        [r for r in records if float(r["similarity"]) > 0.90],
        [r for r in records if float(r["similarity"]) < 0.70],
    ]
    quotas = [50, 25, 25]
    sample: list[dict[str, object]] = []
    used: set[tuple[str, str]] = set()
    for band, quota in zip(bands, quotas):
        band.sort(key=lambda r: (abs(float(r["similarity"]) - 0.8), stable_key(str(r["selected_title"]) + str(r["target_title"]))))
        for record in band:
            key = (str(record["selected_title"]), str(record["target_title"]))
            if key not in used and len([x for x in sample if x.get("band") == id(band)]) < quota:
                record = dict(record)
                record["band"] = id(band)
                sample.append(record)
                used.add(key)
            if len([x for x in sample if x.get("band") == id(band)]) == quota:
                break

    # A narrow boundary band may contain fewer than its quota. Fill the
    # remainder with unused pairs closest to the decision boundary.
    remaining = [
        record for record in records
        if (str(record["selected_title"]), str(record["target_title"])) not in used
    ]
    remaining.sort(key=lambda r: (abs(float(r["similarity"]) - 0.8), stable_key(str(r["selected_title"]) + str(r["target_title"]))))
    for record in remaining[: 100 - len(sample)]:
        sample.append(dict(record))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        fields = ["audit_index", "similarity", "threshold_decision", "selected_title", "target_title", "occurrences", "example_ids", "files", "manual_equivalent", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(sample, 1):
            manual_equivalent, notes = prior.get(
                (str(record["selected_title"]), str(record["target_title"])), ("", "")
            )
            writer.writerow({
                "audit_index": index,
                "similarity": f'{float(record["similarity"]):.6f}',
                "threshold_decision": record["threshold_decision"],
                "selected_title": record["selected_title"],
                "target_title": record["target_title"],
                "occurrences": record["occurrences"],
                "example_ids": ";".join(sorted(record["example_ids"])),
                "files": ";".join(sorted(record["files"])),
                "manual_equivalent": manual_equivalent,
                "notes": notes,
            })
    print(f"Wrote {len(sample)} unique title pairs to {OUT}")
    print(f"All outputs contain {len(accepted_nonexact)} unique accepted non-exact pairs and {len(near_negative)} unique near-threshold negatives.")


if __name__ == "__main__":
    main()
