import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "training_data_collection" / "splits" / "test_2025_balanced.csv"
OUTPUT = ROOT / "outputs" / "citation_audit_100" / "audit_sample.json"
SEED = "citealign-audit-20260828"


def stable_key(value: object) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def excerpt_text(value: object, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit].rstrip() + " [truncated; inspect source paper]"


def load_audit_records() -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    output_dir = ROOT / "training_data_collection" / "output"
    for path in sorted(output_dir.glob("*2025_candidates.audit.jsonl")):
        if "biomed" in path.name or "bionlp" in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    source_file = path.name.removesuffix(".audit.jsonl") + ".csv"
                    records[(source_file, int(row["id"]))] = row
    return records


def main() -> None:
    split = pd.read_csv(SPLIT)
    if len(split) != 299 or sorted(split["venue"].value_counts()) != [59, 60, 60, 60, 60]:
        raise ValueError("Expected the 299-item post-audit test split")

    audit_records = load_audit_records()
    selected = []
    for venue, group in split.groupby("venue", sort=True):
        rows = group.to_dict(orient="records")
        rows.sort(key=lambda row: stable_key(row["id"]))
        selected.extend(rows[:20])

    selected.sort(key=lambda row: stable_key(f"audit-order:{row['id']}"))
    output_rows = []
    for index, row in enumerate(selected, start=1):
        original_id = int(row["original_id"])
        audit = audit_records.get((row["source_file"], original_id))
        if audit is None:
            raise KeyError(
                f"Missing construction audit record for source_file={row['source_file']}, "
                f"original_id={original_id}"
            )
        resolved_id = audit.get("resolved_paper_id", "")
        output_rows.append(
            {
                "audit_id": f"A{index:03d}",
                "dataset_id": int(row["id"]),
                "venue": row["venue"],
                "excerpt": row["excerpt"],
                "citation_marker": audit.get("citation_marker", ""),
                "raw_reference": excerpt_text(audit.get("raw_reference", "")),
                "recorded_target_title": row["target_paper_title"],
                "resolved_title": audit.get("resolved_title", ""),
                "resolved_paper_id": resolved_id,
                "semantic_scholar_url": (
                    f"https://www.semanticscholar.org/paper/{resolved_id}" if resolved_id else ""
                ),
                "target_paper_url": "" if pd.isna(row["target_paper_url"]) else row["target_paper_url"],
                "source_paper_title": row["source_paper_title"],
                "source_paper_url": row["source_paper_url"],
                "resolution_query": audit.get("resolution_query", ""),
                "resolution_score": audit.get("resolution_score", None),
                "discoverability_query": audit.get("discoverability_query", ""),
                "discoverability_rank": audit.get("discoverability_rank", None),
            }
        )

    counts = pd.Series([row["venue"] for row in output_rows]).value_counts().to_dict()
    if len(output_rows) != 100 or any(counts.get(v) != 20 for v in split["venue"].unique()):
        raise ValueError(f"Unexpected sample composition: {counts}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(output_rows), "venue_counts": counts, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
